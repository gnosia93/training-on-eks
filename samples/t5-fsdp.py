"""
torchrun --nnodes=1 --nproc_per_node=1 train_fsdp.py --train_size 1000 --precision fp32
torchrun --nnodes=1 --nproc_per_node=8 train_fsdp.py --batch_size 16 --precision bf16 --use_checkpointing
torchrun --nnodes=2 \
         --nproc_per_node=8 \
         --rdzv_id=101 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=MASTER_IP:29500 \
         train_fsdp.py --precision bf16

샤딩 전략별 특징 
- FULL_SHARD: 파라미터, 그라디언트, 옵티마이저 상태를 모든 GPU에 나눕니다. 메모리를 가장 많이 아낄 수 있어 LLM 학습 시 기본으로 사용됩니다.
- SHARD_GRAD_OP: 그라디언트와 옵티마이저 상태만 샤딩합니다. 파라미터는 복제되어 있으므로 통신량은 줄어들지만 메모리 사용량은 FULL_SHARD보다 큽니다. (ZeRO-2와 유사)
- NO_SHARD: 데이터를 나누어 학습하지만 모델 파라미터는 모든 GPU가 가집니다. 일반 DDP(Distributed Data Parallel)와 동일한 동작을 합니다.
- HYBRID_SHARD: 노드 내부 GPU끼리는 FULL_SHARD를 수행하고, 노드 간에는 모델을 복제합니다. 통신 효율과 메모리 효율의 균형을 맞출 때 사용합니다.

데이터셋
billsum 데이터셋은 미국 의회 및 캘리포니아주 의회에서 발의된 법안(Bills)의 원문과 이를 요약한 요약문을 한데 모아놓은 대표적인 텍스트 요약(Summarization) 데이터셋입니다. [1]
2025년 현재까지도 모델의 긴 문맥 이해 능력과 전문적인 텍스트 요약 성능을 측정하는 벤치마크로 널리 사용되고 있습니다
데이터 크기:
- US Training: 약 18,949건 (모델 학습용) [2]
- US Test: 약 3,269건 (성능 평가용) [2]
- California Test: 약 1,237건 (학습되지 않은 도메인에 대한 일반화 능력 테스트용) [1]
텍스트 길이: 평균적으로 법안 원문은 약 5,000단어 내외이며, 요약문은 약 200단어 내외입니다. [2]
"""
import os
import argparse
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,  # 샤딩 전략 라이브러리
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block
from datasets import load_dataset

# [서브 루틴 1] DCP 체크포인트 저장
def save_checkpoint(model, optimizer, epoch, path, rank):
    state_dict = {"model": model, "optimizer": optimizer, "epoch": torch.tensor(epoch)}
    dcp.save(state_dict, checkpoint_id=path)
    if rank == 0: print(f"--- 체크포인트 저장 완료: {path} ---")

# [서브 루틴 2] 실제 학습 루틴 (train)
def train(args, rank, device):
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    
    # 1. Mixed Precision 설정
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) if args.precision == "bf16" else None
    
    # 2. Sharding Strategy 선택 (문자열 매핑)
    # 2025년 기준 FSDP의 4대 전략
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,       # 파라미터+그라디언트+옵티마이저 모두 샤딩 (메모리 최소화)
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP, # 그라디언트+옵티마이저만 샤딩 (ZeRO-2와 유사)
        "NO_SHARD": ShardingStrategy.NO_SHARD,           # 샤딩 없음 (DDP와 동일)
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD    # 노드 내 FULL_SHARD + 노드 간 복제 (통신 효율)
    }
    selected_strategy = strategy_map.get(args.sharding_strategy, ShardingStrategy.FULL_SHARD)

    # 3. FSDP 래핑
    model = FSDP(
        model,
        mixed_precision=mp_policy,
        sharding_strategy=selected_strategy, # 파라미터로 받은 전략 적용
        device_id=device if device.type == "cuda" else None,
        cpu_offload=CPUOffload(offload_params=args.use_offload),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if args.use_prefetch else None,
        forward_prefetch=args.use_prefetch,
        limit_all_gathers=True
    )

    if args.use_checkpointing:
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda m: isinstance(m, T5Block))

    # 데이터 로딩
    dataset = load_dataset("billsum", split=f"train[:{args.train_size}]")
    tokenized_ds = dataset.map(lambda ex: tokenizer(["summarize: " + d for d in ex["text"]], max_length=512, truncation=True, padding="max_length"), batched=True).with_format("torch")
    
    sampler = DistributedSampler(tokenized_ds, shuffle=True)
    loader = DataLoader(tokenized_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            is_accumulation_step = (i + 1) % args.grad_acc_steps != 0
            with model.no_sync() if is_accumulation_step else torch.enable_grad():
                batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
                loss = model(**batch).loss / args.grad_acc_steps
                loss.backward()

            if (i + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 10 == 0 and rank == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item() * args.grad_acc_steps:.4f} | Strategy: {args.sharding_strategy}")

    save_checkpoint(model, optimizer, args.epochs, f"checkpoint_{args.sharding_strategy}", rank)

# [메인 함수]
def main():
    parser = argparse.ArgumentParser()
    # 기본 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--model_id", type=str, default="google-t5/t5-small")
    
    # --- 핵심 분산 전략 파라미터 ---
    parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD", 
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
                        help="FSDP 샤딩 전략 선택")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_offload", action="store_true")
    parser.add_argument("--use_checkpointing", action="store_true")
    parser.add_argument("--use_prefetch", action="store_true")
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)

    train(args, rank, device)
    dist.destroy_group()

if __name__ == "__main__":
    main()
