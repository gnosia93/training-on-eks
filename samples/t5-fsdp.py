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

2025년 기준, 분산 학습 환경(FSDP)에서 데이터 로딩(Data Loading) 프로세스는 단순히 파일을 읽는 것을 넘어, 성능 최적화와 병목 현상 제거를 위해 다음과 같은 복잡한 단계들을 수행합니다. [1]

1. 데이터셋 로드 및 슬라이싱 (Dataset Loading & Slicing)
원격 저장소 다운로드: datasets 라이브러리를 통해 Hugging Face나 클라우드 저장소에서 데이터를 로컬 캐시로 가져옵니다. [1]
스트리밍/슬라이싱: 10만 개와 같이 대용량 데이터를 처리할 때, 메모리에 한 번에 올리지 않고 필요한 부분만 선택(select)하거나 필요한 시점에 불러오는 스트리밍 방식을 사용하여 RAM 점유율을 낮춥니다. [1, 2]

2. 토큰화 및 전처리 (Tokenization & Preprocessing)
텍스트 수치화: T5Tokenizer를 사용하여 자연어 문장을 모델이 이해할 수 있는 정수(Token ID)로 변환합니다. [3]
패딩 및 잘라내기(Padding & Truncation): 배치(Batch) 내 문장 길이를 일정하게 맞추기 위해 최대 길이에 맞춰 문장을 자르거나 부족한 부분을 0으로 채웁니다. [3]

3. 데이터 분산 (Distributed Sampling)
분할(Sharding): DistributedSampler가 매우 중요한 역할을 합니다. 전체 10만 개 데이터를 전체 GPU 개수(World Size)로 나누어, 각 GPU가 서로 겹치지 않는 고유한 데이터 부분만 학습하도록 보장합니다. [1]
셔플링(Shuffling): 각 에포크마다 데이터를 섞어 모델이 데이터 순서를 외우지 못하게 방지합니다. [1]

4. 병렬 데이터 로딩 (Multi-process Loading)
Worker 활용: num_workers 설정을 통해 여러 개의 CPU 프로세스를 띄웁니다. GPU가 현재 배치를 계산하는 동안, CPU는 백그라운드에서 다음 배치를 미리 읽고 전처리하여 GPU가 노는 시간(Idle time)을 최소화합니다. [3]

5. 메모리 최적화 (Memory Optimization)
페이지 고정(Memory Pinning): pin_memory=True를 설정하면 CPU의 데이터를 '페이지 고정' 영역에 할당합니다. 이는 일반 RAM보다 CPU에서 GPU로 데이터를 전송하는 속도를 2배 이상 향상시킵니다. [4]
비동기 전송(Non-blocking Transfer): pin_memory와 연동되어 데이터를 GPU로 보내는 동안 파이썬 코드가 기다리지 않고 다음 연산을 미리 준비할 수 있게 합니다. [4]

요약하자면
데이터 로딩은 "CPU가 쉬지 않고 데이터를 가공해서 가장 빠른 경로(Pin Memory)로 GPU에게 끊임없이 먹이를 주는 과정"이라고 이해하시면 됩니다. 이 과정에서 DistributedSampler는 중복 학습을 방지하고, num_workers와 pin_memory는 전송 속도를 책임집니다. [1, 3, 4]
"""
import os
import argparse
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
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
    # 저장할 데이터 구조 정의
    state_dict = {
        "model": model, 
        "optimizer": optimizer, 
        "epoch": torch.tensor(epoch)
    }
    # DCP를 이용한 분산 저장
    dcp.save(state_dict, checkpoint_id=path)
    if rank == 0:
        print(f"--- 체크포인트 저장 완료: {path} (Epoch: {epoch}) ---")

# [서브 루틴 2] DCP 체크포인트 로드
def load_checkpoint(model, optimizer, path):
    # 불러올 데이터를 담을 구조 정의 (저장 시와 동일해야 함)
    state_dict = {
        "model": model,
        "optimizer": optimizer,
        "epoch": torch.tensor(0)  # 임시 값
    }
    
    # 해당 경로(폴더)가 존재하고 비어있지 않으면.
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        dcp.load(state_dict, checkpoint_id=path)
        return state_dict["epoch"].item()
    return None

# [서브 루틴 3] 실제 학습 루틴 (train)
def train(args, rank, device):
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    
    # 1. Mixed Precision 설정
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) if args.precision == "bf16" else None
    
    # 2. Sharding Strategy 선택
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD
    }
    selected_strategy = strategy_map.get(args.sharding_strategy, ShardingStrategy.FULL_SHARD)

    # 3. FSDP 래핑
    model = FSDP(
        model,
        mixed_precision=mp_policy,
        sharding_strategy=selected_strategy,
        device_id=device if device.type == "cuda" else None,
        cpu_offload=CPUOffload(offload_params=args.use_offload),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if args.use_prefetch else None,
        forward_prefetch=args.use_prefetch,
        limit_all_gathers=True
    )

    if args.use_checkpointing:
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda m: isinstance(m, T5Block))

    # 데이터 로딩
    dataset = load_dataset("billsum", split=f"train[:{args.train_size}]", trust_remote_code=True)
    tokenized_ds = dataset.map(lambda ex: tokenizer(["summarize: " + d for d in ex["text"]], max_length=512, truncation=True, padding="max_length"), batched=True).with_format("torch")
    
    sampler = DistributedSampler(tokenized_ds, shuffle=True)
    loader = DataLoader(tokenized_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- [재시작 로직 추가] ---
    checkpoint_path = f"checkpoint_{args.sharding_strategy}"
    start_epoch = 0
    
    # 체크포인트가 존재하면 로드
    last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if last_epoch is not None:
        start_epoch = last_epoch + 1
        if rank == 0:
            print(f"--- 체크포인트를 로드했습니다. {start_epoch} 에폭부터 재시작합니다. ---")
    else:
        if rank == 0:
            print("--- 기존 체크포인트가 없습니다. 처음부터 학습을 시작합니다. ---")
    # -----------------------

    model.train()
    for epoch in range(start_epoch, args.epochs):
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

        # 에폭 종료 후 체크포인트 저장
        save_checkpoint(model, optimizer, epoch, checkpoint_path, rank)

# [메인 함수]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5) # 테스트를 위해 에폭 상향
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--model_id", type=str, default="google-t5/t5-small")
    
    parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD", 
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"])
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_offload", action="store_true")
    parser.add_argument("--use_checkpointing", action="store_true")
    parser.add_argument("--use_prefetch", action="store_true")
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()

    # 분산 학습 환경 초기화
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): 
        torch.cuda.set_device(local_rank)

    train(args, rank, device)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
