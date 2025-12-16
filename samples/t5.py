import os
import argparse
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block
from datasets import load_dataset
from functools import partial

# [서브 루틴 1] 메모리 최적화 설정 생성
def get_fsdp_config(args):
    # 1. Mixed Precision 설정 (bf16, fp16, fp32)
    if args.precision == "bf16":
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    elif args.precision == "fp16":
        mp_policy = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    else:
        mp_policy = None

    # 2. 샤딩 전략 설정 (FULL_SHARD가 메모리를 가장 많이 아낌)
    sharding_strategy = ShardingStrategy.FULL_SHARD if args.full_shard else ShardingStrategy.NO_SHARD
    
    return mp_policy, sharding_strategy

# [서브 루틴 2] 체크포인트 저장
def save_checkpoint(model, optimizer, epoch, path, rank):
    state_dict = {"model": model, "optimizer": optimizer, "epoch": torch.tensor(epoch)}
    dcp.save(state_dict, checkpoint_id=path)
    if rank == 0: print(f"--- Saved: {path} ---")

# [학습 서브 루틴]
def train(args, rank, device):
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    
    # 메모리 최적화 설정 적용
    mp_policy, sharding_strategy = get_fsdp_config(args)

    # FSDP 래핑
    model = FSDP(
        model,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=device if device.type == "cuda" else None
    )

    # 3. Activation Checkpointing (그래디언트 체크포인팅) 적용
    # 메모리 사용량을 획기적으로 줄이지만 연산 속도가 약간 느려짐
    if args.use_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, T5Block)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    # 데이터 및 옵티마이저 설정
    raw_ds = load_dataset("billsum", split=f"train[:{args.train_size}]")
    tokenized_ds = raw_ds.map(lambda ex: tokenizer(["summarize: " + d for d in ex["text"]], max_length=512, truncation=True, padding="max_length"), batched=True).with_format("torch")
    train_loader = DataLoader(tokenized_ds, batch_size=args.batch_size, sampler=DistributedSampler(tokenized_ds))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and rank == 0: print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | Precision: {args.precision}")

    save_checkpoint(model, optimizer, args.epochs, f"checkpoint_{args.precision}", rank)

def main():
    parser = argparse.ArgumentParser()
    # 기본 파라미터
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--model_id", type=str, default="google-t5/t5-small")
    
    # --- 메모리/속도 최적화 파라미터 ---
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], 
                        help="연산 정밀도 선택 (bf16 권장)")
    parser.add_argument("--use_checkpointing", action="store_true", 
                        help="Activation Checkpointing 활성화 (메모리 절약)")
    parser.add_argument("--full_shard", type=bool, default=True, 
                        help="FSDP Full Sharding 여부 (True 시 메모리 최소화)")
    
    args = parser.parse_args()

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)

    train(args, rank, device)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
