"""
torchrun --nnodes=1 --nproc_per_node=1 train_fsdp.py --train_size 1000 --precision fp32
torchrun --nnodes=1 --nproc_per_node=8 train_fsdp.py --batch_size 16 --precision bf16 --use_checkpointing
torchrun --nnodes=2 \
         --nproc_per_node=8 \
         --rdzv_id=101 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=MASTER_IP:29500 \
         train_fsdp.py --precision bf16
"""
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

# [서브 루틴 1] DCP 체크포인트 저장
def save_checkpoint(model, optimizer, epoch, path, rank):
    state_dict = {"model": model, "optimizer": optimizer, "epoch": torch.tensor(epoch)}
    dcp.save(state_dict, checkpoint_id=path)
    if rank == 0:
        print(f"--- 체크포인트 저장 완료: {path} ---")

# [서브 루틴 2] 실제 학습 루틴
def train(args, rank, device):
    # 모델 및 토크나이저 로드
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    
    # 정밀도(Precision) 설정
    mp_policy = None
    if args.precision == "bf16":
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    elif args.precision == "fp16":
        mp_policy = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16)

    # FSDP 래핑
    model = FSDP(
        model,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device if device.type == "cuda" else None
    )

    # 메모리 절약을 위한 Activation Checkpointing
    if args.use_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, T5Block)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    # 데이터 로드 (학습용만 사용)
    dataset = load_dataset("billsum", split=f"train[:{args.train_size}]")
    def preprocess(ex):
        inputs = tokenizer(["summarize: " + d for d in ex["text"]], max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(text_target=ex["summary"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_ds = dataset.map(preprocess, batched=True).with_format("torch")
    sampler = DistributedSampler(tokenized_ds, shuffle=True)
    loader = DataLoader(tokenized_ds, batch_size=args.batch_size, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0 and rank == 0:
                print(f"에포크 {epoch} | 단계 {i} | 손실 {loss.item():.4f} | 정밀도 {args.precision}")

    save_checkpoint(model, optimizer, args.epochs, f"checkpoint_{args.precision}", rank)

# [메인 함수]
def main():
    parser = argparse.ArgumentParser()
    # 학습 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--model_id", type=str, default="google-t5/t5-small")
    
    # 메모리 및 성능 최적화
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_checkpointing", action="store_true")
    
    args = parser.parse_args()

    # torchrun이 주입한 환경 변수 읽기
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 분산 환경 초기화
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"--- 2025 FSDP 훈련 시작 ---")
        print(f"월드 사이즈(총 프로세스): {world_size}")
        print(f"백엔드: {backend} | 정밀도: {args.precision}")

    train(args, rank, device)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
