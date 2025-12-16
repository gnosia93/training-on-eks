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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
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

def save_checkpoint(model, optimizer, epoch, path, rank):
    state_dict = {"model": model, "optimizer": optimizer, "epoch": torch.tensor(epoch)}
    dcp.save(state_dict, checkpoint_id=path)
    if rank == 0: print(f"--- 체크포인트 저장 완료: {path} ---")

def train(args, rank, device):
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) if args.precision == "bf16" else None
    
    model = FSDP(
        model,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device if device.type == "cuda" else None,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if args.use_prefetch else None,
        forward_prefetch=args.use_prefetch,
        limit_all_gathers=True
    )

    if args.use_checkpointing:
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda m: isinstance(m, T5Block))

    # 데이터 로드
    dataset = load_dataset("billsum", split=f"train[:{args.train_size}]")
    tokenized_ds = dataset.map(lambda ex: tokenizer(["summarize: " + d for d in ex["text"]], max_length=512, truncation=True, padding="max_length"), batched=True).with_format("torch")
    
    sampler = DistributedSampler(tokenized_ds, shuffle=True)
    
    # --- DataLoader에 Pin Memory 적용 ---
    loader = DataLoader(
        tokenized_ds, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=args.num_workers,  # 데이터 로딩 병렬화
        pin_memory=args.pin_memory,    # CPU -> GPU 전송 최적화
        pin_memory_device=str(device) if args.pin_memory and device.type == "cuda" else "" # PyTorch 최신 기능
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        
        for i, batch in enumerate(loader):
            is_accumulation_step = (i + 1) % args.grad_acc_steps != 0
            
            with model.no_sync() if is_accumulation_step else torch.enable_grad():
                # Pin Memory 사용 시 non_blocking=True로 설정하면 전송과 연산을 겹칠 수 있음
                batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
                loss = model(**batch).loss / args.grad_acc_steps
                loss.backward()

            if (i + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 10 == 0 and rank == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item() * args.grad_acc_steps:.4f}")

    save_checkpoint(model, optimizer, args.epochs, "checkpoint_final", rank)

def main():
    parser = argparse.ArgumentParser()
    # 기본 및 최적화 파라미터
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--model_id", type=str, default="google-t5/t5-small")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--use_checkpointing", action="store_true")
    parser.add_argument("--use_prefetch", action="store_true")
    
    # --- Pin Memory 및 Worker 설정 추가 ---
    parser.add_argument("--pin_memory", action="store_false", help="Pin Memory 비활성화 시 사용 (기본은 활성)")
    parser.set_defaults(pin_memory=True) # 기본값을 True로 설정
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 병렬 워커 수")
    
    args = parser.parse_args()

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)

    train(args, int(os.environ.get("RANK", 0)), device)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
