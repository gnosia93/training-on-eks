import argparse
import os
import logging
import time
import datetime
from datetime import timedelta
import torch
import transformers 
from transformers import AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
from datasets import load_dataset
import torch.distributed as dist  
import deepspeed  
import gc
import psutil

# 1. Hugging Face 타임아웃 연장 (데이터셋 로딩 에러 방지)
os.environ["HF_HUB_READ_TIMEOUT"] = "300"
# 2. 메모리 파편화 방지 (OOM 및 Cache Flush 방지)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 3. 랑데뷰 타임아웃 5분 
os.environ["RDZV_TIMEOUT"] = "300"        

def flush_gpu_memory():
    """강제로 GPU 메모리 캐시를 비우고 가비지 컬렉션을 수행합니다."""
    # 파이썬 수준의 객체 정리
    gc.collect()
    # PyTorch 수준의 캐시 비우기
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 현재 할당된 메모리 정보 초기화 (선택 사항)
        torch.cuda.reset_peak_memory_stats()
    print(f"[Rank {os.environ.get('RANK', '0')}] GPU Memory Flushed.")

# transformers 라이브러리의 로그 레벨을 INFO로 설정하여 학습 과정(Loss 등)을 확인합니다.
transformers.utils.logging.set_verbosity_info()
logger = logging.getLogger(__name__)

def main():
    flush_gpu_memory()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model ID")
    parser.add_argument("--ds_config", type=str, default="llama-3-8b-stage3.json", help="Path to DeepSpeed config file")
    args = parser.parse_args()

    # 랑데뷰(초기화) 필수
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    model_name = "meta-llama/Llama-3.2-1B"
    ds_config_path = "llama-3-1b-stage1.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", # 메모리 효율을 위해 SDPA 유지
    )   
    model.to("cuda") 
        
    # 데이터셋 로드 및 전처리 (전처리 시 CPU 메모리 주의)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")   
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir="/data/fsx",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
    #    max_steps=50,
        num_train_epochs=1.0, 
        bf16=True,
        deepspeed=ds_config_path, 
        gradient_checkpointing=True, 
        report_to="none"
    )
    
    # mlm=False로 설정하여 Next Token Prediction 학습 진행
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator, 
    )

    # 시작 시간 기록
    start_time = time.time()
    print(f"훈련 소요시간 기록 시작: {start_time:.2f}s")
    
    trainer.train()
    
    # 종료 시간 기록 및 소요 시간 계산
    end_time = time.time()
    total_seconds = end_time - start_time
    readable_time = str(timedelta(seconds=int(total_seconds)))

    # 메인 프로세스(Rank 0)에서만 결과 출력
    print(f"\n[학습 종료 보고서]")
    print(f"최종 소요 시간: {readable_time}")
    print(f"전체 초 단위: {total_seconds:.2f}s")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()    

if __name__ == "__main__":
    main()
