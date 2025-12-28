import torch
from transformers import AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import os


# [로그 설정] 
# transformers 라이브러리의 로그 레벨을 INFO로 설정하여 학습 과정(Loss 등)을 확인합니다.
transformers.utils.logging.set_verbosity_info()
logger = logging.getLogger(__name__)

def main():
    # 1. 모델 및 토크나이저 설정 (Llama-3-8B 예시)
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 아주 큰 모델을 초기화할 때 메모리 효율을 위해 'meta' 장치 사용
    # ZeRO-3는 이 설정을 통해 모델을 로드하면서 즉시 GPU들에 분산시킨다.
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config) 
    
    # 3. 데이터셋 로드 (간단한 예시)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 4. 학습 인자 설정 (DeepSpeed 설정 포함)
    training_args = TrainingArguments(
        output_dir="/data/fsx",                        # 분산 체크 포인트 위치                
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,                 # 실제 배치 사이즈 = 4 * 4 * GPU개수
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=True,                                     # A100/H100/B200 GPU 권장
        logging_steps=10,
        deepspeed="llama-3-8b-stage3.json", 
        save_strategy="epoch",
        save_total_limit=2,     
        gradient_checkpointing=True,                   # 메모리 절약을 위한 재계산
        log_level="info",                              # 메인 프로세스 로그 레벨
        log_level_replica="warning",                   # 나머지 워커 노드 로그 제한
    )
    
    # 5. 트레이너 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    trainer.train()

    # 6. 학습 종료 후 최종 모델 및 토크나이저 저장
    # 마스터 노드(Rank 0)에서만 실행하여 파일 중복 쓰기 방지
    if training_args.local_rank in [-1, 0]:
        trainer.save_model("/data/fsx/llama3-deepspeed-final")
        tokenizer.save_pretrained("/data/fsx/llama3-deepspeed-final")


if __name__ == "__main__":
    main()
