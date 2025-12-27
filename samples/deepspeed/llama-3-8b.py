import torch
from transformers import AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import os

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
    output_dir="./llama3-deepspeed",
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
)

# 5. 트레이너 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()

# 6. 학습 종료 후 최종 모델 및 토크나이저 저장
# 이 과정은 추후 배포를 위해 분산된 가중치를 하나로 병합하는 시도를 포함합니다.
trainer.save_model("/data/fsx/llama3-deepspeed-final")
tokenizer.save_pretrained("/data/fsx/llama3-deepspeed-final")

