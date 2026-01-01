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

class SimpleTimeCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 현재 시간을 YYYY-MM-DD HH:MM:SS 형태로 추가
            logs["time"] = datetime.datetime.now().strftime("%H:%M:%S")

# transformers 라이브러리의 로그 레벨을 INFO로 설정하여 학습 과정(Loss 등)을 확인합니다.
transformers.utils.logging.set_verbosity_info()
logger = logging.getLogger(__name__)

def main():
    # 최우선 순위: 프로세스 그룹 초기화 (랑데뷰 시작)
    # torchrun으로 실행 시 환경 변수를 읽어 자동으로 4개의 파드를 하나로 묶습니다.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    start_time = time.time()

    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    # 아주 큰 모델을 초기화할 때 메모리 효율을 위해 'meta' 장치 사용
    # ZeRO-3는 이 설정을 통해 모델을 로드하면서 즉시 GPU들에 분산시킨다.
    # config = AutoConfig.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_config(config) 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,                  # training_args의 bf16과 일치
        attn_implementation="sdpa"                   # sdpa(Scaled Dot Product Attention) 사용
    #   attn_implementation="flash_attention_2"      # 지원되는 GPU라면 성능 향상 / flash-attn 미설치 
    )                            
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")   
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 4. 학습 인자 설정 (DeepSpeed 설정 포함)
    training_args = TrainingArguments(
        output_dir="/data/fsx",                        # 분산 체크 포인트 위치                
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,                 # 실제 배치 사이즈 = 4 * 4 * GPU 개수
        learning_rate=2e-5,
        max_steps=50,                                  # 딱 50번의 스텝만 하고 종료 / 이경우 에포크는 무시됨   
        num_train_epochs=1,
        bf16=True,                                     # A100/H100/B200 GPU 권장
        logging_steps=5,
        deepspeed="llama-3-8b-stage3.json", 
        save_strategy="epoch",
        save_total_limit=2,     
        gradient_checkpointing=False,                  # 메모리 절약을 위한 재계산
        log_level="info",                              # 메인 프로세스 로그 레벨
        log_level_replica="warning",                   # 나머지 워커 노드 로그 제한
    )
    
    # mlm=False로 설정하여 Next Token Prediction 학습 진행
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=[SimpleTimeCallback()] 
    )
    
    trainer.train()
    
    # 종료 시간 기록 및 소요 시간 계산
    end_time = time.time()
    total_seconds = end_time - start_time
    readable_time = str(timedelta(seconds=int(total_seconds)))

    # 메인 프로세스(Rank 0)에서만 결과 출력
    if trainer.is_world_process_zero():
        print(f"\n[학습 종료 보고서]")
        print(f"최종 소요 시간: {readable_time}")
        print(f"전체 초 단위: {total_seconds:.2f}s")

    # 학습 종료 후 최종 모델 및 토크나이저 저장
    if training_args.local_rank in [-1, 0]:
        print("\n--- 학습 완료! 최종 모델 저장 중 ---")
        final_save_path = os.path.join(training_args.output_dir, "final_model")
        
        # Trainer.save_model은 ZeRO-3로 분산된 가중치를 자동으로 통합하여 저장합니다.
        # trainer.save_model(final_save_path)
        
        # 추론 시 필요한 토크나이저 설정 파일들을 함께 저장
        # tokenizer.save_pretrained(final_save_path)
        
        print(f"모델과 토크나이저가 저장되었습니다: {final_save_path}")

if __name__ == "__main__":
    main()
