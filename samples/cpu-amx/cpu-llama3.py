import os
import torch
import torch.distributed as dist  
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import deepspeed

def main():

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    model_name = "meta-llama/Meta-Llama-3-8B"
    output_dir = "./llama3-training-output" 
    ds_config_path = "cpu-ds.json"

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 데이터셋 로드 및 전처리
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # 텍스트가 비어있는 데이터 제외 (학습 효율성)
    dataset = dataset.filter(lambda x: len(x["text"]) > 10)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        num_proc=4                       # 데이터 전처리 병렬화
    )

    # Data Collator 추가: labels를 자동으로 생성해줌 (허깅페이스의 경우 인풋과 레이블은 동일하다. 모델 내부에서 엇깔리게 처리함)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        local_rank=local_rank,                   # deepspeed 가 gloo 환경에서 train batch size 를 제대로 인식하지 못해서 분산환경임을 알려주기 위해 추가

        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        
        deepspeed=ds_config_path,
        use_cpu=True,
        learning_rate=2e-5,
        num_train_epochs=1,
        save_steps=100,
        max_steps=50,                                  # 딱 50번의 스텝만 하고 종료 / 이경우 에포크는 무시됨   
        logging_steps=1,

        # 딥스피드와 중복되지만 유지해야 에러가 나지 않는다. 
        bf16=True,                                     # 인텔 4/5세대 Xeon AMX 가속 활용        
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4
    )

    # 모델 로드 (DeepSpeed가 모델을 쪼개서 로드하도록 처리)
    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=None                    # 딥스피드가 분산 파라미터 로딩시 매핑하도록 함. 
        )

    # 트레이너 초기화 및 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,            # 반드시 포함되어야 함
    )

    print(f"--- {local_rank} RANK 학습 시작 ---")
    trainer.train()
    
    # trainer.save_model(output_dir)

    # 학습 종료 후 자원 해제
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
