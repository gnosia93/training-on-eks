import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import deepspeed
import torch.distributed as dist
from datasets import load_dataset
import time
from datetime import timedelta
import os


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

# 1. TE 레이어 교체 함수 (Linear와 LayerNorm 모두 교체)
def replace_with_te_layers(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            te_linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                params_dtype=torch.bfloat16
            )
            # 가중치 복사 (DeepSpeed Init 안에서 실행되므로 안전함)
            with torch.no_grad():
                te_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    te_linear.bias.copy_(module.bias)
            setattr(model, name, te_linear)
        elif isinstance(module, torch.nn.LayerNorm):
            te_ln = te.LayerNorm(
                hidden_size=module.normalized_shape[0],
                eps=module.eps,
                params_dtype=torch.bfloat16
            )
            with torch.no_grad():
                te_ln.weight.copy_(module.weight)
                te_ln.bias.copy_(module.bias)
            setattr(model, name, te_ln)
        else:
            replace_with_te_layers(module)
    return model

# 2. FP8 Autocast를 적용한 커스텀 Trainer
class TETrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # H100 전용 FP8 훈련 레시피 (분산 훈련 최적화)
        recipe = DelayedScaling(
            fp8_format=Format.HYBRID, 
            amax_history_len=16,
            amax_compute_algo="max"
        )

        # DeepSpeed Stage 3 환경에서도 FP8 연산을 수행하도록 설정
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe, dist_group=dist.group.WORLD):
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach()

def main():
    # flush_gpu_memory() # 정의되어 있다고 가정
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    model_name = "meta-llama/Meta-Llama-3-8B"
    config = AutoConfig.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    ds_config_path = "llama-3-8b-stage3.json"
    
    # 3. 모델 분산 로딩 및 TE 레이어 교체
    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",      # H100에 최적화된 FA2 권장 flash_attention_2
        )
        # 중요: 가중치가 메모리에 할당된 직후, ZeRO가 쪼개기 전에 TE로 변환
        model = replace_with_te_layers(model)

    # 4. 데이터셋 로드 및 전처리
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")   
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 5. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir="/data/fsx",
        per_device_train_batch_size=1, # FP8 사용 시 메모리 여유가 생기므로 1->2 상향 가능
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=50,
        num_train_epochs=1,
        bf16=True, 
        logging_steps=5,
        deepspeed=ds_config_path, 
        save_strategy="no",
        gradient_checkpointing=True,
        log_level="info",
        log_level_replica="warning",                   # 나머지 워커 노드 로그 제한
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. 커스텀 TETrainer 사용
    trainer = TETrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        # callbacks=... (기존 콜백 유지)
    )

    start_time = time.time()
    if trainer.is_world_process_zero():
        print(f"H100 FP8 + DeepSpeed Stage 3 훈련 시작: {start_time:.2f}s")
    
    trainer.train()
    
    end_time = time.time()
    if trainer.is_world_process_zero():
        total_seconds = end_time - start_time
        print(f"\n[학습 종료 보고서]")
        print(f"최종 소요 시간: {str(timedelta(seconds=int(total_seconds)))}")

if __name__ == "__main__":
    main()
