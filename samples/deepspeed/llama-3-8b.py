"""
* 로컬 PC 테스트 고려사항 
1. 하드웨어 체크 (VRAM 용량)
    Llama-3-8B 모델은 BF16 정밀도 기준 모델 크기만 약 16GB.
    성공 가능: GPU VRAM이 24GB 이상(RTX 3090/4090)이라면 단일 GPU에서도 DeepSpeed 없이 또는 ZeRO-2/3를 켜고 테스트가 가능.
    실패 가능: VRAM이 8GB~12항GB라면 일반적인 방식으로는 로딩조차 되지 않고 OOM이 발생. 이 경우 4-bit 양자화(bitsandbytes)를 사용해야 테스트할 수 있음.

2. 운영체제 (OS) 제약
    Linux (권장): DeepSpeed는 리눅스 환경에서 가장 잘 작동하며 설치가 간편.
    Windows: DeepSpeed 설치가 까다롭다. WSL2(Windows Subsystem for Linux)를 설치하고 그 안에서 실행.

3. 로컬 테스트 방법 (터미널 실행)
    torchrun --nproc_per_node=1 your_script.py

* EFA 모니터링
1. Network Throughput: ZeRO-3를 사용하면 매 스텝마다 가중치를 다시 모으고 흩뿌리는 과정이 발생하므로, 네트워크 사용량이 주기적으로 튑니다(Spike). 
   만약 이 속도가 너무 낮다면 EFA 설정이나 보안 그룹 문제를 의심해야 한다.
2. Sent/Recv 균형: 분산 학습은 데이터를 주고받는 구조이므로 송신(Sent)과 수신(Recv)량이 거의 비슷하게 나타나야 정상. 
   한쪽이 너무 높다면 특정 파드에 병목이 걸린 것.
3. EFA 전용 모니터링 (고급): 만약 eth0가 아닌 실제 EFA 장치의 전송량만 보고 싶다면, psutil.net_io_counters(pernic=True)를 사용하여 
   efa0 인터페이스만 필터링하도록 코드를 수정.
"""

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

class SimpleTimeCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 현재 시간을 YYYY-MM-DD HH:MM:SS 형태로 추가
            logs["time"] = datetime.datetime.now().strftime("%H:%M:%S")

class MemoryLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 5스텝마다 혹은 원하는 주기에 맞춰 출력 (logging_steps와 연동 권장)
        if state.global_step % args.logging_steps == 0:
            if torch.cuda.is_available():
                # 현재 프로세스가 사용하는 GPU 번호
                device = torch.cuda.current_device()
                
                # 할당된 메모리 (실제 모델+액티베이션이 쓰는 양)
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                # 예약된 메모리 (PyTorch가 드라이버로부터 빌려온 전체 양)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                # 최대 피크 기록
                max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                print(f"\n[Step {state.global_step}] GPU {device} Memory: "
                      f"Allocated: {allocated:.2f}GB, "
                      f"Reserved: {reserved:.2f}GB, "
                      f"Peak: {max_mem:.2f}GB")
                
                # 피크 통계 초기화 (다음 구간의 피크를 보기 위함 - 선택 사항)
                torch.cuda.reset_peak_memory_stats(device)

class SystemMonitorCallback(TrainerCallback):
    def __init__(self):
        # 초기 네트워크 수치 저장
        net_io = psutil.net_io_counters()
        self.last_net_sent = net_io.bytes_sent
        self.last_net_recv = net_io.bytes_recv
        self.last_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            curr_time = time.time()
            net_io = psutil.net_io_counters()
            
            # 이전 로깅 시점 이후의 전송량 계산 (MB)
            sent_mb = (net_io.bytes_sent - self.last_net_sent) / (1024 ** 2)
            recv_mb = (net_io.bytes_recv - self.last_net_recv) / (1024 ** 2)
            interval = curr_time - self.last_time
            
            # 초당 전송 속도 (MB/s)
            throughput = (sent_mb + recv_mb) / interval

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                
                print(f"\n[Step {state.global_step}] "
                      f"Memory: {allocated:.2f}GB | "
                      f"Net Sent: {sent_mb:.1f}MB, Recv: {recv_mb:.1f}MB | "
                      f"Throughput: {throughput:.1f}MB/s")

            # 수치 업데이트
            self.last_net_sent = net_io.bytes_sent
            self.last_net_recv = net_io.bytes_recv
            self.last_time = curr_time

# transformers 라이브러리의 로그 레벨을 INFO로 설정하여 학습 과정(Loss 등)을 확인합니다.
transformers.utils.logging.set_verbosity_info()
logger = logging.getLogger(__name__)

def main():
    flush_gpu_memory()
    # 1. 최우선 순위: 프로세스 그룹 초기화 (랑데뷰 시작)
    # torchrun으로 실행 시 환경 변수를 읽어 자동으로 4개의 파드를 하나로 묶습니다.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    start_time = time.time()
    model_name = "meta-llama/Meta-Llama-3-8B"
    config = AutoConfig.from_pretrained(model_name)
    
    # 2. 토크나이저는 CPU 작업이므로 먼저 진행해도 무관
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 3. 모델 분산 로딩
    # 모델 로딩시에 처음부터 파라미터 등을 분산해서 로딩 (DeepSpeed가 파라미터를 쪼개서 RANK 별로 로딩 하도록 처리)
    # 이와 관련해서 잘못 설정하는 경우 모든 랭크가 전체 파라미터를 올리게 된다. (GPU 터짐)
    # 각각 모든 파라미터를 올린 후에(Trainer 가 올림) 팁스피드가 관여해서 다시 모델을 쪼개게 된다. 
    ds_config_path = "llama-3-8b-stage3.json"
    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,                                  # 분산 학습 시 필수: None -> deepspeed 가 모델을 조각내도록 함.
            low_cpu_mem_usage=True,        # Meta tensor 에러 방지에 도움
            attn_implementation="sdpa",
        )      
    # DeepSpeed 엔진 초기화
    # 여기서 모델이 실제 8장의 GPU로 완벽히 분산됩니다.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config_path,
        # model_parameters=model.parameters() # 필요 시 추가
    )
    
    
    # 4. 데이터셋 로드 및 전처리 (전처리 시 CPU 메모리 주의)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")   
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 5. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir="/data/fsx",                        # 분산 체크 포인트 위치                
        per_device_train_batch_size=1,                 # 40GB에선 1~2로 시작하는 것이 안전함.
        gradient_accumulation_steps=4,                 # 실제 배치 사이즈 = 4 * 4 * GPU 개수
        learning_rate=2e-5,
        max_steps=50,                                  # 딱 50번의 스텝만 하고 종료 / 이경우 에포크는 무시됨   
        num_train_epochs=1,
        bf16=True,                                     # A100/H100/B200 GPU 권장
        logging_steps=5,
        deepspeed=ds_config_path, 
        save_strategy="no",                            # 테스트 중엔 저장 끔 / save_strategy="epoch",
  #      save_total_limit=2,     
        gradient_checkpointing=True,                   # 40GB 환경에선 필수로 True 권장 / activation 저장 안함.
        log_level="info",                              # 메인 프로세스 로그 레벨
        log_level_replica="warning",                   # 나머지 워커 노드 로그 제한
        report_to="none"                               # 불필요한 로그 방지
    )
    
    # mlm=False로 설정하여 Next Token Prediction 학습 진행
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=[SimpleTimeCallback(), MemoryLoggingCallback(), SystemMonitorCallback()] 
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
