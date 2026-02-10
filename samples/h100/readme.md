H100과 NVLink라는 '슈퍼카'를 뽑으셨으니, 엔진(GPU)이 기름(Data)을 기다리느라 서행하는지 감시하는 게 핵심입니다. GPU Starvation을 잡기 위해 꼭 보셔야 할 지표 3가지를 정리해 드립니다.

### 1. GPU Utilization (사용률) - 가장 직관적인 지표 ###
nvidia-smi나 NVIDIA DCGM Exporter를 통해 확인합니다.
* 지표: Volatile GPU-Util
* 현상: 연산 중인데 사용률이 100% 근처에서 놀지 않고 60~80%를 왔다 갔다 하거나, 주기적으로 0% 근처로 툭툭 떨어진다면 100% 데이터 로딩 병목입니다.
* 이유: GPU 연산은 끝났는데 다음 데이터가 CPU에서 아직 안 넘어와서 노는 상태입니다.

### 2. GPU Memory Copy (데이터 전송 통로) ###
* 지표: fb_used가 아닌 utilization.memory (또는 MemCpy 대역폭)
* 현상: GPU 내부 연산(sm_util)은 낮은데, 메모리 복사(mem_util) 지표만 가끔 튄다면 CPU에서 GPU로 데이터를 쏘는 과정이 너무 느린 것입니다.
* 체크: 이때는 앞서 말한 shm 설정이나 pin_memory=True 옵션을 다시 점검해야 합니다.

### 3. DataLoader Bottleneck (CPU 측면) ###
K8s 환경에서 PyTorch Profiler를 코드에 살짝 심어서 보시면 정확합니다.
* 코드 한 줄: with torch.profiler.profile(...)
* 확인: 결과 리포트에서 Enumerating DataLoader 또는 DataLoader.next에 걸리는 시간이 전체 스텝 시간의 상당 부분을 차지한다면, CPU의 num_workers가 부족하거나 shm 병목입니다.

### 💡 실전 모니터링 팁 (K8s) ###
쿠버네티스 대시보드나 그라파나(Grafana)를 쓰신다면 NVIDIA GPU Dashboard를 띄우고 다음 대조군을 보세요:
* 정상: GPU Power Usage가 설계 전력(TDP) 근처에서 일정하게 유지됨.
* Starvation: GPU Power Usage가 파도치듯 출렁임 (데이터 올 때만 전기를 먹고 안 올 땐 쉬기 때문).

### 🛠️ 바로 실행해볼 수 있는 명령어 ###
```bash
# 1초마다 GPU 상태 감시 (H100 8대 기준)
nvidia-smi dmon -i 0,1,2,3,4,5,6,7 -s u
```
* 여기서 sm(연산) 수치는 낮고 enc나 dec 혹은 아무것도 안 하는 시간이 길다면 바로 데이터 로딩을 의심하세요!

---

### ⚠️ K8s에서 주의할 점: "휘발성 컨테이너" ###
K8s 파드(Pod)는 떴다 죽으면 내부 파일이 사라집니다.
* 중복 다운로드: 파드를 새로 띄울 때마다 수백 GB의 데이터셋과 모델을 매번 새로 받느라 GPU가 노는 시간이 생깁니다. (GPU Starvation의 주범!)
* 스토리지 병목: 만약 K8s 노드의 기본 디스크(Root FS)가 느리면, GPU는 H100인데 데이터를 읽어오는 속도가 못 따라와서 성능이 깎입니다.

### ✅ 추천 해결책: PVC 또는 로컬 마운트 ###
주말에 테스트하실 때 아래 방식 중 하나를 권장합니다.
* HostPath 마운트: 노드의 빠른 NVMe SSD 디렉토리를 파드의 /root/.cache/huggingface에 연결하세요. (가장 빠름)
* Shared PVC: 여러 GPU 노드가 같은 데이터를 쓴다면 공유 스토리지를 사용하되, ReadWriteMany 설정을 확인하세요.

#### 💡 팁: 다운로드 경로 고정하기 ####
```python
# 환경변수로 설정 (K8s YAML의 env 섹션 추천)
export HF_HOME="/mnt/fast_ssd/hf_cache"
```
데이터셋 로딩이 시작될 때 네트워크 대역폭보다는 디스크 I/O(읽기 속도)를 모니터링해 보세요. H100이라면 초당 수 GB는 읽어줘야 GPU가 배불리 먹습니다! 🦾

---

4개의 프로세스(또는 파드)가 동시에 "나 데이터 없네? 내가 받을게!" 하고 같은 폴더에 쓰기를 시작하면 파일이 깨지거나(Corrupted), 네트워크 대역폭을 나눠 쓰느라 다운로드 속도가 기어가는 지옥을 보게 됩니다.이를 해결하는 표준적인 방법들을 정리해 드릴게요.

### 1. Hugging Face의 내장 락(File Lock) 메커니즘 ###
다행히 Hugging Face의 transformers와 datasets 라이브러리에는 파일 기반 락(.lock) 기능이 들어있습니다.
* 동작: 한 프로세스가 다운로드를 시작하면 해당 폴더에 .lock 파일을 만듭니다.
* 효과: 다른 프로세스들은 이 락 파일을 보고 "누가 받고 있구나" 하고 기다립니다(Waiting).
* 문제점: 하지만 네트워크 스토리지(NFS 등)를 공유할 때는 이 파일 락이 가끔 제대로 작동하지 않아 먹통이 되기도 합니다.

### 2. 가장 깔끔한 방법: '메인 프로세스 우선' 로직 ###
Hugging Face의 accelerator나 Trainer를 쓰신다면 코드에 이런 로직을 넣어 경합을 원천 차단합니다.
```python
from accelerate import Accelerator

accelerator = Accelerator()

# 0번 프로세스(Main)만 다운로드를 먼저 수행
with accelerator.main_process_first():
    # 여기서 모델/데이터셋 로딩 (나머지 프로세스는 여기서 대기)
    model = AutoModelForCausalLM.from_pretrained(...)
    dataset = load_dataset(...)
```

* 작동 원리: main_process_first() 컨텍스트 안에서는 0번 GPU(Rank 0)만 먼저 지나가서 다운로드를 완료합니다. 그동안 나머지 GPU들은 문 앞에서 기다렸다가, 다운로드가 끝나면 이미 받아진 파일을 읽기만 합니다.

### 3. 인프라 수준의 해결책 (K8s 특화) ###
K8s 환경이라면 코드를 고치기보다 인프라 설정을 활용하는 게 더 세련된 방법입니다.
* Init Containers: 메인 학습 컨테이너가 뜨기 전, Init Container를 하나 띄워 데이터를 미리 다운로드 받아둡니다.
* Pre-download Script: 학습 스크립트 실행 직전에 if [ $RANK -eq 0 ]; then ... 구문을 넣어 0번 파드만 다운로드하게 강제합니다.

### 💡 실전 팁 ###
주말에 테스트하실 때, H100 8장(또는 4장)이 동시에 같은 저장소에 접근하면 파일 시스템 부하가 엄청납니다. 가급적 0번 프로세스만 다운로드하게 락을 걸거나, 학습 시작 전 Hugging Face CLI의 download 명령어로 미리 데이터를 다 받아두시는 걸 강력 추천합니다

### 컨테이너 hostpath Mount ###
이미 호스트의 특정 디레토리에 (NVME 사용하느 디렉토리에) 훈련 샘플 데이터를 미리 받아 놓은게 골든 룰이다..
```
spec:
  containers:
  - name: h100-trainer
    volumeMounts:
    - mountPath: /root/.cache/huggingface  # 컨테이너 안의 캐시 경로
      name: hf-cache
  volumes:
  - name: hf-cache
    hostPath:
      path: /data/hf_cache  # 호스트(노드)의 실제 경로
      type: DirectoryOrCreate
```

---
### 1. 전용 데이터 로딩 라이브러리: NVIDIA DALI ###
* 개념: CPU가 하던 데이터 전처리(복호화, 리사이징 등)까지 GPU가 직접 하게 만드는 라이브러리입니다.
* 장점: H100은 연산 능력이 너무 남아서 CPU가 데이터를 다듬어주는 속도조차 느리게 느껴질 수 있습니다. DALI를 쓰면 GPU가 직접 NVMe에서 데이터를 땡겨와서 스스로 전처리하므로 CPU 병목을 완전히 제거할 수 있습니다.
* 단점: 코드를 DALI 전용으로 수정해야 해서 주말에 당장 적용하기에는 공수가 듭니다.

### 2. 가상 분산 파일 시스템: Alluxio 또는 JuiceFS ###
* 개념: 여러 노드의 로컬 NVMe들을 하나로 묶어서 거대한 메모리 캐시 층처럼 만드는 방식입니다.
* 장점: 데이터셋이 수십 TB라서 노드 한 대의 NVMe에 다 안 들어갈 때 유용합니다. 첫 노드가 데이터를 받으면 다른 노드들도 로컬 속도로 그 데이터를 공유받을 수 있습니다.
* 현실: 질문자님이 다루시는 데이터가 노드 한 대의 NVMe(보통 수 TB)에 다 들어가는 수준이라면, 단순한 hostPath가 훨씬 빠르고 관리하기 편합니다.

### 3. Hugging Face Datasets Streaming ###
* 개념: 데이터를 다 받지 않고, 학습에 필요한 만큼만 실시간으로 스트리밍해서 쓰는 방식입니다.
* 장점: 다운로드 완료를 기다릴 필요 없이 즉시 학습을 시작할 수 있습니다.
* 위험: 인터넷 환경이나 HF 서버 상태에 따라 GPU Starvation이 발생할 확률이 매우 높습니다. H100 클러스터에서는 가급적 피해야 할 방식입니다.

