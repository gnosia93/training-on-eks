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
