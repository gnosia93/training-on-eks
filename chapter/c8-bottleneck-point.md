### 1. 데이터 공급 병목 (Data Ingestion Pipeline) ###
GPU가 연산을 끝냈는데 다음 데이터가 준비되지 않아 노는 현상입니다.
* 발생 원인: 느린 스토리지(Lustre 메타데이터 부하), CPU 코어 부족, 공유 메모리(shm) 부족, 네트워크 스토리지 대역폭 포화.
* 현상: GPU Utilization(사용률)이 주기적으로 요동침, GPU Power Consumption(전력 소모)이 급격히 떨어짐.
* 모니터링 항목:
  * GPU-Util: nvidia-smi에서 사용률이 100% 미만으로 출렁이는지 확인.
  * CPU-Util: 특정 CPU 코어만 100%를 찍고 있는지(I/O Wait).
  * DataLoader Latency: PyTorch Profiler에서 enumerate(dataloader)에 걸리는 시간 측정.

### 2. 통신 병목 (Communication Overheads) ###
분산 훈련 시 가중치와 미분값을 주고받는 과정에서 지연이 발생하는 경우입니다.
* 발생 원인: NVLink 미작동(P2P 통신 실패), 노드 간 네트워크(InfiniBand/RoCE) 대역폭 부족, NCCL 설정 오류.
* 현상: Step Time(한 스텝당 시간)이 비정상적으로 길어짐, GPU 사용률은 높으나 실제 연산 속도가 느림.
* 모니터링 항목:
  * NCCL_P2P_DISABLE: 설정이 0(활성화)인지 확인.
  * Bandwidth: nccl-tests를 돌려 노드 간 실제 대역폭(Gbps) 측정.
  * Bus Bandwidth: nvidia-smi nvlink -gt로 NVLink 통과 데이터량 모니터링.

### 3. 메모리 및 정밀도 병목 (Memory & Precision Issues) ###
메모리 관리가 효율적이지 못해 연산 효율이 떨어지는 지점입니다.
* 발생 원인: FP8/BF16 미적용, 과도한 Checkpointing 주기, ZeRO-3 가중치 수집(Gather) 지연.
* 현상: Out Of Memory(OOM) 발생, 연산 정밀도 하락으로 인한 Loss 발산, 잦은 캐시 미스로 인한 속도 저하.
* 모니터링 항목:
  * GPU Memory Usage: fb_used가 한계치에 도달했는지.
  * Reserved Memory: torch.cuda.memory_reserved()로 실제 할당된 공간 확인.
  * Throughput (TFLOPS): 이론적 최대 성능 대비 실제 계산량(tokens/sec) 비교.

### 4. 소프트웨어 및 커널 병목 (Kernel & Framework) ###
최적화되지 않은 연산 커널이 GPU 자원을 낭비하는 경우입니다.
* 발생 원인: Triton/CUDA 커널 비효율, 비효율적인 Attention 구현(FlashAttention 미사용), Python 인터프리터 오버헤드.
* 현상: GPU 사용률은 100%인데 학습 속도가 예상보다 훨씬 느림.
* 모니터링 항목:
  * Kernel Trace: PyTorch Profiler로 어떤 CUDA 커널(예: aten::add_)이 시간을 가장 많이 잡아먹는지 확인.
  * SM Efficiency: nvidia-smi의 SM 이용률을 세부 분석하여 연산 유닛이 실제로 일하는지 확인..  
  
|구분|	모니터링| 지표|	도구|	임계치 및 이상 징후|
|*-*|*-*|*-*|*-*|*-*|
|GPU|	Volatile GPU-Util|	nvidia-smi|	95% 미만 유지 시 데이터/통신 병목 의심|
|Power|	Power Draw|	nvidia-smi|	TDP 대비 현저히 낮으면 GPU Starvation|
|Comm|	NCCL_DEBUG=INFO|	환경변수|	All-Reduce 단계에서 멈춤 현상 발생 여부|
|Storage|	I/O Wait| / Disk| Read	iostat, dstat	Lustre 읽기 속도가 학습 요구량보다 낮음|
|System|	/dev/shm| usage|	df -h	공유 메모리 가득 참 (Bus error 유발)|
