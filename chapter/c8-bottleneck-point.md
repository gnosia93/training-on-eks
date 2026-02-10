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
