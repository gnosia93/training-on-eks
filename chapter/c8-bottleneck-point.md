### 1. 데이터 공급 병목 (Data Ingestion Pipeline) ###
GPU가 연산을 끝냈는데 다음 데이터가 준비되지 않아 노는 현상입니다.
* 발생 원인: 느린 스토리지(Lustre 메타데이터 부하), CPU 코어 부족, 공유 메모리(shm) 부족, 네트워크 스토리지 대역폭 포화.
* 현상: GPU Utilization(사용률)이 주기적으로 요동침, GPU Power Consumption(전력 소모)이 급격히 떨어짐.
* 모니터링 항목:
  * GPU-Util: nvidia-smi에서 사용률이 100% 미만으로 출렁이는지 확인.
  * CPU-Util: 특정 CPU 코어만 100%를 찍고 있는지(I/O Wait).
  * DataLoader Latency: PyTorch Profiler에서 enumerate(dataloader)에 걸리는 시간 측정.
