
## 1. 하드웨어 장애 (Hardware Failures) ##
AI 클러스터는 고성능 연산 장치를 장시간 풀로드로 돌리기 때문에 하드웨어 결함 빈도가 매우 높습니다.

* **GPU 장애 / 오류:** GPU Failure / XID Error *(NVIDIA GPU 하드웨어·드라이버 치명적 오류)*
* **GPU 메모리(HBM) 오류:** GPU Memory Error / ECC Error *(메모리 비트 반전 등)*
* **과열 및 전력 문제:** Thermal Throttling / Overheating / Power Supply Failure
* **노드 다운 (서버 먹통):** Node Failure / Node Down / Unresponsive Node
* **스토리지 장애:** Storage Failure / Disk I/O Bottleneck *(체크포인트 저장 지연 등)*

## 2. 네트워크 및 통신 장애 (Network & Interconnect Failures) ##
분산 학습(Distributed Training) 시 노드 간 데이터 병목이나 단절로 인해 전체 학습이 멈추는 장애입니다.

* **고속 인터커넥트 장애:** InfiniBand (IB) Link Down / NVLink Failure
* **집단 통신 타임아웃:** NCCL Timeout / Collective Communication Failure *(GPU 간 동기화 실패)*
* **패킷 손실 및 지연:** Packet Loss / Network Congestion / High Latency
* **NIC(네트워크 카드) 장애:** NIC Failure / RoCE Error

## 3. 소프트웨어 및 런타임 장애 (Software & Runtime Failures) ##
* **메모리 부족 (OOM):** Out of Memory *(GPU OOM / Host OOM)*
* **커널 패닉 및 드라이버 충돌:** Kernel Panic / CUDA Driver Crash
* **컨테이너 및 오케스트레이션 오류:** Pod CrashLoopBackOff / Kubelet NotReady *(Kubernetes 환경)*
* **스케줄러 장애:** Slurm Daemon Failure / Job Preemption Error *(Slurm 환경)*

## 4. 분산 학습 특화 현상 (Distributed Training Anomalies) ##
명확한 에러 코드 없이 학습이 멈추거나 성능이 급감하는 현상입니다.

* **학습 멈춤 (행 걸림):** Training Hang / Deadlock
* **성능 저하 노드 (낙오자):** Straggler Node *(특정 GPU만 연산이 느려져 전체 속도를 저하시키는 현상)*
* **조용한 데이터 손상:** Silent Data Corruption (SDC) *(에러 없이 잘못된 연산 결과를 내뱉는 하드웨어 결함)*


