### 1. GPU 연산 속도 저하 식별 ###
GPU의 물리적 결함이나 스로틀링(Throttling)으로 인한 속도 저하는 NVIDIA DCGM Exporter를 통해 수집된 지표로 확인할 수 있다. 

#### 주요 지표 (Prometheus/Grafana): ####
* DCGM_FI_DEV_GPU_UTIL: GPU 연산 사용률이 100%에 도달했음에도 실제 처리 속도가 느리다면 하드웨어 스로틀링을 의심해야 한다.
* DCGM_FI_DEV_CLOCK_THROTTLE_REASONS: 클럭 저하 원인(온도, 전력 제한 등)을 직접적으로 나타낸다.
* DCGM_FI_DEV_GPU_TEMP: 온도가 높을 경우 성능이 자동으로 제한된다.
직접 확인: 특정 노드에 접속하여 nvidia-smi -q -d PERFORMANCE 명령어로 현재 성능 상태와 스로틀링 여부를 즉시 조회할 수 있다. 

### 2. 네트워크 카드(NIC) 및 네트워크 성능 저하 식별 ###
네트워크 카드 자체의 문제나 CNI 설정 오류로 인한 병목은 지연 시간(Latency)과 패킷 손실 지표를 통해 식별한다. 

#### 식별 방법: ####
* 도구 활용: iperf 또는 netperf를 파드 간에 실행하여 실제 대역폭과 지연 시간을 측정한다.
* CNI 모니터링: Cilium이나 Calico 같은 CNI는 자체적인 대시보드나 로그를 통해 패킷 드롭 및 재전송(Retransmission) 발생 여부를 제공한다.
* 노드 레벨 지표: node_network_transmit_errs_total (전송 에러), node_network_receive_drop_total (수신 드롭) 지표가 급증하는 노드의 NIC를 점검해야 한다. 

### 3. 노드(Node) 및 하드웨어 병목 식별 ###
노드 전체의 성능 저하는 CPU/메모리 부족뿐만 아니라 디스크 I/O 병목에 의해서도 발생한다. 

#### 주요 모니터링 포인트: ####
* CPU Throttling: container_cpu_cfs_throttled_seconds_total 지표를 통해 노드 리소스가 충분함에도 특정 컨테이너가 제한을 받는지 확인한다.
* Disk I/O Wait: 노드의 iowait 수치가 높다면 디스크 성능 저하가 전체 시스템 연산 속도를 늦추고 있는 것이다.
* Kube-state-metrics: 노드의 상태(Ready/NotReady) 외에도 DiskPressure, PIDPressure 등의 이벤트를 실시간으로 감시하여 하드웨어 한계 도달 여부를 확인한다.
