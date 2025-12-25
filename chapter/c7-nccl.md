② NCCL 백엔드 성능 튜닝
PyTorch 분산 학습에서 사용하는 NCCL(NVIDIA Collective Communications Library)은 노드 간 데이터를 주고받을 때 가장 빠른 경로를 찾습니다. 이때 네트워크 토폴로지를 확인하거나 소켓 버퍼 크기 등을 조정하여 통신 속도를 극대화하기 위해 NET_ADMIN 권한이 활용됩니다.

#### 1. AWS 환경 최적화 (EFA 활성화) ####
AWS의 고성능 인스턴스(P4, P5, G5 등)를 사용한다면 EFA(Elastic Fabric Adapter) 사용은 필수입니다.
* FI_EFA_USE_DEVICE_RDMA=1: RDMA(Remote Direct Memory Access)를 활성화하여 CPU 개입 없이 GPU 메모리 간 직접 통신을 수행합니다.
* NCCL_PROTO=simple: EFA 환경에서는 복잡한 프로토콜보다 simple이 더 높은 처리량을 기록하는 경우가 많습니다.

#### 2. 네트워크 인터페이스 명시 (가장 중요) ####
다중 네트워크 인터페이스가 있는 환경에서 NCCL이 엉뚱한 인터페이스(예: 관리용 느린 네트워크)를 잡지 않도록 강제해야 합니다.
* NCCL_SOCKET_IFNAME=eth0,en: 통신에 사용할 인터페이스 이름을 명시합니다. (AWS는 보통 eth0 또는 en으로 시작하는 인터페이스를 사용합니다.)
* NCCL_IB_DISABLE=0: 인피니밴드(InfiniBand)나 RDMA 인터페이스 사용을 강제합니다. (EFA 사용 시 필수)

#### 3. 공유 메모리(Shared Memory) 및 버퍼 최적화 ####
노드 내 GPU 간 통신(NVLink 등) 속도를 높이기 위한 설정입니다.
* NCCL_SHM_DISABLE=0: 노드 내 GPU 간 데이터 전달 시 공유 메모리 사용을 활성화합니다.
* NCCL_BUFFSIZE=2097152: 통신 버퍼 크기를 늘립니다(기본 2MB). 고해상도 이미지나 거대 모델(LLM) 학습 시 4MB~8MB로 늘리면 성능이 향상될 수 있습니다.
* NCCL_P2P_LEVEL=5: GPU 간 Peer-to-Peer 통신 단계를 설정합니다. 5는 시스템의 모든 하드웨어 경로(NVLink, PCI-E 등)를 최대한 활용하도록 합니다.

#### 4. 디버깅 및 분석 (성능 병목 확인) ####
튜닝 전후의 차이를 확인하기 위해 로그를 활성화합니다.
* NCCL_DEBUG=INFO: 학습 시작 시 NCCL이 어떤 인터페이스를 찾았고, 어떤 알고리즘(Tree, Ring 등)을 선택했는지 출력합니다.
* NCCL_DEBUG_SUBSYS=GRAPH,INIT,ENV: 더 상세한 그래프 연결 상태를 확인하여 병목 지점을 찾습니다.
