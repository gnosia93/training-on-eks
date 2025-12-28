<< 작성중 ..>>

### 1. EFA 환경 최적화 (AWS 필수 설정) ###
AWS의 고속 네트워크망을 제대로 쓰려면 NCCL이 EFA를 기본 통신 계층으로 사용하도록 강제해야 합니다.
* FI_PROVIDER="efa": 통신 프로바이더를 EFA로 지정합니다.
* NCCL_PROTO=simple: EFA 환경에서는 복잡한 프로토콜보다 simple이 더 안정적이고 성능이 잘 나오는 경우가 많습니다.
* FI_EFA_USE_DEVICE_RDMA=1: GPU 간 직접 통신(RDMA)을 활성화하여 CPU 개입을 최소화합니다.

### 2. NCCL 성능 디버깅 (로깅) ###
튜닝 전, 현재 NCCL이 어떻게 작동하는지 파악하는 것이 우선입니다.
* NCCL_DEBUG=INFO: 모든 통신 로그를 출력합니다. 로그에서 "Selected Provider is EFA" 또는 "NVLink" 사용 여부를 반드시 확인하세요.
* NCCL_DEBUG_SUBSYS=GRAPH,INIT,ENV: 토폴로지 구성과 환경 변수 인식 과정을 더 자세히 들여다볼 때 사용합니다.

### 3. 주요 환경 변수 튜닝 (Performance Tuning) ###
훈련 속도(Throughput)를 높이기 위해 다음 변수들을 조정해 보며 최적값을 찾아야 합니다.
* NCCL_BUFFSIZE: 통신 버퍼 크기입니다. 기본값은 2MB(2097152)이나, 대규모 모델 훈련 시 4194304 (4MB) 또는 8388608 (8MB)로 늘리면 성능이 향상될 수 있습니다.
* NCCL_P2P_LEVEL: GPU 간 P2P(Point-to-Point) 통신 방식을 제어합니다. (예: 5는 NVLink를 통한 직접 연결 사용)
* NCCL_IB_DISABLE=1: AWS EFA 사용 시 InfiniBand(IB) 관련 에러가 발생한다면 이를 비활성화하여 EFA만 타도록 유도합니다.

## 레퍼런스 ##

* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
