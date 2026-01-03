NVIDIA NCCL은 대부분의 경우 시스템 토폴로지를 자동으로 감지하고 최적의 통신 경로 및 알고리즘을 선택하므로 사용자가 직접 튜닝할 필요는 없다. 그러나 특정 작업 부하나 하드웨어 구성에서 성능 병목 현상이 발생할 경우, 환경 변수 설정, 하드웨어 최적화, 소프트웨어 최신 버전 유지 등의 기법을 통해 성능을 추가로 튜닝할 수 있다.

## NCCL EFA 플러그인 로딩 확인 ##
NCCL 최적화를 수행하기 전에 EFA를 통한 분산 학습이 이뤄지고 있는지 확인이 필요하다. RANK 0 로그를 열어 nccl 이 efa 플러그인 성공적으로 로딩했는지 확인한다. 아래 [RANK 0 로그 예시] 에서 확인할 항목들은 아래와 같다. 이 항목들을 로그에서 관찰할 수 있다면 nccl 이 efa 를 이용하여 분산 학습을 하고 있다는 것을 의미한다.  

[efa 플러그인 관련 항목]
```
# 1. EFA 플러그인 로드 - AWS OFI(Open Fabric Interface) NCCL 플러그인(libnccl-net.so, aws-ofi-nccl)
NCCL INFO NET/Plugin: Successfully loaded external plugin libnccl-net.so
NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2

# 2. EFA 프로바이더 선택
NCCL INFO NET/OFI Setting provider_filter to efa
NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)

# 3. GPU 및 NCCL 버전 확인 - CUDA 13.0
NCCL INFO cudaDriverVersion 13000
NCCL INFO NCCL version 2.27.3+cuda12.9

# 4. 프로세스 및 GPU 매핑 - 16 랭크 구성
llama-3-8b-node-0-0:194:724 [3] ... cudaDev 3 nvmlDev 3 busId 3e000 commId ... rank 3 nranks 16
```

## NCCL 최적화 ##

### 1. NCCL 성능 디버깅 (로깅) ###
* NCCL_DEBUG=INFO: 모든 통신 로그를 출력.
* NCCL_DEBUG_SUBSYS=GRAPH,INIT,ENV: 토폴로지 구성과 환경 변수 인식 과정을 더 자세히 들여다볼 때 사용.

### 2. 주요 환경 변수 튜닝 (Performance Tuning) ###

훈련 속도(Throughput)를 높이기 위해 다음 변수들을 조정해 보며 최적값을 찾아야 한다.
* NCCL_BUFFSIZE: 통신 버퍼 크기입니다. 기본값은 2MB(2097152)이나, 대규모 모델 훈련 시 4194304 (4MB) 또는 8388608 (8MB)로 늘리면 성능이 향상될 수 있다.
* NCCL_P2P_LEVEL: GPU 간 P2P(Point-to-Point) 통신 방식을 제어한다. 
* NCCL_IB_DISABLE=1: AWS EFA 사용 시 InfiniBand(IB) 관련 에러가 발생한다면 이를 비활성화하여 EFA만 타도록 유도한다.
* NCCL_P2P_DISABLE=1: GPU 간 P2P(Peer-to-Peer) 통신을 비활성화 하는 것으로 NVLink 사용이 차단된다.
* NCCL_ALGO: 집합 통신(collective communication) 알고리즘을 지정한다.
  * RING: 일반적으로 작은 메시지 크기에 효율적.
  * TREE: 일반적으로 큰 메시지 크기나 복잡한 토폴로지(노드 간 연결이 불균일한 경우 등)에서 더 나은 성능을 보일 수 있다.
* NCCL_PROTO: 통신 프로토콜을 지정.
  * LL (Low Latency): 작은 메시지 대기 시간을 줄이는 데 적합.
  * SIMPLE: 더 큰 데이터 전송에 맞춰진 프로토콜
* NCCL_SOCKET_NTHREADS: 네트워크 작업을 처리하는 CPU 스레드 수를 조정하여 처리량을 개선할 수 있다.
* NCCL_NET_GDR_LEVEL: GPU Direct RDMA를 지원하는 경우, 이를 활성화하여 CPU 오버헤드를 줄이고 지연 시간을 단축

### 3. 모니터링 / 밴치마킹 ###
* nccl-tests 실행: NVIDIA에서 제공하는 nccl-tests 벤치마크 도구를 사용하여 다양한 구성에서의 NCCL 성능(대역폭 및 지연 시간)을 확인할 수 있다.
* 프로파일링 도구 활용: NVIDIA Nsight Systems, Nsight Compute와 같은 프로파일링 도구를 사용하여 통신 패턴, GPU 활용률, 병목 현상 등을 분석할 수 있다.
* [NCCL Inspector](https://developer.nvidia.com/ko-kr/blog/enhancing-communication-observability-of-ai-workloads-with-nccl-inspector/) 사용: NCCL 2.23부터 도입된 NCCL Inspector 플러그인을 통해 AI 워크로드의 통신 성능에 대한 상세한 가시성을 확보할 수 있다.

## 레퍼런스 ##
* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
* https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/
* [Optimizing cuDNN and NCCL Performance on NVIDIA GPUs Using NVIDIA's Deep Learning SDK](https://massedcompute.com/faq-answers/?question=Can%20you%20provide%20examples%20of%20how%20to%20optimize%20cuDNN%20and%20NCCL%20performance%20on%20NVIDIA%20GPUs%20using%20NVIDIA%27s%20Deep%20Learning%20SDK?)
* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
