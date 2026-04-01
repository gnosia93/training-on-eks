## NCCL 최적화 ##

NVIDIA NCCL은 대부분의 경우 시스템 토폴로지를 자동으로 감지하고 최적의 통신 경로 및 알고리즘을 선택하므로 사용자가 직접 튜닝할 필요는 없다. 그러나 특정 작업 부하나 하드웨어 구성에서 성능 병목 현상이 발생할 경우, 환경 변수 설정, 하드웨어 최적화, 소프트웨어 최신 버전 유지 등의 기법을 통해 성능을 추가로 튜닝할 수 있다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/nccl-comm.png)

### 1. NCCL 성능 디버깅 ###
* NCCL_DEBUG=INFO: 모든 통신 로그를 출력.
* NCCL_DEBUG_SUBSYS=GRAPH,INIT,ENV: 토폴로지 구성과 환경 변수 인식 과정을 더 자세히 들여다볼 때 사용.

### [2. 주요 환경 변수](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) ###

훈련 속도(Throughput)를 높이기 위해 다음 변수들을 조정해 보며 최적값을 찾아야 한다.
* NCCL_BUFFSIZE: 통신 버퍼 크기. 기본값은 4194304 (4MB) 이나, 대규모 모델 훈련 시 8388608 (8MB) 이상으로  늘리면 성능이 향상될 수 있다.
* NCCL_P2P_LEVEL: GPU 간 P2P(Point-to-Point) 통신 방식을 제어한다. 
* NCCL_P2P_DISABLE=1: GPU 간 P2P(Peer-to-Peer) 통신을 비활성화 하는 것으로 NVLink 사용이 차단된다.
* NCCL_ALGO: 집합 통신(collective communication) 알고리즘을 지정한다.
  * RING: 일반적으로 큰 메시지 크기에 효율적 (대역폭 효율이 높음)
  * TREE: 일반적으로 작은 메시지 크기에 유리 (레이턴시 낮음)
* NCCL_PROTO: 통신 프로토콜을 지정.
  * LL (Low Latency): 작은 메시지 대기 시간을 줄이는 데 적합.
  * SIMPLE: 더 큰 데이터 전송에 맞춰진 프로토콜
* NCCL_SOCKET_NTHREADS: 네트워크 작업을 처리하는 CPU 스레드 수를 조정하여 처리량을 개선할 수 있다.
* NCCL_NET_GDR_LEVEL: GPU Direct RDMA를 지원하는 경우, GPU-NIC 간의 거리를 지정하여 CPU 오버헤드를 줄이고 지연 시간을 단축
 ```
 LOC : Never use GPU Direct RDMA (always disabled).
 PIX : Use GPU Direct RDMA when GPU and NIC are on the same PCI switch.
 PXB : Use GPU Direct RDMA when GPU and NIC are connected through PCI switches (potentially multiple hops).
 PHB : Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. Traffic will go through the CPU.
 SYS : Use GPU Direct RDMA even across the SMP interconnect between NUMA nodes (e.g., QPI/UPI) (always enabled).
 ```
   하지만 대부분 NCCL 이 잘 설정하므로 그냥 둔다~ 


### 3. 성능 프로파일링 ###
* nccl-tests 실행: NVIDIA에서 제공하는 nccl-tests 벤치마크 도구를 사용하여 다양한 구성에서의 NCCL 성능(대역폭 및 지연 시간)을 확인할 수 있다.
* 프로파일링 도구 활용: NVIDIA Nsight Systems, Nsight Compute와 같은 프로파일링 도구를 사용하여 통신 패턴, GPU 활용률, 병목 현상 등을 분석할 수 있다.
* [NCCL Inspector](https://developer.nvidia.com/ko-kr/blog/enhancing-communication-observability-of-ai-workloads-with-nccl-inspector/) 사용: NCCL 2.23부터 도입된 NCCL Inspector 플러그인을 통해 AI 워크로드의 통신 성능에 대한 상세한 가시성을 확보할 수 있다.

### 4. 토폴로지 덤프 및 수정 ###
nccl 성능 측정 도구인 nccl-tests를 이용하는게 가장 간단하다 
```
NCCL_TOPO_DUMP=system.xml ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```
토폴로지가 예상과 다르게 인식되어 성능이 낮게 나올 경우, 추출된 XML을 수정하여 NCCL_TOPO_FILE 변수로 다시 적용함으로써 하드웨어 인식을 강제로 교정할 수 있다.

## 레퍼런스 ##
* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
* https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/
* [Optimizing cuDNN and NCCL Performance on NVIDIA GPUs Using NVIDIA's Deep Learning SDK](https://massedcompute.com/faq-answers/?question=Can%20you%20provide%20examples%20of%20how%20to%20optimize%20cuDNN%20and%20NCCL%20performance%20on%20NVIDIA%20GPUs%20using%20NVIDIA%27s%20Deep%20Learning%20SDK?)
