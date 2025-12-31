***이 챕터는 GPU 간의 인터커넥트 통신 매커니즘을 다루는 이론 파트이다. 실습용 스크립트나 테스트 환경은 제공하지 않는다.***  

## GPU 토폴로지 ##
GPU 와 GPU 간의 데이터를 주고 받은 방식에는 아래와 같이 4가지 타입이 있다. 이중 P2P 방식은 CPU 의 개입 없이 GPU 가 서로의 메모리(VRAM)에 직접 접근하여 데이터를 주고받는 기술이다. 이에 비해 SHM 방식은 CPU 개입이 필요한 2회에 걸친 메모리 복사 과정(GPU->RAM->GPU)과 메모리 대역폭 병목으로 인해 통신 성능이 저하된다. NCCL은 GPU 간 데이터 전송 시 지연 시간을 최소화하기 위해 CPU 개입 없이 GPU 메모리에 직접 접근하는 P2P(Peer-to-Peer) 통신 아키텍처를 최우선 순위로 할당한다. 만약 하드웨어 토폴로지나 시스템 제약(하이퍼바이저 설정)으로 인해 Direct Access가 차단될 경우, 시스템 메인 메모리를 중간 버퍼로 활용하는 SHM(Shared Memory) 프로토콜을 차선책(Fallback)으로 채택하여 통신 가용성을 보장하게 된다. GPUDirect RDMA 는 서로 다른 노드의 GPU 간의 통신으로 RoCE, IB, EFA 등을 사용하게 된다.  

* GPU P2P 
  * NVLink / NVSwitch
  * PCIe BUS
  * GPUDirect RDMA -  다른 노드 GPU 간의 통신 
* SHM (Shared Memory)

### P2P 지원 여부 확인 ###
아래는 g6e.12xlarge 의 GPU 토폴로지로 NODE는 CPU 통신을 의미한다.
```
# nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    NODE    NODE    0-47    0               N/A
GPU1    NODE     X      NODE    NODE    0-47    0               N/A
GPU2    NODE    NODE     X      NODE    0-47    0               N/A
GPU3    NODE    NODE    NODE     X      0-47    0               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

아래는 p4d.24xlarge 의 GPU 토폴로지로 NVLink 를 사용하고 있다.
```
# nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12    0-23,48-71      0               N/A
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12    0-23,48-71      0               N/A
GPU2    NV12    NV12     X      NV12    NV12    NV12    NV12    NV12    0-23,48-71      0               N/A
GPU3    NV12    NV12    NV12     X      NV12    NV12    NV12    NV12    0-23,48-71      0               N/A
GPU4    NV12    NV12    NV12    NV12     X      NV12    NV12    NV12    24-47,72-95     1               N/A
GPU5    NV12    NV12    NV12    NV12    NV12     X      NV12    NV12    24-47,72-95     1               N/A
GPU6    NV12    NV12    NV12    NV12    NV12    NV12     X      NV12    24-47,72-95     1               N/A
GPU7    NV12    NV12    NV12    NV12    NV12    NV12    NV12     X      24-47,72-95     1               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

### 컨테이너 필수 옵션 ###

* hostIPC: true  
  컨테이너가 호스트의 IPC(Inter-Process Communication) 네임스페이스를 공유하게 하여 GPU 간 P2P 핸드쉐이크를 가능하게 한다.
이때 주의할 점은 한 컨테이너 안에 통신에 필요한 모든 GPU를 몰아 넣어야 한다는 것이다. GPU P2P는 기본적으로 같은 메모리 주소 체계를 공유하는 동일 프로세스 또는 공유 메모리로 묶인 그룹 내에서만 작동하기 때문이다.
  ```
  apiVersion: v1
  kind: Pod
  metadata:
    name: host-ipc-example
  spec:
    hostIPC: true  # 호스트의 IPC 네임스페이스 공유 설정
    containers:
    - name: shared-memory-app
      image: ubuntu
      command: ["/bin/sh", "-c", "sleep 3600"]
  ```
  컨테이너 환경에서 host IPC(Inter-Process Communication)를 사용하면 컨테이너가 호스트 머신의 IPC 네임스페이스를 공유하게 된다. 이를 통해 컨테이너 내부의 프로세스가 호스트나 다른 컨테이너의 프로세스와 공유 메모리, 세마포어, 메시지 큐 등을 통해 직접 통신할 수 있다.

* hostNetwork: true   
  쿠버네티스에서 이 설정은 파드가 호스트(노드)의 네트워크 환경을 그대로 사용하도록 만드는 설정으로 주로 고성능 네트워킹(예: GPUDirect RDMA)이 필요한 경우, CNI의 오버헤드를 줄이기 위해 사용한다.
  호스트의 IP 주소를 그대로 사용하고 포트 역시 직접 바인딩되며, 호스트상의 모든 인터페이스 접근이 가능해 진다. 가상 네트워크(CNI)를 거치지 않으므로 패킷 캡슐화나 포워딩 오버헤드가 줄어들어 지연 시간(Latency)이 감소하고 대역폭 효율이 높아진다.
  
* resource limit:  
  nvidia.com/gpu 를 2개 이상 할당해야 단일 노드 내 P2P 통신이 가능하다

NCCL을 사용하는 경우 컨테이너 환경변수를 통해 통신 경로를 강제할 수 있다.  
* NCCL_P2P_DISABLE=0 (P2P 통신 활성화, 기본값임)
* NCCL_DEBUG=INFO 를 설정하면 P2P [NVLink] 또는 P2P [PCIe] 여부를 확인할 수 있다.

## 멀티 GPU 환경에서의 Pod 배치 ##

현재 AWS 의 가속 인스턴스들은 GPU 1, 4, 8 개 타입의 인스턴스들을 제공하고 있다. 하나의 노드에 GPU를 여러개 가지고 있는 경우 Pod를 어떤식으로 배치하는 것이 통신 효율성을 최대화 할 수 있는지에 대해서 다루고자 한다.   
결론 부터 말하자면 Pod는 노드별로 하나씩 배치하는 것이 효과적이다. 즉 8개의 GPU를 가지고 있는 EC2 인스턴스에 Pod를 배치할때 8개가 아니라 1개의 Pod를 배치하고, 해당 Pod 내부에서 8개의 파이썬 프로세스를 실행하는 것이 훨씬 유리하다.   
일반 VM 환경에서 하나의 서버가 쿠버네티스 환경에서 하나의 Pod 이고 서로 완전히 독립적인 존재로 취급되기 때문에, 여러 Pod 가 동시에 같은 공간 즉 같은 EC2 인스턴스에서 실행되더라도 GPU / CPU / Memory 와 같은 리소스를 완전히 별개이며 같은 공간(서버)에 있는 GPU 인지 아닌지 구별하지 못한다. 그러므로 인해 같은 서버에서 실행되지만 NVLink 나 PCIe 인터페이스를 통해 통신 하는게 아니라 EFA 또는 ENI 를 통해 서로 통신 하게 된다.   
물리적으로 같은 장치(서버)안에 있는 GPU 끼리도 NVLink 로 직접 쓰지 못하고 네트워크 스택을 한 번 거쳐야 하는 병목이 생길 가능성이 매우 높다.

#### 1. 통신 경로(Topology) ####
* (1 Pod x 8 GPU): 하나의 Pod(컨테이너) 안에 GPU 8장이 모두 보이는 구조로, NCCL은 이들이 같은 메모리 주소 공간에 있음을 인지하고 NVLink 또는 PCIe P2P(Peer-to-Peer)를 통해 통신한다.
* (8 Pod x 1 GPU): 각 파드는 완전히 격리된 환경에서 동작하므로 데이터를 보낼 때 GPU 0 -> 호스트 메모리 -> 네트워크 카드(EFA/TCP) -> 호스트 메모리 -> GPU 1의 복잡한 경로를 거치게 된다.

아래는 p4d.48xlarge 인스턴스에서 Pod(1 GPU /1 EFA 할당) 를 4개 띄웠을때의 NCCL 로그이다. 동일 서버에 있는 Pod 이지만, NVLink 를 사용하지 못하고 
NET/Libfabric/0/GDRDMA 즉 EFA 네트워크 인터페이스를 통해 통신하고 있다.   
```
df: /root/.triton/autotune: No such file or directory
llama-3-8b-node-0-0:188:188 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:188:188 [0] NCCL INFO Bootstrap: Using eth0:10.0.5.112<0>
llama-3-8b-node-0-0:188:188 [0] NCCL INFO cudaDriverVersion 13000
llama-3-8b-node-0-0:188:188 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
llama-3-8b-node-0-0:188:188 [0] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Successfully loaded external plugin libnccl-net.so
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Using Libfabric version 2.1
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Configuring AWS-specific options
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Setting provider_filter to efa
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Running on p4d.24xlarge platform, topology file /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Internode latency set at 75.0 us
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Using transport protocol SENDRECV (platform set)
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Creating one domain per process
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI GUID of rdmap16s27: 0000000000000000
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI GUID for dev[0]: 00000000000000000a00057000000000
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Setting FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES not supported.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Need to force simple protocol: byte delivery ordering not supported
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Support for global registrations: false
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Support for DMA-BUF registrations: false
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Adding FI_EFA_FORK_SAFE=1 to environment
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Adding NCCL_BUFFSIZE=8388608 to environment
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Adding NCCL_P2P_NET_CHUNKSIZE=524288 to environment
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Adding NCCL_PROTO=simple to environment
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Adding NCCL_TOPO_FILE=/opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml to environment
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI Adding NCCL_TUNER_PLUGIN=libnccl-net.so to environment
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Initialized NET plugin Libfabric
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:188:585 [0] NCCL INFO DMA-BUF is available on GPU device 0
llama-3-8b-node-0-0:188:585 [0] NCCL INFO ncclCommInitRankConfig comm 0x556d504b7460 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101d0 commId 0xb079519ed44b724 - Init START
llama-3-8b-node-0-0:188:585 [0] NCCL INFO RAS client listening socket at ::1<28028>
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Bootstrap timings total 8.128340 (create 0.000044, send 0.000162, recv 1.970281, ring 0.000162, delay 0.000000)
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NCCL_TOPO_FILE set by environment to /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Setting affinity for GPU 0 to 0-23,48-71
llama-3-8b-node-0-0:188:585 [0] NCCL INFO comm 0x556d504b7460 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Channel 00/02 : 0 1 2 3
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Channel 01/02 : 0 1 2 3
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Trees [0] 2/-1/-1->0->-1 [1] -1/-1/-1->0->1
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NCCL_BUFFSIZE set by environment to 8388608.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NCCL_P2P_NET_CHUNKSIZE set by environment to 524288.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO P2P Chunksize set to 524288
llama-3-8b-node-0-0:188:585 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
llama-3-8b-node-0-0:188:655 [0] NCCL INFO [Proxy Service] Device 0 CPU core 18
llama-3-8b-node-0-0:188:656 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 65
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Enabled NCCL Func/Proto/Algo Matrix:
     Function |       LL     LL128    Simple   |          Tree           Ring  CollNetDirect   CollNetChain           NVLS       NVLSTree            PAT  
    Broadcast |        0         0         1   |             1              1              1              1              1              1              1  
       Reduce |        0         0         1   |             1              1              1              1              1              1              1  
    AllGather |        0         0         1   |             1              1              1              1              1              1              1  
ReduceScatter |        0         0         1   |             1              1              1              1              1              1              1  
    AllReduce |        0         0         1   |             1              1              1              1              1              1              1  

llama-3-8b-node-0-0:188:585 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
llama-3-8b-node-0-0:188:585 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
llama-3-8b-node-0-0:188:585 [0] NCCL INFO CC Off, workFifoBytes 1048576
llama-3-8b-node-0-0:188:585 [0] NCCL INFO TUNER/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:188:585 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
llama-3-8b-node-0-0:188:585 [0] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
llama-3-8b-node-0-0:188:585 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:188:585 [0] NCCL INFO ncclCommInitRankConfig comm 0x556d504b7460 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101d0 commId 0xb079519ed44b724 - Init COMPLETE
llama-3-8b-node-0-0:188:585 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 8.31 (kernels 0.15, alloc 0.01, bootstrap 8.13, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
max_steps is given, it will override any value given in num_train_epochs
Using auto half precision backend
Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4. Using DeepSpeed's value.
Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)
[rank0]:W1231 01:46:57.058000 188 site-packages/torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
[rank0]:W1231 01:46:57.058000 188 site-packages/torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
Adam Optimizer #0 is created with AVX512 arithmetic capability.
Config: alpha=0.000020, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
llama-3-8b-node-0-0:188:188 [0] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:188:706 [0] NCCL INFO DMA-BUF is available on GPU device 0
llama-3-8b-node-0-0:188:706 [0] NCCL INFO ncclCommSplit comm 0x556d50e02a40 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101d0 parent 0x556d504b7460 splitCount 1 color 2003953581 key 0- Init START
llama-3-8b-node-0-0:188:706 [0] NCCL INFO NCCL_TOPO_FILE set by environment to /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Setting affinity for GPU 0 to 0-23,48-71
llama-3-8b-node-0-0:188:706 [0] NCCL INFO comm 0x556d50e02a40 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Channel 00/02 : 0 1 2 3
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Channel 01/02 : 0 1 2 3
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Trees [0] 2/-1/-1->0->-1 [1] -1/-1/-1->0->1
llama-3-8b-node-0-0:188:706 [0] NCCL INFO P2P Chunksize set to 524288
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
llama-3-8b-node-0-0:188:707 [0] NCCL INFO [Proxy Service] Device 0 CPU core 48
llama-3-8b-node-0-0:188:708 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 21
llama-3-8b-node-0-0:188:706 [0] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Enabled NCCL Func/Proto/Algo Matrix:
     Function |       LL     LL128    Simple   |          Tree           Ring  CollNetDirect   CollNetChain           NVLS       NVLSTree            PAT  
    Broadcast |        0         0         1   |             1              1              1              1              1              1              1  
       Reduce |        0         0         1   |             1              1              1              1              1              1              1  
    AllGather |        0         0         1   |             1              1              1              1              1              1              1  
ReduceScatter |        0         0         1   |             1              1              1              1              1              1              1  
    AllReduce |        0         0         1   |             1              1              1              1              1              1              1  

llama-3-8b-node-0-0:188:706 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
llama-3-8b-node-0-0:188:706 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
llama-3-8b-node-0-0:188:706 [0] NCCL INFO CC Off, workFifoBytes 1048576
llama-3-8b-node-0-0:188:706 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:188:706 [0] NCCL INFO ncclCommSplit comm 0x556d50e02a40 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101d0 parent 0x556d504b7460 splitCount 1 color 2003953581 key 0 - Init COMPLETE
llama-3-8b-node-0-0:188:706 [0] NCCL INFO Init timings - ncclCommSplit: rank 0 nranks 4 total 2.79 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.02, topo 0.01, graphs 0.00, connections 0.00, rest 2.75)
llama-3-8b-node-0-0:188:710 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 56
llama-3-8b-node-0-0:188:709 [0] NCCL INFO Channel 00/0 : 3[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:709 [0] NCCL INFO Channel 01/0 : 3[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:709 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:709 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:709 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
Stage 3 initialize beginning
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 52.66 GB, percent = 4.7%
DeepSpeedZeRoOffload initialize [begin]
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 53.11 GB, percent = 4.7%
llama-3-8b-node-0-0:188:712 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 67
llama-3-8b-node-0-0:188:711 [0] NCCL INFO Channel 00/0 : 3[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:711 [0] NCCL INFO Channel 01/0 : 3[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:711 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:711 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:711 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
Parameter Offload - Persistent parameters statistics: param_count = 65, numel = 266240
DeepSpeedZeRoOffload initialize [end]
MA 0.0 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 69.59 GB, percent = 6.2%
Before creating fp16 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 69.29 GB, percent = 6.2%
llama-3-8b-node-0-0:188:760 [0] NCCL INFO Channel 00/0 : 2[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:760 [0] NCCL INFO Channel 00/0 : 0[0] -> 2[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:760 [0] NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:760 [0] NCCL INFO Connected all trees
After creating fp16 partitions: 3
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 113.11 GB, percent = 10.1%
Before creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 111.26 GB, percent = 9.9%
After creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 116.03 GB, percent = 10.3%
Before initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 136.5 GB, percent = 12.2%
After initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 155.25 GB, percent = 13.8%
llama-3-8b-node-0-0:188:761 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:761 [0] NCCL INFO Channel 00/0 : 0[0] -> 3[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:761 [0] NCCL INFO Channel 01/0 : 0[0] -> 3[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:761 [0] NCCL INFO Channel 01/0 : 2[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:761 [0] NCCL INFO Channel 01/0 : 0[0] -> 2[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:761 [0] NCCL INFO Connected binomial trees
After initializing ZeRO optimizer
MA 0.03 GB         Max_MA 1.99 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 172.96 GB, percent = 15.4%
***** Running training *****
  Num examples = 36,718
  Num Epochs = 1
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 50
  Number of trainable parameters = 8,030,261,248
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 00/0 : 0[0] -> 3[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 01/0 : 0[0] -> 3[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 00/0 : 2[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 01/0 : 2[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 00/0 : 0[0] -> 2[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Channel 01/0 : 0[0] -> 2[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:188:764 [0] NCCL INFO Connected binomial trees
llama-3-8b-node-0-0:188:765 [0] NCCL INFO Connected all trees
```
* 3[0] -> 글로벌 랭크 3 / 로컬 랭크 0  

#### 2. 성능 차이 (Bottleneck) ###
* NVLink 속도: 최신 GPU(A100/H100/GB200) 기준 노드 내부 통신은 보통 600 GB/s [NVLink 3.0] ~ 1,800 GB/s [NVLink 5.0]
* 네트워크(EFA) 속도: EFA 는 100Gbps ~ 400Gbps (약 12.5GB/s ~ 50GB/s) 수준 제공.

#### 3. 기술적 예외 (Pod Affinity & Shared Memory) ####
hostNetwork: true를 사용하고 IPC 설정을 정교하게 하면 파드가 달라도 NVLink를 쓸 수는 있지만, 설정이 매우 까다롭고 보안상 권장되지 않는다.


### cf. GPU별 개별 Pod 설정 ###

만약, 운영상 개별 GPU 별로 하나의 Pod 를 할당하고 싶다면 아래와 같은 설정으로 가능하다. 하지만 이는 성능을 대가로 관리 편의성을 얻는 선택으로 권장하진 않는다. 
아래 예시에서는 16개의 노드(Pod)를 분산 훈련에 사용하고 있는데 노드(Pod)당 1개의 프로세스를 띄우고 있으며, 파드당 리소스는 1 GPU / 1 EFA 인터페이스를 할당하고 있다.
노드 할당을 담당하는 카펜터는 GPU를 8장 탑재하고 있는 p4d.48xlarge 와 같은 인스턴스를 2대 띄우거나 GPU 4장을 가진 인스턴스를 4 대 띄우거나 아니면 GPU 1 장을 가진 인스턴스를 16대 띄울 수도 있다.  
```
trainer:
    numNodes: 16                               # Pod(노드 단위)를 16개 할당
    numProcPerNode: 1                          # Pod 내부에서 프로세스는 1개만 실행  
    image: ...                                 # Pod 의 리소스 limit 설정이 nvidia.com: "1" 이므로 1 이상의 값을 주면 에러 발생.

    command:
        # ... (중략) ...
        torchrun \
          --nnodes=16 \                        # 전체 노드 수를 16으로 명시 (생략가능)
          --nproc_per_node=1 \                 # Pod당 1개 프로세스만 생성 명시 (생략가능)
          --rdzv_id=llama-3-8b-job \
          --rdzv_backend=c10d \
          --rdzv_endpoint=${PET_MASTER_ADDR}:${PET_MASTER_PORT} \
          llama-3-8b.py 

    resourcesPerNode:
      limits:
        nvidia.com: "1"                        # Pod GPU를 1개로 제한
        vpc.amazonaws.com: "1"                 # (EFA 사용 시) 1개 할당
```


