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
  
* resource limit:  
  nvidia.com/gpu 를 2개 이상 할당해야 단일 노드 내 P2P 통신이 가능하다

NCCL을 사용하는 경우 컨테이너 환경변수를 통해 통신 경로를 강제할 수 있다.  
* NCCL_P2P_DISABLE=0 (P2P 통신 활성화, 기본값임)
* NCCL_DEBUG=INFO 를 설정하면 P2P [NVLink] 또는 P2P [PCIe] 여부를 확인할 수 있다.

## 멀티 GPU 환경에서의 Pod 배치 ##

현재 AWS 의 가속 인스턴스들은 GPU 1, 4, 8 개 타입의 인스턴스들을 제공하고 있다. 하나의 노드에 GPU를 여러개 가지고 있는 경우 Pod를 어떤식으로 배치하는 것이 통신 효율성을 최대화 할 수 있는지에 대해서 다루고자 한다.   
결론 부터 말하자면 Pod는 노드별로 하나씩 배치하는 것이 효과적이다. 즉 8개의 GPU를 가지고 있는 EC2 인스턴스에 Pod를 배치할때 8개가 아니라 1개의 Pod를 배치하고, 해당 Pod 내부에서 8개의 파이썬 프로세스를 실행하는 것이 훨씬 유리하다.   
일반 VM 환경에서 하나의 서버가 쿠버네티스 환경에서 하나의 Pod 이고 서로 완전히 독립적인 존재로 취급되기 때문에, 여러 Pod 가 동시에 같은 공간 즉 같은 EC2 인스턴스에서 실행되더라도 GPU / CPU / Memory 와 같은 리소스를 완전히 별개이며 같은 공간(서버)에 있는 GPU 인지 아닌지 구별하지 못한다. 그러므로 인해 같은 서버에서 실행되지만 NVLink 나 PICe 인터페이스를 통해 통신 하는게 아니라 EFA 또는 ENI 를 통해 서로 통신 하게 된다.   
물리적으로 같은 장치(서버)안에 있는 GPU 끼리도 NVLink 로 직접 쓰지 못하고 네트워크 스택을 한 번 거쳐야 하는 병목이 생길 가능성이 매우 높다.

#### 1. 통신 경로(Topology) ####
* (1 Pod x 8 GPU): 하나의 Pod(컨테이너) 안에 GPU 8장이 모두 보이는 구조로, NCCL은 이들이 같은 메모리 주소 공간에 있음을 인지하고 NVLink 또는 PCIe P2P(Peer-to-Peer)를 통해 통신한다.
* (8 Pod x 1 GPU): 각 파드는 완전히 격리된 환경에서 동작하므로 데이터를 보낼 때 GPU 0 -> 호스트 메모리 -> 네트워크 카드(EFA/TCP) -> 호스트 메모리 -> GPU 1의 복잡한 경로를 거치게 된다.

```
llama-3-8b-node-0-0:195:1185 [3] NCCL INFO Channel 00 : 3[3] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:195:1185 [3] NCCL INFO Channel 01 : 3[3] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:194:1186 [2] NCCL INFO Channel 00 : 2[2] -> 1[1] via SHM/direct/direct

llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 01/0 : 0[0] -> 4[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 00/0 : 8[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/Socket/0

llama-3-8b-node-0-0:253:2401 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
llama-3-8b-node-0-0:253:2401 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
llama-3-8b-node-0-0:253:2401 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/CUMEM/read

llama-3-8b-node-0-0:195:1185 [3] NCCL INFO Connected all trees
```
* 3[3] -> 글로벌 랭크 3 / 로컬 랭크 3, 8[0] -> 글로벌 랭크 8 / 로컬 랭크 0  

#### 2. 성능 차이 (Bottleneck) ###
* NVLink 속도: 최신 GPU(A100/H100/GB200) 기준 노드 내부 통신은 보통 600 GB/s [NVLink 3.0] ~ 1,800 GB/s [NVLink 5.0]
* 네트워크(EFA) 속도: EFA 는 100Gbps ~ 400Gbps (약 12.5GB/s ~ 50GB/s) 수준 제공.

#### 3. 기술적 예외 (Pod Affinity & Shared Memory) ####
hostNetwork: true를 사용하고 IPC 설정을 정교하게 하면 파드가 달라도 NVLink를 쓸 수는 있지만, 설정이 매우 까다롭고 보안상 권장되지 않는다.


#### cf. GPU별 개별 Pod 설정 ####

만약, 운영상 개별 GPU 별로 하나의 Pod 를 할당하고 싶다면 아래와 같은 설정으로 가능하다. 하지만 이는 성능을 대가로 관리 편의성을 얻는 선택으로 권장하진 않는다. 
```
trainer:
    numNodes: 16                               # 파드(노드 단위)를 16개로 증가
    numProcPerNode: 1                          # 파드당 프로세스는 1개만 실행
    image: ...

    command:
        # ... (중략) ...
        torchrun \
          --nnodes=16 \                        # 전체 노드 수를 16으로 명시
          --nproc_per_node=1 \                 # 파드당 1개 프로세스만 생성
          --rdzv_id=llama-3-8b-job \
          --rdzv_backend=c10d \
          --rdzv_endpoint=${PET_MASTER_ADDR}:${PET_MASTER_PORT} \
          llama-3-8b.py 

    resourcesPerNode:
      limits:
        nvidia.com: "1"                        # 파드당 GPU를 1개로 제한
        vpc.amazonaws.com: "1"                 # (EFA 사용 시) 1개 할당
```
## NCCL & P2P ##

### NCCL Shared Memory 통신의 이해 ####
#### 1. SHM/direct/direct의 의미 ####
SHM (Shared Memory): 데이터가 GPU 간에 직접 오가는 대신, 호스트의 메인 메모리(RAM)를 거쳐서 전달됨을 뜻이다.
Direct/Direct: 데이터 복사 과정에서 중간 단계를 최소화하고 메모리 영역에 직접 접근(Direct Access)하여 읽고 쓰는 방식을 사용한다는 로그 상의 세부 상태이다. 

#### 2. 왜 PCIe(P2P)가 아닌 SHM이 사용되었을까? ####
NCCL은 보통 P2P(Peer-to-Peer) 통신(NVLink 또는 PCIe Direct)을 우선순위로 두지만, 다음과 같은 경우 SHM으로 폴백(Fallback)한다. 
* 격리 환경의 제한: 컨테이너나 가상화 환경에서 GPU 간 직접적인 P2P 접근 권한이 설정되지 않았을 때 발생한다.
* 하드웨어 구조 문제: GPU들이 서로 다른 CPU 소켓(NUMA 노드)에 연결되어 있어 PCIe를 통한 직접 통신이 비효율적이라고 판단될 때, 차선책으로 시스템 메모리를 통한 통신을 선택한다.
* P2P 비활성화: 환경 변수(NCCL_P2P_DISABLE=1) 등을 통해 직접 통신이 명시적으로 꺼져 있는 경우이다. 

#### 3. 성능 영향 ####
* 속도 차이: PCIe를 통한 직접 통신(P2P)보다 호스트 메모리를 거치는 SHM 방식이 일반적으로 지연 시간이 길고 대역폭이 낮아 학습 성능이 저하될 수 있다.
* 해결 방법: 만약 하드웨어가 P2P를 지원한다면, 컨테이너 실행 시 --ipc=host 옵션이나 --privileged 옵션, 혹은 Kubernetes의 hostIPC: true 설정을 통해 격리를 완화하면 P2P(PCIe/NVLink) 통신이 활성화될 수 있다.

