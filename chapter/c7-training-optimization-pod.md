## 멀티 GPU 통신 토폴로지 ##

EC2 인스턴스 또는 하나의 Pod 내부에서 사용 가능한 GPU 간의 통신은 다음과 같이 3가지 종류가 있다. 이중 GPU P2P 는 SHM(Shared Memory) 방식과는 달리 CPU 및 시스템 메모리(RAM) 를 사용하지 않기 때문에 훨씬 빠른 전송 속도를 제공해 준다.    
GPU P2P(Peer-to-Peer)는 멀티 GPU 시스템 내에서 두 개 이상의 GPU가 CPU나 메인 메모리(RAM)를 거치지 않고 서로의 메모리에 직접 접근하여 데이터를 주고받는 기술이다.
분산 훈련시 NCCL 로그에 각각 [NVL], [P2P], [SHM] 형태로 기록이 되는데 통신 성능은 [NVL] >>> [P2P] > [SHM] 이다.

* GPU P2P 
  * NVLink / NVSwitch
  * PCIe BUS
* SHM (Shared Memory)

### P2P 지원 여부 확인 ###
토폴로지 확인은 호스트 터미널에서 nvidia-smi topo -m 를 이용하면 확인할 수 있다. 
* NV# (예: NV1, NV2): NVLink로 연결됨 (최상급 속도).
* PIX, PXB, PHB: PCIe를 통해 연결됨 
* SYS 또는 SOC: CPU를 거쳐야 하므로 P2P 통신이 불가능하고 성능이 낮음.

### 컨테이너 실행 필수 옵션 ###
* hostIPC: true
컨테이너가 호스트의 IPC(Inter-Process Communication) 네임스페이스를 공유하게 하여 GPU 간 P2P 핸드쉐이크를 가능하게 한다.
* resource limit:
nvidia.com/gpu 를 2개 이상 할당해야 단일 노드 내 P2P 통신이 가능하다

P2P 통신 라이브러리인 NCCL을 사용하는 경우 컨테이너 환경변수를 통해 통신 경로를 강제할 수 있다.  
* NCCL_P2P_DISABLE=0 (기본값이지만 명시 가능)
* NCCL_DEBUG=INFO 를 설정하여 P2P [NVLink] 또는 P2P [PCIe] 여부를 확인할 수 있다.

## 멀티 GPU 환경에서의 Pod 배치 ##

현재 AWS 의 가속 인스턴스들은 GPU 1, 4, 8 개 타입의 인스턴스들을 제공하고 있다. 하나의 노드에 GPU를 여러개 가지고 있는 경우 Pod를 어떤식으로 배치하는 것이 통신 효율성을 최대화 할 수 있는지에 대해서 다루고자 한다.   
결론 부터 말하자면 Pod는 노드별로 하나씩 배치하는 것이 효과적이다. 즉 8개의 GPU를 가지고 있는 EC2 인스턴스에 Pod를 배치할때 8개가 아니라 1개의 Pod를 배치하고, 해당 Pod 내부에서 8개의 파이썬 프로세스를 실행하는 것이 훨씬 유리하다.   
일반 VM 환경에서 하나의 서버가 쿠버네티스 환경에서 하나의 Pod 이고 서로 완전히 독립적인 존재로 취급되기 때문에, 여러 Pod 가 동시에 같은 공간 즉 같은 EC2 인스턴스에서 실행되더라도 GPU / CPU / Memory 와 같은 리소스를 완전히 별개이며 같은 공간(서버)에 있는 GPU 인지 아닌지 구별하지 못한다. 그러므로 인해 같은 서버에서 실행되지만 NVLink 나 PICe 인터페이스를 통해 통신 하는게 아니라 EFA 또는 ENI 를 통해 서로 통신 하게 된다.   
물리적으로 같은 장치(서버)안에 있는 GPU 끼리도 NVLink 로 직접 쓰지 못하고 네트워크 스택을 한 번 거쳐야 하는 병목이 생길 가능성이 매우 높다.

#### 1. 통신 경로(Topology) ####
* (4파드 x  8GPU): 하나의 파드(컨테이너) 안에 GPU 8장이 모두 보이는 구조로, NCCL은 이들이 같은 메모리 주소 공간에 있음을 인지하고 NVLink 또는 PCIe P2P(Peer-to-Peer)를 통해 통신한다.
* (32파드 x 1GPU): 각 파드는 완전히 격리된 환경에서 동작하므로 데이터를 보낼 때 GPU 0 -> 호스트 메모리 -> 네트워크 카드(EFA/TCP) -> 호스트 메모리 -> GPU 1의 복잡한 경로를 거치게 된다.

```
llama-3-8b-node-0-0:195:1185 [3] NCCL INFO Channel 00 : 3[3] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:195:1185 [3] NCCL INFO Channel 01 : 3[3] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 01/0 : 0[0] -> 4[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:194:1186 [2] NCCL INFO Channel 00 : 2[2] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:194:1186 [2] NCCL INFO Channel 01 : 2[2] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:193:1187 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
llama-3-8b-node-0-0:193:1187 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 00/0 : 8[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:195:1185 [3] NCCL INFO Connected all trees
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Channel 01/0 : 4[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:194:1186 [2] NCCL INFO Connected all trees
llama-3-8b-node-0-0:192:1188 [0] NCCL INFO Connected all trees
```
* 3[3] -> 글로벌 랭크 3 / 로컬 랭크 3, 8[0] -> 글로벌 랭크 8 / 로컬 랭크 0  

#### 2. 성능 차이 (Bottleneck) ###
* NVLink 속도: 최신 GPU(A100/H100/GB200) 기준 노드 내부 통신은 보통 600 GB/s [NVLink 3.0] ~ 1,800 GB/s [NVLink 5.0]
* 네트워크(EFA) 속도: EFA 는 100Gbps ~ 400Gbps (약 12.5GB/s ~ 50GB/s) 수준 제공.

#### 3. 기술적 예외 (Pod Affinity & Shared Memory) ####
hostNetwork: true를 사용하고 IPC 설정을 정교하게 하면 파드가 달라도 NVLink를 쓸 수는 있지만, 설정이 매우 까다롭고 보안상 권장되지 않는다.
컨테이너 환경에서 리소스 격리가 기본 원칙이기 때문에 동일 노드의 다른 파드의 GPU 정보를 확인할 수 없다.

#### cf. GPU별 개별 Pod 설정 ####
만약, 개별 GPU 별로 하나의 Pod 를 할당하고 싶다면 아래와 같은 설정으로 가능하다. 하지만 이는 성능을 대가로 관리 편의성을 얻는 선택이 된다. 
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

 
