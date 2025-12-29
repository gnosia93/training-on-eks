## 멀티 GPU 환경에서의 Pod 배치 ##

현재 AWS 의 가속 인스턴스들은 GPU 1, 2, 4, 8 개 타입의 인스턴스들을 제공하고 있다. 하나의 노드에 GPU를 여러개 가지고 있는 경우 Pod를 어떤식으로 배치하는 것이 통신 효율성을 최대화 할 수 있는지에 대해서 다루고자 한다.
결론 부터 말하자면 Pod는 노드별로 하나씩 배치하는 것이 효과적이다. 
즉 8개의 GPU를 가지고 있는 EC2 인스턴스에 Pod를 배치할때 8개가 아니라 1개의 Pod를 배치하고, 해당 Pod 내부에서 8개의 파이썬 프로세스를 실행하는 것이 훨씬 유리하다.
일반 VM 환경에서 하나의 서버가 쿠버네티스 환경에서 하나의 Pod 이고 서로 완전히 독립적인 존재로 취급되기 때문에, 여러 Pod 가 동시에 같은 공간 즉 같은 EC2 인스턴스에서 실행되더라도 GPU / CPU / Memory 와 같은 리소스를 완전히 별개이며 같은 공간(서버)에 있는 GPU 인지 아닌지 구별하지 못한다. 그러므로 인해 같은 서버에서 실행되지만 NVLink 나 PICe 인터페이스를 통해 통신 하는게 아니라 EFA 또는 ENI 를 통해 서로 통신 하게 된다.   
물리적으로 같은 장치(서버)안에 있는 GPU 끼리도 NVLink 로 직접 쓰지 못하고 네트워크 스택을 한 번 거쳐야 하는 병목이 생길 가능성이 매우 높다.

#### 1. 통신 경로(Topology) ####
* (4파드 x  8GPU): 하나의 파드(컨테이너) 안에 GPU 8장이 모두 보이는 구조로, NCCL은 이들이 같은 메모리 주소 공간에 있음을 인지하고 NVLink 또는 PCIe P2P(Peer-to-Peer)를 통해 통신한다.
* (32파드 x 1GPU): 각 파드는 완전히 격리된 환경에서 동작하므로 데이터를 보낼 때 GPU 0 -> 호스트 메모리 -> 네트워크 카드(EFA/TCP) -> 호스트 메모리 -> GPU 1의 복잡한 경로를 거치게 된다.

#### 2. 성능 차이 (Bottleneck) ###
* NVLink 속도: 최신 GPU(A100/H100) 기준 노드 내부 통신은 보통 300GB/s ~ 900GB/s.
* 네트워크(EFA) 속도: EFA 는 100Gbps ~ 400Gbps (약 12.5GB/s ~ 50GB/s) 수준 제공.
즉, 같은 물리 노드 안에 있는 GPU끼리 대화하는데 속도가 10배 이상 느려지는 결과를 초래한다. 

#### 3. 기술적 예외 (Pod Affinity & Shared Memory) ####
hostNetwork: true를 사용하고 IPC 설정을 정교하게 하면 파드가 달라도 NVLink를 쓸 수는 있지만, 설정이 매우 까다롭고 보안상 권장되지 않는다.


#### cf. GPU별 개별 Pod 설정 ####
만약 개별 GPU 별로 하나의 Pod 를 할당하고 싶다면 아래와 같은 설정으로 가능하다. 하지만 이는 성능을 대가로 관리 편의성을 얻는 선택이 된다. 
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

 
