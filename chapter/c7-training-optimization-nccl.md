<< 작성중 ..>>

## NCCL Topology ##
분산 훈련 환경에서 GPU 및 노드 간의 통신 경로 즉, NCCL Topology를 확인하는 방법은 크게 두가지가 있다. 

### 1. NCCL 디버그 로그 활성화 ###
#### 환경변수 ####
Pod를 실행할 때 환경 변수를 설정하면, NCCL이 초기화될 때 감지된 모든 하드웨어 토폴로지(EFA 인터페이스, PCIe 스위치 구조 등)를 표준 출력(stdout)으로 보여준다.
```
env:
- name: NCCL_DEBUG
  value: "INFO"
- name: NCCL_DEBUG_SUBSYS
  value: "GRAPH,INIT,ENV"
```
#### Pod 로그 확인 ####
로그를 통해 EFA 인터페이스가 정상적으로 감지되었는지, GPU들이 NVLink로 연결되었는지 확인할 수 있다.
```
kubectl logs <pod-name> | grep -i "topology"
```
### 2. nvidia-smi topo -m 명령 ###
Pod 내부 또는 EKS 노드에 직접 접속하여 현재 호스트의 GPU 간 연결 상태를 매트릭스 형태로 확인할 수 있다.
```
kubectl exec -it <pod-name> -- nvidia-smi topo -m
```
* NV#: GPU 간 NVLink로 연결됨 (가장 빠름)
* PHB/PIX: PCIe 스위치 또는 호스트 브리지를 통해 연결됨
* NODE: 다른 CPU 소켓 또는 원격 노드 (네트워크 통신 필요)

## NCCL Tests ##
네트워크를 포함한 전체 토폴로지의 성능을 측정하려면 NVIDIA에서 제공하는 nccl-tests를 사용해야 한다.
* 일반적으로 nccl-tests는 MPI와 연동되어 실행되므로, 단순 실행보다는 EKS 노드 내의 전용 컨테이너 또는 KubeFlow/MPI Operator 환경에서 실행하는 것이 정확하다.
* EFA를 사용하는 경우, 실행 시 LD_LIBRARY_PATH에 Libfabric 경로를 포함해야 하며, FI_EFA_USE_DEVICE_RDMA=1과 같은 EFA 전용 환경 변수를 함께 사용하여 토폴로지 최적화 여부를 확인해야 한다.

### 싱글노드 (8개 GPU 기준) ###
```
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=INFO

./all_reduce_perf -b 8 -e 1G -f 2 -g 8
```
* algbw (Algorithm Bandwidth): 실제 데이터 통신 속도로 EFA가 활성화된 경우 노드 간 통신에서 이 수치가 네트워크 대역폭(예: P4d의 경우 400Gbps)에 근접해야 한다.
* busbw (Bus Bandwidth): 하드웨어 토폴로지(NVLink 등)의 효율성을 나타낸다.
* NCCL INFO 로그: 실행 초기 출력되는 로그에서 NET/OFI 문구가 보인다면, NCCL이 일반 이더넷이 아닌 EFA(Libfabric) 토폴로지를 정상적으로 인식하고 사용 중임을 의미한다.
* 인스턴스 타입: P4d, P4de, P5 인스턴스를 사용 중이라면 각 노드에 4개 또는 그 이상의 EFA 인터페이스가 존재한다. nccl-tests 실행 시 모든 인터페이스가 토폴로지에 포함되었는지 확인이 필요하다.


### 멀티노드 ###

#### 1. 멀티 노드 테스트를 위한 핵심 조건 ####
* MPI 환경: 노드 간 통신을 조율하기 위해 mpirun 또는 Kubernetes의 MPI Operator(Kubeflow)가 필요합니다.
* Hostfile: 테스트에 참여할 노드(Pod)들의 IP 목록이 필요합니다.
* EFA 활성화: 모든 노드에 EFA 인터페이스가 할당되어 있어야 합니다.

#### 2. 멀티 노드 실행 명령어 예시 (2대 노드, 총 16개 GPU) ####
Pod 내부에 MPI가 설치되어 있다고 가정할 때, 실행 방식은 다음과 같습니다.
```
# 1. 환경 변수 설정 (노드 간 EFA 통신 강제)
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=INFO

# 2. mpirun을 통한 실행
# -n 16: 전체 GPU 개수 (2개 노드 * 8개)
# -N 8: 노드당 프로세스 개수
# --hostfile hosts: 노드 IP가 적힌 파일
mpirun -n 16 -N 8 --hostfile hosts \
    --allow-run-as-root \
    -x FI_PROVIDER=efa \
    -x FI_EFA_USE_DEVICE_RDMA=1 \
    -x NCCL_DEBUG=INFO \
    ./all_reduce_perf -b 8 -e 1G -f 2 -g 1
```
(참고: -g 1은 각 프로세스가 1개의 GPU를 담당한다는 의미이며, MPI와 결합하여 노드당 8개를 사용하게 됩니다.)

#### 3. 결과 해석 ####
이 테스트 결과에서 다음을 확인해야 합니다.
* algbw (Algorithm Bandwidth): 노드 간 EFA를 거쳐 전송된 실제 속도입니다.
* 성능 저하 여부: 단일 노드 테스트 결과보다 현저히 낮다면 Placement Group(배치 그룹)이 적용되지 않았거나, Security Group(보안 그룹)에서 EFA용 자기 참조 규칙이 빠졌을 가능성이 큽니다.

#### 4. 2025년 EKS 권장 도구: Kubeflow MPI Operator ####
* EKS에서 이를 가장 쉽게 수행하는 방법은 MPIJob 리소스를 사용하는 것입니다.
* Kubeflow MPI Operator를 설치합니다.
* MPIJob YAML 파일에 replicas: 2, procPerNode: 8을 설정하여 배포합니다.
* EFA 설정을 template의 env 섹션에 포함합니다.


## NCCL Topoloty 설정 ##
* 강제로 경로를 설정하는 방법을 다룬다.


----
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

[yaml 예시]
```
spec:
  trainer:
    env:
      - name: NCCL_DEBUG
        value: "INFO"
      - name: NCCL_IB_DISABLE
        value: "0"  # RDMA/EFA 활성화
      - name: NCCL_SOCKET_IFNAME
        value: "eth,en" # 실제 노드의 인터페이스명 확인 필요
      - name: NCCL_BUFFSIZE
        value: "2097152"
      - name: FI_EFA_USE_DEVICE_RDMA
        value: "1"
    # ... 이전의 securityContext 및 command 설정 유지

```
### NCCL 알고리즘 선택 ###
최근 대규모 클러스터에서는 NCCL이 자동으로 선택하는 알고리즘이 항상 최적은 아닐 수 있습니다.
* NCCL_ALGO=Tree: 노드 수가 매우 많을 때 지연 시간을 줄이는 데 유리합니다.
* NCCL_ALGO=Ring: 대규모 데이터를 전송할 때 대역폭 활용도가 높습니다.
보통은 기본값(Automatic)을 쓰되, 특정 규모에서 성능 저하가 보인다면 두 값을 직접 테스트해보는 것이 좋습니다.
이러한 설정들은 앞서 논의한 NET_ADMIN과 privileged: true 권한이 있어야만 커널 및 하드웨어 레벨에서 정상적으로 동작합니다. NVIDIA NCCL 공식 가이드에서 더 자세한 변수 목록을 확인하실 수 있습니다.

## 레퍼런스 ##

* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
