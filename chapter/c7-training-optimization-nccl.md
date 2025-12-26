<< 테스트 필요 >>

## 클러스터 배치 그룹(Cluster Placement Group) ##
네트워크 지연 시간을 줄이려면, EKS 노드 그룹 생성 시 AWS 수준에서 Cluster Placement Group을 적용해야 한다. 

### EKS 노드그룹 -- 확인필요 ###
노드그룹은 placement group 및 Capacity Block 설정을 지원한다. 
```
aws ec2 create-placement-group --group-name "ml-training-pg" --strategy cluster

eksctl create nodegroup \
  --cluster ${CLUSTER_NAME} \
  --name efa-gpu-nodes \
  --node-type p4d.24xlarge \
  --placement-group "ml-training-pg" \
  --nodes 8 \
  --node-zones "ap-northeast-2a"       # 반드시 동일 AZ 지정
```
cluster 배치 그룹 내에서 노드 생성이 실패(InsufficientCapacity)한다면, 해당 AZ에 가용 자원이 부족한 것이다. 
이 경우 다른 AZ를 시도하거나 AWS EC2 Capacity Blocks를 통해 사전에 자원을 예약하는 방식이 있다.
```
eksctl create nodegroup \
  --cluster ${CLUSTER_NAME} \
  --name ml-capacity-block-nodes \
  --node-type p5.48xlarge \
  --nodes 16 \
  --placement-group "ml-training-pg" \
  --capacity-reservation-id cr-0123456789abcdefg \
  --node-zones "ap-northeast-2a"
```

### 카펜터 Capacity Block ###
카펜터는 ODCR 과 Capacity Block 설정을 지원하지만 (아래 메뉴얼 참고), Placement Group 은 명시적으로 지원하지 않는다. 
* https://karpenter.sh/docs/tasks/odcrs/
* https://karpenter.sh/docs/concepts/nodeclasses/
* https://karpenter.sh/docs/concepts/nodepools/
  
## NCCL ##
분산 훈련 환경(특히 EKS와 EFA가 활성화된 환경)에서 GPU 간 및 노드 간 통신 경로를 확인하기 위해 NCCL Topology를 확인하는 방법은 크게 두가지 이다. 

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


## 레퍼런스 ##

* https://github.com/eksctl-io/eksctl/tree/main
  
