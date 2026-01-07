## EFA 사용하기 ##

일반적인 TCP 통신은 컨테이너 네트워크 계층을 거치면서 성능 저하가 발생할 수 있지만, 컨테이너 환경에서는 EFA는 EC2와 동일한 OS Bypass 방식으로 동작 한다. 
EFA Device Plugin이 호스트의 하드웨어 장치(/dev/infiniband/uverbsX)를 컨테이너 내부로 직접 넣어주면, NCCL(libfabric)이 컨테이너의 네트워크 스택을 아예 무시하고 EFA 전용 네트워크 카드에 직접 데이터를 보내게 된다. 따라서 컨테이너에서 호스트 네트워크를 사용할 수 있는 hostNetwork 설정값이 false 라도 EFA만 제대로 인식된다면 네이티브 EC2와 동일한 성능을 낼 수 있다. 컨테이너 환경에서의 EFA 역시 중간에 거치는 소프트웨어 계층이 없기 때문에 지연 시간(Latency)이나 처리량(Throughput)에서 손실이 발생할 구조적 원인이 없다.

#### VPC CNI 우회 ####
분산 학습의 성능을 결정짓는 NCCL/EFA 통신 단계에서
* 통로: NCCL 라이브러리가 컨테이너 내부에 노출된 /dev/infiniband/uverbsX 장치에 직접 데이터를 쓴다.
* 흐름: 애플리케이션 RAM/GPU 메모리 → EFA 하드웨어(NIC) → 네트워크
* 결과: VPC CNI의 네트워크 네임스페이스나 커널 오버헤드가 전혀 발생하지 않는다. (VPC CNI는 아무런 관여를 하지 않음)


### 1. EFA 지원 GPU 인스턴스 ###
```
aws ec2 describe-instance-types \
    --filters Name=network-info.efa-supported,Values=true \
    --query "InstanceTypes[?GpuInfo.Gpus!=null].InstanceType" \
    --output text | sort
```
[결과]
```
g5.16xlarge     p4d.24xlarge
g5.8xlarge      g4dn.8xlarge    g6e.24xlarge    gr6.8xlarge     g5.48xlarge
g6.24xlarge     g6e.48xlarge    g6e.8xlarge
g6.48xlarge     g4dn.metal      g6e.16xlarge    g4dn.12xlarge   g6.8xlarge      g5.24xlarge
g6e.12xlarge    g5.12xlarge
p5en.48xlarge   g6.16xlarge     g6.12xlarge     g4dn.16xlarge
```

인스턴스별 EFA 상세정보는 아래와 같이 조회할 수 있다.
```
$ aws ec2 describe-instance-types \
    --instance-types p4d.24xlarge \
    --query "InstanceTypes[*].{InstanceType:InstanceType, \
        EfaSupported:NetworkInfo.EfaSupported, \
        MaxNetworkInterfaces:NetworkInfo.MaximumNetworkInterfaces, \
        MaxEfaInterfaces: NetworkInfo.EfaInfo.MaximumEfaInterfaces, \
        NetworkPerformance:NetworkInfo.NetworkPerformance}" --output table
----------------------------------------------------------------------------------------------------
|                                       DescribeInstanceTypes                                      |
+--------------+---------------+-------------------+------------------------+----------------------+
| EfaSupported | InstanceType  | MaxEfaInterfaces  | MaxNetworkInterfaces   | NetworkPerformance   |
+--------------+---------------+-------------------+------------------------+----------------------+
|  True        |  p4d.24xlarge |  4                |  60                    |  4x 100 Gigabit      |
+--------------+---------------+-------------------+------------------------+----------------------+
```
* 최대 60개의 NIC 중 4개가 EFA 이고 나머지는 ENI(ENA)
* 100Gbp의 EFA 4개


### 2. EFA 디바이스 플러그인 배포 ### 

EFA의 핵심 기술인 OS Bypass(커널을 거치지 않고 하드웨어에 직접 접근)가 EKS에서도 Device Plugin을 통한 Pass-through 방식으로 동일하게 구현되어 있다.
EFA 용 디바이스 플러그인을 설치한다. 노드의 Taint 설정으로 인해서 데몬 파드가 랜딩하지 못하는 경우 있는 관계로, 아래와 같이 모든 테인트를 무력화 시키는 오퍼레이터를 추가해 준다. (- operator: Exists)
실제 해당 노드에서는 nvidia.com/gpu 및 vpc.amazonaws.com/efa 등과 같은 테인트가 존재할 수 있다. 
```
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin --namespace kube-system

kubectl patch ds aws-efa-k8s-device-plugin -n kube-system --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/tolerations/-", "value": {"operator": "Exists"}}
]'

kubectl get ds aws-efa-k8s-device-plugin -n kube-system
```
[결과]
``` 
NAME                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
aws-efa-k8s-device-plugin   0         0         0       0            0           <none>          13s
```

### 3. EKS 노드 시큐리티 그룹 확인 ### 

AWS Elastic Fabric Adapter(EFA)는 OS 커널을 우회(OS bypass)하는 방식을 사용하므로, 보안 그룹(Security Group) 설정 시 다음과 같은 엄격한 규칙이 필수적이다. 
표준 TCP/UDP 스택이 아닌 전용 프로토콜(Scalable Reliable Datagram, SRD)을 사용하여 네트워크 카드 수준에서 직접 통신하므로 특정 포트나 프로토콜만 허용하면 이 저수준 통신이 차단된다.
EFA를 사용하는 인스턴스들은 클러스터 형태로 서로 통신하므로 보안 그룹이 자기 자신을 소스/대상으로 허용해야만 노드 간 제한 없는 고성능 데이터 전송이 가능하다.

* 인바운드(Inbound):
  * 유형: 모든 트래픽 (All Traffic)
  * 프로토콜: 전체 (All)
  * 소스(Source): 자기 자신(현재 보안 그룹 ID)
    
* 아웃바운드(Outbound):
  * 유형: 모든 트래픽 (All Traffic)
  * 프로토콜: 전체 (All)
  * 대상(Destination): 자기 자신(현재 보안 그룹 ID)

```
# EFA 노드들이 사용할 보안 그룹 ID
NODE_SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=tag:karpenter.sh/discovery,Values=training-on-eks" \
    --query "SecurityGroups[*].GroupId" \
    --output text)
echo $NODE_SG_ID

# 인 바운드: 자기 자신(Self)을 목적지로 하는 모든 트래픽 허용
aws ec2 authorize-security-group-ingress --group-id ${NODE_SG_ID} --protocol all \
    --source-group ${NODE_SG_ID}

# 아웃 바운드: 자기 자신(Self)을 목적지로 하는 모든 트래픽 허용
aws ec2 authorize-security-group-egress --group-id ${NODE_SG_ID} --protocol all \
    --source-group ${NODE_SG_ID}
```
[결과]
```
An error occurred (InvalidPermission.Duplicate) when calling the AuthorizeSecurityGroupIngress operation: the specified rule "peer: sg-0b2f992c6b3d46431, ALL, ALLOW" already exists
```
클러스터 생성시 자동으로 만들어지게 되어 있기 때문에, 이와 같이 에러가 발생하는 것이 정상이다. 


## EFA 테스트 ## 
```
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: efa-test-pod
  labels:
    app: efa-test
spec:
  nodeSelector:
    karpenter.sh/nodepool: gpu                
  tolerations:                                             
    - key: "nvidia.com/gpu"
      operator: "Exists"                      # 노드의 테인트는 nvidia.com/gpu=present:NoSchedule 이나, Exists 연산자로 nvidia.com/gpu 키만 체크  
      effect: "NoSchedule"
#    - key: "vpc.amazon.com/efa"              # 카펜터 gpu 풀의 노드들은 nvidia.com/gpu 테인트만 가지고 있다.
#      operator: "Exists"                       
#      effect: "NoSchedule"        
  containers:
    - name: efa-container                               # public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0 
      image: public.ecr.aws/hpc-cloud/nccl-tests:latest           # EFA 드라이버와 NCCL 테스트 도구가 포함된 이미지 사용 (NVIDIA 공식 이미지 권장)
      command: ["/bin/bash", "-c", "sleep infinity"]
      resources:
        limits:
          vpc.amazonaws.com/efa: 1                      # EFA 장치를 파드에 직접 할당 (VPC CNI가 이 장치를 인식함)
          nvidia.com/gpu: 1                             # GPU 인스턴스인 경우
      securityContext:
        capabilities:                                   # EFA 통신을 위해 메모리 잠금 권한 필요
          add: ["IPC_LOCK"]
EOF
```
EFA는 하드웨어가 시스템 메모리에 직접 접근하여 데이터를 읽고 쓰는 RDMA(Remote Direct Memory Access) 기술을 사용한다. 통신에 사용되는 메모리 주소가 스왑 처리되어 디스크로 이동해버리면 하드웨어가 메모리를 찾지 못해 시스템 장애나 통신 에러가 발생한다. IPC_LOCK은 해당 메모리를 RAM에 "고정"시켜 이 문제를 방지한다. 학습 데이터가 메모리에서 스왑 영역으로 넘어가면 다시 읽어올 때 엄청난 속도 저하(Latency)가 발생함으로 실시간으로 수 기가바이트의 파라미터를 교환해야 하는 FSDP 학습에서 메모리 고정은 일관된 고성능을 유지하기 위한 필수 조건이다. 

```
kubectl exec -it efa-test-pod -- /bin/bash
fi_info -p efa
```
[결과]
```
provider: efa
    fabric: efa-direct
    domain: rdmap47s0-rdm
    version: 201.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap47s0-rdm
    version: 201.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap47s0-dgrm
    version: 201.0
    type: FI_EP_DGRAM
    protocol: FI_PROTO_EFA
```
* 성공 시: provider: efa, fabric: efa와 같은 정보가 상세하게 출력된다.
* 실패 시: fi_info 결과에 아무것도 나오지 않거나 에러가 발생한다. 이 경우 보안 그룹의 인/아웃 바운드 셀프 참조 존재여부를 확인한다. 

### efa 디바이스 조회 ###
```
ls -la /sys/class/infiniband/
```
[결과]
```
total 0
drwxr-xr-x.  2 root root 0 Jan  7 09:10 .
drwxr-xr-x. 40 root root 0 Jan  7 09:10 ..
lrwxrwxrwx.  1 root root 0 Jan  7 09:10 rdmap0s29 -> ../../devices/pci0000:00/0000:00:1d.0/infiniband/rdmap0s29
```

### efa 하드웨어 카운터 조회 ###
EC2 콘솔에서 EFA 인스턴스를 선택하고 시스템 매니저로 호스트에 로그인 한후, efa hw counter 를 조회한다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/efa-hw-counter.png)

  
## 레퍼런스 ##

* [NCCL 테스트](https://github.com/NVIDIA/nccl-tests)
* https://github.com/aws/deep-learning-containers/blob/master/available_images.md
* https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/node-efa.html

