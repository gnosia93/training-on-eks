## EFA ##

### 1. EFA를 지원하는 GPU 인스턴스 유형 ###
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
aws ec2 describe-instance-types \
    --instance-types p4d.24xlarge \
    --query "InstanceTypes[*].{InstanceType:InstanceType, \
        EfaSupported:NetworkInfo.EfaSupported, \
        MaxNetworkInterfaces:NetworkInfo.MaximumNetworkInterfaces, \
        NetworkPerformance:NetworkInfo.NetworkPerformance}" --output table
```
```
--------------------------------------------------------------------------------
|                             DescribeInstanceTypes                            |
+--------------+---------------+------------------------+----------------------+
| EfaSupported | InstanceType  | MaxNetworkInterfaces   | NetworkPerformance   |
+--------------+---------------+------------------------+----------------------+
|  True        |  p4d.24xlarge |  60                    |  4x 100 Gigabit      |
+--------------+---------------+------------------------+----------------------+
* 최대 60개의 ENI 사용가능
* 100Gbp의 EFA 4개
```


### 2. EFA 설정하기 ###

#### 1. EKS 노드 시큐리티 그룹 수정 #### 

EFA는 일반적인 TCP/UDP 스택을 우회하여 하드웨어 수준에서 통신하기 때문에 훨씬 엄격하고 명확한 규칙을 요구하는데, 아웃바운드 '셀프' 명시와 모든 프로토콜(All Traffic) 허용이 필수적 이다.
```
# EFA 노드들이 사용할 보안 그룹 ID
NODE_SG_ID="sg-xxxxxxxxxxxx"                <-------- aws cli 로 찾아야 한다..

aws ec2 describe-security-groups \
    --filters "Name=tag:karpenter.sh/discovery,Values=training-on-eks" \
    --query "SecurityGroups[*].GroupId" \
    --output text



# 아웃바운드: 자기 자신(Self)을 목적지로 하는 모든 트래픽 허용
aws ec2 authorize-security-group-egress \
    --group-id $NODE_SG_ID \
    --protocol all \                # 모든 프로토콜 - FA가 사용하는 커스텀 프로토콜(SRD 등)이 일반적인 포트 번호 개념과 다르게 동작
    --port -1 \                     # 모든 포트
    --source-group $NODE_SG_ID      # 자기 자신을 명시
```


#### 2. 카펜터 노드풀 생성 ####

분산 학습 성능을 극대화하려면 EFA 지원 노드들을 물리적으로 가까운 곳에 배치하는 'Cluster' 전략의 Placement Group 이 필요하다. 
EC2 생성시 ENI 설정에서 InterfaceType=efa를 설정해야 하나 카펜터의 경우 EFA 전용 옵션 필드는 제공하지 않는다.
별도의 체크박스 옵션은 없으며, 지원 인스턴스 타입 선택 + 배치 그룹 지정 + (필요시) EFA 전용 AMI 사용의 조합으로 EFA 사용 환경을 완성한다.

[efa-nodepool.yaml]
```
VPC_AZ=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].ZoneName" --output text)
aws ec2 create-placement-group --group-name "training-on-eks" --strategy cluster

cat <<EOF > efa-nodepool.yaml
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu-efa
spec:
  amiFamily: AL2
  role: "KarpenterNodeRole-MyCluster" # 실제 역할 이름으로 변경
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks"
  # --- 배치 그룹 설정 부분 ---
  placementGroupName: "training-on-eks"
  spec:
  userData: |
    #!/bin/bash
    # EFA 드라이버 확인 및 로드 (필요시)
    # EKS 최적화 AMI는 보통 드라이버가 포함되어 있지만, 
    # 인식 확인을 위해 fi_info -p efa 같은 명령을 사전에 체크할 수 있음
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-efa
spec:
  template:
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-efa
      requirements:
        - key: "karpenter.k8s.aws/instance-category"
          operator: In
          values: ["c", "p", "g"]                   # capacity를 고려하여 c 타입도 포함
        - key: "karpenter.k8s.aws/instance-size"
          operator: In
          values: ["8xlarge", "12xlarge", "16xlarge", "24xlarge", "32xlarge", "48xlarge", "metal"]
        - key: "karpenter.k8s.aws/instance-generation"
          operator: Gt
          values: ["3"]                             # 4세대 이상(g4, g5, g6 등)만 사용 
        - key: "karpenter.sh/capacity-type"
          operator: In
          values: ["on-demand"]                     # 클러스터 배치는 안정성을 위해 온디맨드 권장
        # 중요: 클러스터 배치 그룹은 단일 AZ 내에서만 작동하므로 하나만 지정
        - key: "topology.kubernetes.io/zone"
          operator: In
          values: [${VPC_AZ}]                       # ${VPC_AZ} 환경변수 값으로 대체    
      taints:                                       # efa-workload 테인트 생성
        - key: "efa-workload"
          value: "true"
          effect: NoSchedule
EOF

kubectl apply -f efa-nodepool.yaml
```

#### 3. 디바이스 플러그인 배포 #### 
```
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin --namespace kube-system
```

#### 4. 파드 스펙(Pod Spec) 구성 #### 
```
resources:
  limits:
    vpc.amazonaws.com/efa: 1  # 요청할 EFA 장치 수
  requests:
    vpc.amazonaws.com/efa: 1
```

컨테이너 이미지에는 EFA를 활용할 수 있는 MPI, NCCL과 같은 고성능 컴퓨팅(HPC) 라이브러리 및 도구가 설치되어 있어야 합니다. 컨테이너 내에서 FI_PROVIDER 환경 변수를 efa로 설정하는 것이 좋습니다. 

#### 5. 동작 확인 ####
* fi_info -p efa
* nccl-tests (예: all_reduce_perf)를 실행할 때 NCCL_DEBUG=INFO를 함께 설정하여 노드 간 트래픽이 EFA 인터페이스를 타는지 확인
* pytorch 에서 확인
   * 환경변수 설정 
   ```
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=INIT,NET
   ```
   * 출력로그
   ```
   NCCL INFO NET/OFI Selected Provider is efa
   NCCL INFO NET/OFI Running on P4d platform (P4d 등 특정 플랫폼 사용 시)
   NCCL INFO NET/OFI Forcing AWS OFI ndev ... 
   ```

----

## fi_getinfo: -61 (No data available) ##

#### 1. EC2 인스턴스의 EFA 활성화 여부 확인 ####
p4d.24xlarge 인스턴스라고 해서 자동으로 EFA가 켜지는 것은 아닙니다. 인스턴스 생성 시 네트워크 인터페이스(ENI) 설정에서 EFA가 'Enabled' 되어 있어야 합니다.
```
aws ec2 describe-instances --filters "Name=tag:Name,Values=노드이름" \
--query "Reservations[*].Instances[*].NetworkInterfaces[*].InterfaceType"
```
만약 ["efa"]가 출력되지 않고 ["interface"]만 나온다면, 인스턴스 레벨에서 EFA가 비활성화된 것입니다

#### 2. 보안 그룹(Security Group) 설정 (가장 흔한 원인) ####
EFA는 통신을 위해 보안 그룹 내의 모든 트래픽이 자기 자신(Self-referencing)에게 허용되어야 합니다.

해결 방법: 노드가 속한 보안 그룹의 Inbound와 Outbound 규칙에 다음을 추가하세요.
* Type: All Traffic
* Protocol: All
* Port Range: All
Source/Destination: 현재 보안 그룹 ID (예: sg-123456)를 그대로 입력

#### 3. Kubernetes Pod에 EFA 장치 할당 (Device Plugin) ####
EFA 장치가 노드에 있더라도, Pod가 해당 장치를 사용할 수 있도록 Kubernetes Device Plugin이 설정되어야 하며, Pod 스펙에도 리소스 요청이 포함되어야 합니다.
```
resources:
  limits:
    vpc.amazonaws.com: 1 # EFA 장치를 Pod에 할당
    nvidia.com: 8
```
Device Plugin 확인: 클러스터에 aws-efa-k8s-device-plugin이 실행 중인지 확인하세요.

* EFA 설정이 복잡하다면, 테스트를 위해 우선 NCCL_DEBUG=INFO와 함께 NCCL_P2P_DISABLE=1 또는 NCCL_IB_DISABLE=1을 설정하여 TCP 모드로 통신이 되는지 먼저 확인해 볼 수도 있습니다. (단, 성능은 저하됩니다.)
