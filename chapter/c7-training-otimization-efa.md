## EFA ##

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
NODE_SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=tag:karpenter.sh/discovery,Values=training-on-eks" \
    --query "SecurityGroups[*].GroupId" \
    --output text)
echo $NODE_SG_ID

# 아웃바운드: 자기 자신(Self)을 목적지로 하는 모든 트래픽 허용
# cf) 인바운드의 경우 클러스터를 생성하는 시점에 자동으로 설정되어져 있다. 
aws ec2 authorize-security-group-egress \
    --group-id $NODE_SG_ID --protocol all \
    --source-group ${NODE_SG_ID}
```
아래와 같은 오류가 발생하는 경우, EFA를 위한 아웃 바운드 규칙이 이미 eks 클러스터의 노드 시큐리티 그룹에 설정되어져 있다는 의미로, 다음단계로 진행하면 된다. 
```
An error occurred (InvalidPermission.Duplicate) when calling the AuthorizeSecurityGroupEgress operation: the specified rule "peer: sg-0856697271a3b5fad, ALL, ALLOW" already exists
```

#### 2. 카펜터 노드풀 생성 ####

분산 학습 성능을 극대화하려면 EFA 지원 노드들을 물리적으로 가까운 곳에 배치하는 'Cluster' 전략의 Placement Group 이 필요하다. 
EC2 생성시 ENI 설정에서 InterfaceType=efa를 설정해야 하나 카펜터의 경우 EFA 전용 옵션 필드는 제공하지 않는다.
별도의 체크박스 옵션은 없으며, 지원 인스턴스 타입 선택 + 배치 그룹 지정 + (필요시) EFA 전용 AMI 사용의 조합으로 EFA 사용 환경을 완성한다.

[efa-nodepool.yaml]
```
VPC_AZ=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].ZoneName" --output text)
echo "placement-group az: ${VPC_AZ}"
aws ec2 create-placement-group --group-name "training-on-eks" --strategy cluster

cat <<EOF > efa-nodepool.yaml
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu-efa
spec:
  role: "eksctl-KarpenterNodeRole-training-on-eks"
  amiSelectorTerms:
    # Required; when coupled with a pod that requests NVIDIA GPUs or AWS Neuron
    # devices, Karpenter will select the correct AL2023 accelerated AMI variant
    # see https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/
    - alias: al2023@latest
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks"
  # --- 배치 그룹 설정 부분 ---
  placementGroupName: "training-on-eks"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
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
          values: ["p", "g"]                        # p 와 g 타입
        - key: "karpenter.k8s.aws/instance-size"
          operator: In
          values: ["8xlarge", "12xlarge", "16xlarge", "24xlarge", "32xlarge", "48xlarge", "metal"]
        - key: "karpenter.k8s.aws/instance-generation"
          operator: Gt
          values: ["3"]                             # 4세대 이상(g4, g5, g6 등)만 사용 
        - key: "karpenter.sh/capacity-type"
          operator: In
          values: ["on-demand", "spot"]                     
        # 중요: 클러스터 배치 그룹은 단일 AZ 내에서만 작동하므로 하나만 지정
        - key: "topology.kubernetes.io/zone"
          operator: In
          values: [${VPC_AZ}]                       # ${VPC_AZ} 환경변수 값으로 대체    
      expireAfter: 720h # 30 * 24h = 720h
      taints:                                       # efa-workload 테인트 생성
        - key: "efa-workload"
          value: "true"
          effect: NoSchedule
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 30m
EOF

kubectl apply -f efa-nodepool.yaml
```

#### 3. 디바이스 플러그인 배포 #### 
```
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin --namespace kube-system
```

#### 4. 파드 생성 #### 
```
apiVersion: v1
kind: Pod
metadata:
  name: efa-test-pod
  labels:
    app: efa-test
spec:
  # 1. 앞에서 생성한 EFA 노드풀에 배치되도록 설정
  nodeSelector:
    karpenter.sh/nodepool: gpu-efa
  
  # 2. 노드풀에 설정한 테인트 허용
  tolerations:
    - key: "efa-workload"
      operator: "Equal"
      value: "true"
      effect: "NoSchedule"

  containers:
    - name: efa-container
      # EFA 드라이버와 NCCL 테스트 도구가 포함된 이미지 사용 (NVIDIA 공식 이미지 권장)
      image: nvcr.io/nvidia/pytorch:24.01-py3 
      command: ["/bin/bash", "-c", "sleep infinity"]
      resources:
        limits:
          # 3. EFA 장치를 파드에 직접 할당 (VPC CNI가 이 장치를 인식함)
          vpc.amazonaws.com: 1
          nvidia.com: 1 # GPU 인스턴스인 경우
      securityContext:
        # EFA 통신을 위해 메모리 잠금 권한 필요
        capabilities:
          add: ["IPC_LOCK"]
```

```
kubectl exec -it efa-test-pod -- /bin/bash
fi_info -p efa
```
* 성공 시: provider: efa, fabric: efa와 같은 정보가 상세하게 출력됩니다.
* 실패 시: fi_info 결과에 아무것도 나오지 않거나 에러가 발생합니다. (이 경우 보안 그룹의 아웃바운드 셀프 참조나 배치 그룹 설정을 다시 점검해야 합니다.)


### 3. NCCL 통신 테스트 (다중 노드 시) ###
만약 노드를 2대 이상 띄웠다면, AWS NCCL Test 도구를 사용하여 실제 노드 간 네트워크 대역폭(Bandwidth)을 측정할 수 있습니다.

