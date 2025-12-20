<<>>
<< 제대로 동작하지 않는다.>>
<< 노드는 생성되고, 클러스터에 조인하는 것처럼 보이나, 파드는 pending 되어 있다>>

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

#### 2-1. EKS 노드 시큐리티 그룹 수정 #### 

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

#### 2-2. 카펜터 노드풀 생성 ####

분산 학습 성능을 극대화하려면 EFA 지원 노드들을 물리적으로 가까운 곳에 배치하는 'Cluster' 전략의 Placement Group 이 필요하다. 
EC2 생성시 ENI 설정에서 InterfaceType=efa를 설정해야 하나 카펜터의 경우 EFA 전용 옵션 필드는 제공하지 않는다.
별도의 체크박스 옵션은 없으며, 지원 인스턴스 타입 선택 + 배치 그룹 지정 + (필요시) EFA 전용 AMI 사용의 조합으로 EFA 사용 환경을 완성한다.

먼저 placement 그룹과 인스턴스 프로파일을 아래와 같이 생성한다. 
```
export VPC_AZ=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].ZoneName" --output text)
echo "placement-group az: ${VPC_AZ}"
aws ec2 create-placement-group --group-name "training-on-eks" --strategy cluster

aws iam create-instance-profile --instance-profile-name EFAInstanceProfile
aws iam add-role-to-instance-profile \
    --instance-profile-name EFAInstanceProfile \
    --role-name eksctl-KarpenterNodeRole-training-on-eks
```

efa 노드풀을 생성한다.
```

cat <<EOF > efa-nodepool.yaml
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu-efa
spec:
  # role: "eksctl-KarpenterNodeRole-training-on-eks"       # 인스턴스 프로파일을 설정하는 경우 주식처리한다. 
  # --- 배치 그룹 설정 부분 ---
  instanceProfile: "EFAInstanceProfile"
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
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
  userData: |
    #!/bin/bash
    # 런타임 ulimit 설정은 세션이 종료되면 사라지므로 설정 파일에 직접 기록.
    cat <<EOF > /etc/security/limits.d/99-efa.conf
    * soft memlock unlimited
    * hard memlock unlimited
    * soft stack unlimited
    * hard stack unlimited
    EOF
    # 즉시 적용을 위해 현재 세션에도 적용
    ulimit -l unlimited
    ulimit -s unlimited
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
          values: ["c", "p", "g"]                        # c, p 와 g 타입
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
          values: ["${VPC_AZ}"]                       # ${VPC_AZ} 환경변수 값으로 대체    
      taints:                                       # efa-workload 테인트 생성
        - key: "nvidia.com/gpu"            # nvidia-device-plugin 데몬은 nvidia.com/gpu=present:NoSchedule 테인트를 Tolerate 한다. 
          value: "present"                 # value 값으로 present 와 다른값을 설정하면 nvidia-device-plugin 이 동작하지 않는다 (GPU를 찾을 수 없다)   
          effect: NoSchedule               # nvidia-device-plugin 이 GPU 를 찾으면 Nvidia GPU 관련 각종 테인트와 레이블 등을 노드에 할당한다.  
        - key: "vpc.amazonaws.com"
          value: "true"
          effect: NoSchedule
      expireAfter: 720h    
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 30m
EOF

kubectl apply -f efa-nodepool.yaml
```
* 노드 클래스를 확인한다. 
```
kubectl get ec2nodeclass
```
[결과] gpu-efa 노드 클래스의 READY 필드값이 True 이어야 한다. 
```
NAME      READY   AGE
cpu       True    5d1h
gpu       True    4d22h
gpu-efa   True    21m
```
* 노드풀을 확인한다. 
```
kubectl get nodepool
```
[결과] gpu-efa 노드풀의 READY 필드값이 True 이어야 한다. 
```
NAME      NODECLASS   NODES   READY   AGE
gpu       gpu         0       True    4d22h
gpu-efa   gpu-efa     0       True    22m
```

#### 2-3. 디바이스 플러그인 배포 #### 
```
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin --namespace kube-system

kubectl get ds -n kube-system
```
[결과]
``` 
NAME                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR              AGE
aws-efa-k8s-device-plugin   0         0         0       0            0           <none>                     31m
aws-node                    2         2         2       2            2           <none>                     4d22h
ebs-csi-node                2         2         2       2            2           kubernetes.io/os=linux     4d10h
ebs-csi-node-windows        0         0         0       0            0           kubernetes.io/os=windows   4d10h
kube-proxy                  2         2         2       2            2           <none>                     4d22h
```

#### 2-4. VPC CNI 설정 ####       <-------------- 이 설정은 보류하도록 한다... 
```
kubectl set env daemonset/aws-node -n kube-system ENABLE_EFA_SUPPORT=true
kubectl get daemonset aws-node -n kube-system -o yaml | grep ENABLE_EFA_SUPPORT
```
EKS 클러스터의 aws-node (VPC CNI)가 EFA를 지원하도록 설정되어야 한다.

### 3. EFA 테스트 ### 
nodeSelector 를 이용하여 Karpenter가 관리하는 gpu-efa 노드풀을 사용하여 파드가 스케줄링되도록 한다 (특정 노드풀을 쓰도록 강제하는 방식)
```
cat <<EOF > efa-test-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: efa-test-pod
  labels:
    app: efa-test
spec:
  nodeSelector:
    karpenter.sh/nodepool: gpu-efa                      # 앞에서 생성한 EFA 노드풀에 배치되도록 설정
  tolerations:                                             
    - key: "nvidia.com/gpu"
      operator: "Exists"                      # 노드의 테인트는 nvidia.com/gpu=present:NoSchedule 이나, Exists 연산자로 nvidia.com/gpu 키만 체크  
      effect: "NoSchedule"
    - key: "vpc.amazon.com/efa"
      operator: "Exists"                      # 노드의 테인트는 nvidia.com/gpu=present:NoSchedule 이나, Exists 연산자로 nvidia.com/gpu 키만 체크  
      effect: "NoSchedule"        
  containers:
    - name: efa-container
      image: nvcr.io/nvidia/pytorch:24.01-py3           # EFA 드라이버와 NCCL 테스트 도구가 포함된 이미지 사용 (NVIDIA 공식 이미지 권장)
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
EFA는 하드웨어가 시스템 메모리에 직접 접근하여 데이터를 읽고 쓰는 RDMA(Remote Direct Memory Access) 기술을 사용한다. 통신에 사용되는 메모리 주소가 스왑 처리되어 디스크로 이동해버리면 하드웨어가 메모리를 찾지 못해 시스템 장애나 통신 에러가 발생합니다. IPC_LOCK은 해당 메모리를 RAM에 "고정"시켜 이 문제를 방지한다. 학습 데이터가 메모리에서 스왑 영역으로 넘어가면 다시 읽어올 때 엄청난 속도 저하(Latency)가 발생함으로 실시간으로 수 기가바이트의 파라미터를 교환해야 하는 FSDP 학습에서 메모리 고정은 일관된 고성능을 유지하기 위한 필수 조건이다. 
* 메모리 제한: IPC_LOCK 권한을 주더라도 시스템의 ulimit (memlock) 제한이 낮으면 학습 중 오류가 발생할 수 있다. Karpenter의 EC2NodeClass에서 사용자 데이터(UserData)를 통해 sudo ulimit -s unlimited 및 ulimit -l unlimited 설정을 추가하는 것이 안전하다. 
    * sudo ulimit -s unlimited 명령어는 프로세스가 사용하는 스택 크기(Stack Size)의 제한을 해제
    * ulimit -l unlimited는 프로세스가 물리적 메모리에 고정(Lock)할 수 있는 메모리의 크기 제한을 해제

```
kubectl apply -f efa-test-pod.yaml
kubectl exec -it efa-test-pod -- /bin/bash
fi_info -p efa
```
* 성공 시: provider: efa, fabric: efa와 같은 정보가 상세하게 출력됩니다.
* 실패 시: fi_info 결과에 아무것도 나오지 않거나 에러가 발생합니다. (이 경우 보안 그룹의 아웃바운드 셀프 참조나 배치 그룹 설정을 다시 점검해야 합니다.)


### 4. NCCL 통신 테스트 (다중 노드 시) ###
만약 노드를 2대 이상 띄웠다면, AWS NCCL Test 도구를 사용하여 실제 노드 간 네트워크 대역폭(Bandwidth)을 측정할 수 있습니다.

## 명령어.. ##
* ec2nodeclass yaml 스팩 확인
``` 
kubectl explain ec2nodeclass.spec
```

## 레퍼런스 ##
* https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/node-efa.html
