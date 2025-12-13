<< 아키텍처 다이어그램 >> 

* graviton 에만 kubectl, eksctl 를 설치한다. 


## [kubectl 및 eksctl 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl) ##
그라비톤을 사용하여 EKS 클러스터를 관리할 예정이므로, code-server-graviton 에만 kubectl 과 eksctl을 설치한다. 
 
1. kubectl 을 설치한다 
```
ARCH=arm64     # amd64 or arm64
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.33.3/2025-08-03/bin/linux/$ARCH/kubectl
chmod +x ./kubectl
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc

kubectl version --client
```

2. eksctl 을 설치한다.
```
ARCH=arm64     # amd64 or arm64
PLATFORM=$(uname -s)_$ARCH
curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
sudo install -m 0755 /tmp/eksctl /usr/local/bin && rm /tmp/eksctl
```


## 클러스터 생성 ##

### 클러스터 생성 권한 ###
eks 클러스터를 생성하기 위해서는 아래와 같이 최소한의 권한을 가지고 있어야 한다. 이번 워크샵에서는 EC2 인스턴스에 해당당 Role인 TOE_EKS_EC2_ROLE 이 AdminFullAccess 권한을 가지고 있다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/previllege_For_EKS.png)

### VPC 정보 조회 ###
* VPC ID
```
VPC_ID=$(aws ec2 describe-vpcs --filters Name=tag:Name,Values=training-on-eks --query "Vpcs[].VpcId" --output text)
echo ${VPC_ID}
```
[결과]
```
vpc-030b927274aa21417
```

* 서브넷 ID
```
aws ec2 describe-subnets \
    --filters "Name=tag:Name,Values=TOE-priv-subnet-*" "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].SubnetId" \
    --output text
```  
[결과]
```
```


### 클러스터 생성 ###
[cluster-config.yaml]
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: training-on-eks
  version: "1.33"
  region: ap-northeast-2

vpc:
  id: vpc-030b927274aa21417           # VPC ID를 여기에 지정해야 합니다. 
  subnets:
    private:
      subnet-009f634c97979d460: { }
      subnet-05f66b53201e3c4cf: { }

# 관리형 노드 그룹을 정의합니다.
managedNodeGroups:
  - name: ng-arm
    instanceType: c7g.2xlarge
    minSize: 2
    maxSize: 2
    desiredCapacity: 2
    amiFamily: AmazonLinux2_ARM_64
    privateNetworking: true     # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다.
   
  - name: ng-x86
    instanceType: c6i.2xlarge
    minSize: 2
    maxSize: 2
    desiredCapacity: 2
    privateNetworking: true     # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다. 
```

클러스터를 생성한다. 
```
eksctl create cluster -f cluster-config.yaml
```

### 서브넷 태깅 ### 
```
aws ec2 create-tags --resources subnet-01bd51c8c77af6d59 subnet-0de148d8e62debe6d \
  --tags Key=kubernetes.io/role/elb,Value=1 \
  --region ap-northeast-2

aws ec2 create-tags --resources subnet-01bd51c8c77af6d59 subnet-0de148d8e62debe6d \
  --tags Key=kubernetes.io/cluster/training-on-eks,Value=owned \
  --region ap-northeast-2

aws ec2 create-tags --resources subnet-009f634c97979d460 subnet-05f66b53201e3c4cf \
  --tags Key=kubernetes.io/role/internal-elb,Value=1 \
  --region ap-northeast-2
```

생성된 클러스터를 확인한다. 
```
kubectl config current-context
```
[결과]
```
i-0693a2a2c5ae6c4dd@training-on-eks.ap-northeast-2.eksctl.io
```



## gpu 노드풀 준비 ##
EKS 오토모드에서 아래와 같이 두개의 노드풀이 자동으로 생성되지만, gpu 파드를 스케줄링 할 수는 없다. 노드풀의 세부 설정을 describe 해 보면
C, M, R 인스턴스 타입(CPU) 만을 가지고 있어 CPU를 사용하는 파드만 스케줄링이 가능하다.
```
kubectl get nodepools -n karpenter
```
[결과]
```
NAME              NODECLASS   NODES   READY   AGE
general-purpose   default     0       True    8h
system            default     1       True    8h
```
[참고] 아래는 카펜터 노드풀의 상세 설정을 조회할 수 있는 명령어 셋이다.
```
kubectl describe nodepool system -n karpenter
kubectl describe nodepool general-purpose -n karpenter
```   

또한 카펜터에서 제공하는 CRD와는 다른 별도의 CRD 를 사용하고, 사용할 수 있는 레이블 역시 오픈 소스 카펜터와는 다른 것을 사용한다. 예를들어 
기존의 karpenter.k8s.aws/instance-category 레이블은 오토모드에서 eks.amazonaws.com/instance-category 으로 변경되었다 (자세한 내용은 https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html 참조) 
```
kubectl get crd -o wide
```
[결과]
```
NAME                                            CREATED AT
applicationnetworkpolicies.networking.k8s.aws   2025-12-09T17:19:41Z
clusternetworkpolicies.networking.k8s.aws       2025-12-09T17:19:41Z
clusterpolicyendpoints.networking.k8s.aws       2025-12-09T17:19:42Z
cninodes.eks.amazonaws.com                      2025-12-09T17:21:32Z
cninodes.vpcresources.k8s.aws                   2025-12-09T17:19:42Z
ingressclassparams.eks.amazonaws.com            2025-12-09T17:21:32Z
nodeclaims.karpenter.sh                         2025-12-09T17:21:32Z
nodeclasses.eks.amazonaws.com                   2025-12-09T17:21:32Z
nodediagnostics.eks.amazonaws.com               2025-12-09T17:21:32Z
nodepools.karpenter.sh                          2025-12-09T17:21:32Z
policyendpoints.networking.k8s.aws              2025-12-09T17:19:41Z
securitygrouppolicies.vpcresources.k8s.aws      2025-12-09T17:19:42Z
targetgroupbindings.eks.amazonaws.com           2025-12-09T17:21:32Z
```
crd 를 조회해 보면 api 가 변경된 것을 확인할 수 있다. 노드풀의 경우 karpenter.sh 에 있으나, 노드 클래스의 경우 eks.amazonaws.com 으로 변경되었다.

### [gpu 노드풀 생성](https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html) ###

여기서는 gpu-pool 을 신규로 생성 예정인데, 노드 클래스의 경우 EKS 오토모드에서 기본으로 제공하는 default 클래스를 사용할 것이다. 노드 클래스 역시 기존의 EC2NodeClass 와 비교해서 상당 부분이 바뀌였다. (세부적인 내용은 https://docs.aws.amazon.com/eks/latest/userguide/create-node-class.html 에서 참조)

[gpu-nodepool.yaml] 
```
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-pool
spec:
  template:
    spec:
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: default
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]            # X86 만 설정  
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]        # 온디맨드 인스턴스 사용        
        - key: eks.amazonaws.com/instance-category 
          operator: In
          values: ["g", "p"]           # GPU 인스턴스 사용  
        
        # 특정 세대(예: 4세대 이상) 또는 특정 타입만 허용할 수 있습니다.
        # - key: karpenter.k8s.aws/instance-type
        #   operator: In
        #   values: ["g5.xlarge", "p3.2xlarge"]
        
      # GPU 노드임을 명시하는 Taint 추가 (GPU Pod만 스케줄링되도록 유도)
      taints:
        - key: "gpu-workload"
          effect: "NoSchedule"
```

GPU 파드를 실행할 수 있는 노드풀을 생성하고 READY 상태를 확인한다.   
```
kubectl apply -f gpu-nodepool.yaml
kubectl get nodepool
```
[결과]
```
NAME              NODECLASS   NODES   READY   AGE
general-purpose   default     0       True    12h
gpu-pool          default     0       True    8s
system            default     1       True    12h
```

## GPU 파드 스케줄링 ##

[도커허브 nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) 로 가서 nvidia-smi 가 설치되어 있는 컨테이너 이미지를 확인한다. 해당 페이지에서 아래로 스크롤하면 최신 컨테이너 이미지 정보를 확인할 수 있다.    
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/cuda-container.png)
이번 워크샵에서는 13.0.2-runtime-ubuntu22.04 도커 이미지로 nvidia-smi 를 실행할 예정이다.   

[gpu-pod.yaml]
```
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:13.0.2-runtime-ubuntu22.04    # runtime 이미지 사용
      command: ["nvidia-smi"]                          # 컨테이너 시작 시 실행할 프로그램
      resources:
        limits:
          nvidia.com/gpu: 1
  tolerations:                                             
    - key: "gpu-workload"                              # GPU 노드풀에 파드를 스케줄링하기 위해서 toleration 을 설정한다.        
      operator: "Exists"
      effect: "NoSchedule"
```

파드를 생성하고 nvidia-smi 가 동작하는 확인한다.  
```
kubectl apply -f gpu-pod.yaml
kubectl describe pod gpu-pod
kubectl logs gpu-pod
```
[출력]
```
Node-Selectors:              <none>
Tolerations:                 gpu-workload:NoSchedule op=Exists
                             node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:NoSchedule op=Exists
Events:
  Type     Reason            Age                  From                   Message
  ----     ------            ----                 ----                   -------
  Warning  FailedScheduling  5m12s                default-scheduler      0/2 nodes are available: 1 node(s) had untolerated taint {CriticalAddonsOnly: }, 1 node(s) had untolerated taint {karpenter.sh/disrupted: }. preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
  Normal   Scheduled         4m25s                default-scheduler      Successfully assigned default/gpu-pod to i-0b2714b5d7951c695
  Normal   Nominated         5m12s                eks-auto-mode/compute  Pod should schedule on: nodeclaim/gpu-pool-6csft
  Normal   Pulling           4m20s                kubelet                Pulling image "nvidia/cuda:13.0.2-runtime-ubuntu22.04"
  Normal   Pulled            3m21s                kubelet                Successfully pulled image "nvidia/cuda:13.0.2-runtime-ubuntu22.04" in 58.965s (58.965s including waiting). Image size: 1766399024 bytes.
  Normal   Created           12s (x6 over 3m21s)  kubelet                Created container: cuda-container
  Normal   Started           12s (x6 over 3m21s)  kubelet                Started container cuda-container
  Normal   Pulled            12s (x5 over 3m10s)  kubelet                Container image "nvidia/cuda:13.0.2-runtime-ubuntu22.04" already present on machine
  Warning  BackOff           0s (x15 over 3m9s)   kubelet                Back-off restarting failed container cuda-container in pod gpu-pod_default(4b7846fb-2e8f-4ba7-9d6d-ee5711363436)


Wed Dec 10 06:44:46 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA T4G                     On  |   00000000:00:1F.0 Off |                    0 |
| N/A   49C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
(참고) describe 의 출력 결과중 마지막 라인의 Warning BackOff 의 경우 컨테이너는 종료하였으나 파드가 살아있기 때문에 발생하는 메시지이다. 즉 nvidia-smi 는 실행을 종료하였으나 파드는 살아있다.

## EFA 테스트 하기 ##

현재 EKS 오토모드는 EFA(Elastic Fabric Adapter)를 지원하지 않는다 ?. 클러스터 생성시 EKS 관리형 노드 그룹이나 자체 관리형 노드를 선택해야 한다. 
이번 워크샵에서는 ENA 를 이용하여 분산 훈련을 테스트해 볼 예정이다.

### 필수 전제 조건 요약 (Pod 배포 전 완료되어야 함): ###
* EKS 모드: EKS 관리형 노드 그룹 또는 자체 관리형 노드를 사용해야 합니다 (Fargate 불가).
#### 2. 인스턴스 유형: EFA를 지원하는 GPU 인스턴스 유형 ####
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
* 디바이스 플러그인 배포: 클러스터에 aws-efa-k8s-device-plugin이 DaemonSet으로 배포되어 실행 중이어야 합니다. 이 플러그인이 aws.amazon.com 리소스를 노출시킵니다.
#### 4. [DLC 이미지](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/appendix-dlc-release-notes-pytorch.html) 에서 헤딩 이미지를 찾는다. ####
  

```
apiVersion: v1
kind: Pod
metadata:
  name: efa-gpu-dl-pod
  labels:
    app: dl-training
spec:
  # EKS Auto Mode 사용을 위한 필수 셀렉터
  nodeSelector:
    eks.amazonaws.com/compute-type: auto
    # EFA 지원 인스턴스 선택을 위한 레이블 (AWS가 자동으로 붙여줌)
    # P4, P5 인스턴스 등에서 EFA가 활성화됩니다.
    # 클러스터 구성에 따라 레이블 키가 달라질 수 있습니다.
    # 예시 레이블:
    # networking.eks.amazonaws.com: "true" 

  # GPU 노드에 스케줄링될 수 있도록 톨러레이션 추가
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "gpu-workload"
    operator: "Exists"
    effect: "NoSchedule"

  containers:
  - name: dl-container-efa
    # 위에서 추천한 AWS DLC 이미지 사용 (리전과 태그를 실제 값으로 변경하세요)
    image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
    command: ["/bin/bash", "-c"]
    args:
        - |
          echo "Checking for EFA fabric interface using fi_info..."
          # EFA 활성화 확인 명령어
          fi_info -p efa
          
          # 추가적인 연결 테스트는 여기 아래에 명령어를 추가할 수 있습니다.
          # 예시: /opt/amazon/efa/bin/efa_test.sh
          
          if [ $? -eq 0 ]; then
            echo "EFA interface found successfully."
          else
            echo "Failed to find EFA interface."
          fi
    resources:
      limits:
        # 8개의 NVIDIA GPU 할당 요청
        nvidia.com/gpu: 1 
        # EFA 리소스 할당 요청 (이 리소스 타입은 EFA Device Plugin이 설치되어야 사용 가능)
        # Auto Mode에서는 AWS가 EFA 플러그인 설치를 관리합니다.
        # aws.amazon.com: "1" # 필요한 경우 주석 해제하여 사용
      requests:
        nvidia.com/gpu: 1

    # EFA 사용을 위한 환경 변수 설정 (컨테이너 내 라이브러리 설정)
    env:
    - name: NCCL_DEBUG
      value: "INFO"
    - name: NCCL_ALGO
      value: "RING"
    # AWS Libfabric을 NCCL 네트워크 제공자로 지정
    - name: NCCL_NET
      value: "AWS Libfabric"
    - name: FI_EFA_USE_DEVICE_RDMA
      value: "1"
    - name: FI_PROVIDER
      value: "efa"
    - name: FI_LOG_LEVEL
      value: "INFO"    
```



## 레퍼런스 ##

* [Getting Started with Karpenter](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/)
* [EKS Auto Mode에 대해서](https://devops-james.tistory.com/m/514#:~:text=AWS%EC%97%90%EC%84%9C%20%EA%B4%80%EB%A6%AC%20%2D%20SSH%EC%99%80%20SSM%20%EC%97%91%EC%84%B8%EC%8A%A4%EA%B9%8C%EC%A7%80%20%EA%B8%88%EC%A7%80%ED%95%98%EA%B3%A0,pod%EB%93%A4%EC%9D%B4%20%EC%97%86%EC%8A%B5%EB%8B%88%EB%8B%A4.%20%2D%20GPU%EB%8F%84%20%EB%82%B4%EC%9E%A5%20%ED%94%8C%EB%9F%AC%EA%B7%B8%EC%9D%B8%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%A9%EB%8B%88%EB%8B%A4.)
