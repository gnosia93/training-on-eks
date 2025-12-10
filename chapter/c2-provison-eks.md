<< 아키텍처 다이어그램 >> 


## [kubectl 및 eksctl 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl) ##
code server 에 접속하여 kubectl 과 eksctl을 설치한다. 
 
1. kubectl 을 설치한다 
```
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.33.3/2025-08-03/bin/linux/arm64/kubectl
chmod +x ./kubectl
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc

kubectl version --client
```

2. eksctl 을 설치한다.
```
# for ARM systems, set ARCH to: `arm64`, `armv6` or `armv7`
ARCH=arm64
PLATFORM=$(uname -s)_$ARCH
curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
sudo install -m 0755 /tmp/eksctl /usr/local/bin && rm /tmp/eksctl
```

## 클러스터 생성 ##
eks 클러스터를 생성하기 위해서는 아래와 같이 최소한의 권한을 가지고 있어야 한다. 이번 워크샵에서는 EC2 인스턴스에 해당당 Role인 TOE_EKS_EC2_ROLE 이 AdminFullAccess 권한을 가지고 있다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/previllege_For_EKS.png)

로컬 PC 에서 테라폼으로 퍼블릭 및 프라이빗 서브넷 리스트를 조회한다. 
```
cd training-on-eks/tf
terraform output 
```
[결과]
```
instance_public_dns = "ec2-43-203-120-143.ap-northeast-2.compute.amazonaws.com"
private_subnet = [
  "subnet-009f634c97979d460",
  "subnet-05f66b53201e3c4cf",
]
public_subnet = [
  "subnet-01bd51c8c77af6d59",
  "subnet-0de148d8e62debe6d",
]
vscode_url = "http://ec2-43-203-120-143.ap-northeast-2.compute.amazonaws.com:8080"
```

eksctl 파라미터 값인 public 및 private-subnets 값을 조회된 값으로 수정한 후 클러스터를 생성한다.  
```
eksctl create cluster --name=training-on-eks \
  --enable-auto-mode \
  --version=1.33 \
  --region=ap-northeast-2 \
  --vpc-public-subnets="subnet-01bd51c8c77af6d59,subnet-0de148d8e62debe6d" \
  --vpc-private-subnets="subnet-009f634c97979d460,subnet-05f66b53201e3c4cf" 
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

1. 카펜터 노드 클래스 조회
```
kubectl get nodeclass
```
[결과]
```
NAME      ROLE                                                           READY   AGE
default   eksctl-training-on-eks-cluster-AutoModeNodeRole-DB4LZ4lI0H7N   True    11h
``` 

2. 카펜터 노드풀 조회
   
기본적으로 아래와 같이 두개의 노드풀이 생성되지만, gpu 파드는 스케줄링 할 수 없다. 노드풀의 세부 설정을 describe 해 보면
C, M, R 인스턴스 타입(CPU) 에만 파드가 스케줄링 하게 되었다. 
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

3. CRD 조회
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

4. [gpu 노드풀 생성](https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html)

노드 클래스는 EKS Auto 모드 클러스터 생성시 자동으로 만들어지는 default 클래스를 사용한다.   

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
          values: ["arm64", "amd64"]   # X86, ARM 시영  
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

[도커허브 nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) 로 방문해서 nvidia-smi 가 설치되어 있는 컨테이너 이미를 확인한다.  

[gpu-pod.yaml]
```
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda: 13.0.2-runtime-ubuntu22.04    # runtime 이미지 사용
      command: ["nvidia-smi"]                           # 컨테이너 시작 시 실행할 프로그램
      resources:
        limits:
          nvidia.com/gpu: 1
  tolerations:
    - key: "gpu-workload"
      operator: "Exists"
      effect: "NoSchedule"
 # affinity:
 #   nodeAffinity:
 #     # 필수 조건 (이 조건에 맞는 노드가 없으면 파드가 스케줄링되지 않음)
 #     requiredDuringSchedulingIgnoredDuringExecution:
 #       nodeSelectorTerms:
 #       - matchExpressions:
 #         - key: karpenter.k8s.aws/instance-type
 #           operator: In
 #           values:
 #           - g5.2xlarge
 #           - g5.4xlarge
      # 선호 조건 (가능하다면 이 인스턴스 유형을 사용하지만, 없어도 다른 인스턴스 사용 가능)
      # preferredDuringSchedulingIgnoredDuringExecution:
      # - weight: 1
      #   preference:
      #     matchExpressions:
      #     - key: karpenter.k8s.aws/instance-family
      #       operator: In
      #       values:
      #       - p4d

```





## 목차 ##
* 사전준비 - 소프트웨어 설치
* 클러스터 생성
* 카펜터 설정
* kubeflow 트레이닝 설정 
* GPU POD 생성해 보기. 


## 레퍼런스 ##

* [Getting Started with Karpenter](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/)

