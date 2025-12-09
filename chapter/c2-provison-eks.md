<< 아키텍처 다이어그램 >> 
* x-86-ng : 2 instances
* node scaling with kapenter
  * for gpu
  * for graviton

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

## gpu 파드 스케줄링 ##
```
apiVersion: v1
kind: Pod
metadata:
  name: my-gpu-pod-specific-type
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:11.4.0-base-ubuntu20.04
      resources:
        limits:
          nvidia.com: 1
  affinity:
    nodeAffinity:
      # 필수 조건 (이 조건에 맞는 노드가 없으면 파드가 스케줄링되지 않음)
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: karpenter.k8s.aws/instance-type
            operator: In
            values:
            - g5.2xlarge
            - g5.4xlarge
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
