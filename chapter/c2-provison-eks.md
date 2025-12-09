<< 아키텍처 다이어그램 >> 
* x-86-ng : 2 instances
* node scaling with kapenter
  * for gpu
  * for graviton

## [kubectl 및 eksctl 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl) ##
 
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
eks 클러스터를 생성하기 위해서는 아래와 같이 최소한의 권한을 가지고 있어야 한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/previllege_For_EKS.png)

```
aws ec2 describe-vpcs --filters "Name=tag:Name,Values=training-on-eks"

```


```
eksctl create cluster --name=training-on-eks \
  --enable-auto-mode --version=1.33 --region=ap-northeast-2 \
  --vpc-public-subnets="subnet-0abcdef1234567890,subnet-0fedcba9876543210" \
  --vpc-private-subnets="subnet-0bbccddeeff112233,subnet-0aaffeeccbb112233" \
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
