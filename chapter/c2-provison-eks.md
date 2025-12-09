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
eksctl create cluster --name=my-auto-cluster --enable-auto-mode --version=1.29 --region=ap-northeast-2
```

## 목차 ##
* 사전준비 - 소프트웨어 설치
* 클러스터 생성
* 카펜터 설정
* kubeflow 트레이닝 설정 
* GPU POD 생성해 보기. 
