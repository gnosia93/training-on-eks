<< 아키텍처 다이어그램 >> 
* x-86-ng : 2 instances
* node scaling with kapenter
  * for gpu
  * for graviton

## 사전 준비 ##
* https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl
  
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
aws sts get-caller-identity

```  







## 목차 ##
* 사전준비 - 소프트웨어 설치
* 클러스터 생성
* 카펜터 설정
* kubeflow 트레이닝 설정 
* GPU POD 생성해 보기. 
