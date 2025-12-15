## [kubectl 및 eksctl 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl) ##
그라비톤을 사용하여 EKS 클러스터를 관리할 예정이므로, code-server-graviton 에만 kubectl 과 eksctl을 설치한다. 
 
#### 1. kubectl 설치 #### 
```
ARCH=arm64     # amd64 or arm64
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.33.3/2025-08-03/bin/linux/$ARCH/kubectl
chmod +x ./kubectl
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc

kubectl version --client
```

#### 2. eksctl 설치 ####
```
ARCH=arm64     # amd64 or arm64
PLATFORM=$(uname -s)_$ARCH
curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
sudo install -m 0755 /tmp/eksctl /usr/local/bin && rm /tmp/eksctl

eksctl version
```

#### 3. helm 설치 ####
```
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4
sh get_helm.sh

helm version
``` 

## EKS 클러스터 생성하기 ##

### 1. 환경 설정 ###
```
export AWS_DEFAULT_REGION="ap-northeast-2"
export CLUSTER_NAME="training-on-eks"
export K8S_VERSION="1.33"
export KARPENTER_VERSION="1.8.3"
export VPC_ID=$(aws ec2 describe-vpcs --filters Name=tag:Name,Values="${CLUSTER_NAME}" --query "Vpcs[].VpcId" --output text)
```

### 2. 서브넷 식별 ###
클러스터의 데이터 플레인은 다음의 프라이빗 서브넷에 위치하게 된다. 
```
aws ec2 describe-subnets \
    --filters "Name=tag:Name,Values=TOE-priv-subnet-*" "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].{ID:SubnetId, AZ:AvailabilityZone, Name:Tags[?Key=='Name']|[0].Value}" \
    --output table

SUBNET_IDS=$(aws ec2 describe-subnets \
    --region "${AWS_DEFAULT_REGION}" \
    --filters "Name=tag:Name,Values=TOE-priv-subnet-*" "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].AvailabilityZone" \
    --output text)

if [ -z "$SUBNET_IDS" ]; then
    echo "에러: VPC ${VPC_ID} 에 서브넷이 존재하지 않습니다.."
fi

# YAML 형식에 맞게 동적 문자열 생성 (각 ID 뒤에 ": {}" 추가 및 앞쪽 Identation과 줄바꿈)
SUBNET_YAML=""
if [ -f SUBNET_IDS ]; then
    rm SUBNET_IDS
fi
for id in $SUBNET_IDS; do
#   SUBNET_YAML+="      ${id}: {}" # 이 위치에서 엔터 키를 쳐서 실제 줄바꿈을 만듭니다.
   echo "      ${id}: {}" >> SUBNET_IDS
done
```

### 3. 클러스터 생성 ### 
```
cat <<EOF | eksctl create cluster -f -
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: "${CLUSTER_NAME}"
  version: "${K8S_VERSION}"
  region: "${AWS_DEFAULT_REGION}"

vpc:
  id: "${VPC_ID}"                    
  subnets:
    private:                                 # 프라이빗 서브넷에 데이터플레인 설치
$(cat SUBNET_IDS)

managedNodeGroups:                           # 관리형 노드 그룹
  - name: ng-arm
    instanceType: c7g.2xlarge
    minSize: 1
    maxSize: 1
    desiredCapacity: 1
    amiFamily: AmazonLinux2023
    privateNetworking: true                  # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다.
   
  - name: ng-x86
    instanceType: c6i.2xlarge
    minSize: 1
    maxSize: 1
    desiredCapacity: 1
    amiFamily: AmazonLinux2023
    privateNetworking: true           # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다. 

iam:
  withOIDC: true 

karpenter:
  version: "${KARPENTER_VERSION}"
EOF
```

[결과]
```
2025-12-13 13:33:19 [ℹ]  eksctl version 0.220.0
2025-12-13 13:33:19 [ℹ]  using region ap-northeast-2
2025-12-13 13:33:20 [✔]  using existing VPC (vpc-030b927274aa21417) and subnets (private:map[subnet-010db3e6a658817d6:{subnet-010db3e6a658817d6 ap-northeast-2c 10.0.6.0/24 0 } subnet-099acb450b8051d06:{subnet-099acb450b8051d06 ap-northeast-2a 10.0.4.0/24 0 } subnet-0e521bd6de96308b8:{subnet-0e521bd6de96308b8 ap-northeast-2b 10.0.5.0/24 0 }] public:map[])
2025-12-13 13:33:20 [!]  custom VPC/subnets will be used; if resulting cluster doesn't function as expected, make sure to review the configuration of VPC/subnets
2025-12-13 13:33:20 [ℹ]  nodegroup "ng-arm" will use "" [AmazonLinux2023/1.33]
2025-12-13 13:33:20 [ℹ]  nodegroup "ng-x86" will use "" [AmazonLinux2023/1.33]
2025-12-13 13:33:20 [!]  Auto Mode will be enabled by default in an upcoming release of eksctl. This means managed node groups and managed networking add-ons will no longer be created by default. To maintain current behavior, explicitly set 'autoModeConfig.enabled: false' in your cluster configuration. Learn more: https://eksctl.io/usage/auto-mode/
3:36:05 [ℹ]  using Kubernetes version 1.33
...
```

### 4. 클러스터 확인 ### 
#### 4.1 생성된 클라스터 ####
```
kubectl config current-context
```
[결과]
```
i-048265208fb345ec5@training-on-eks.ap-northeast-2.eksctl.io
```
#### 4.2 노드그룹 ####
```
eksctl get nodegroup --cluster=training-on-eks
```
```
CLUSTER         NODEGROUP       STATUS  CREATED                 MIN SIZE        MAX SIZE        DESIRED CAPACITY        INSTANCE TYPE   IMAGE ID                ASG NAME                                  TYPE
training-on-eks ng-arm          ACTIVE  2025-12-13T13:47:35Z    2               2               2                       c7g.2xlarge     AL2023_ARM_64_STANDARD  eks-ng-arm-a2cd8bfb-ba01-1252-3342-5cabc45b0b0b    managed
training-on-eks ng-x86          ACTIVE  2025-12-13T13:47:34Z    2               2               2                       c6i.2xlarge     AL2023_x86_64_STANDARD  eks-ng-x86-e8cd8bfb-ba1b-0f17-c83a-0db24ba49f87    managed
```

## 레퍼런스 ##

* [eksctl 사용 설명서](https://docs.aws.amazon.com/ko_kr/eks/latest/eksctl/what-is-eksctl.html)
* [Install on EKS](https://www.kubeai.org/installation/eks/)
