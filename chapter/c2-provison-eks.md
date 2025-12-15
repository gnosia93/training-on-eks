## [kubectl 및 eksctl 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl) ##

웹브라우저를 이용하여 code-server-graviton 코드 서버로 접속한 후 터미널을 열고 kubectl, eksctl, helm 을 설치한다 (그라비톤에만 설치)
별다른 코멘트가 없는 경우, 모든 작업은 code-server-graviton 에서 수행한다. 
 
#### 1. kubectl 설치 #### 
```
ARCH=arm64     
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.33.3/2025-08-03/bin/linux/$ARCH/kubectl
chmod +x ./kubectl
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc

kubectl version --client
```

#### 2. eksctl 설치 ####
```
ARCH=arm64    
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
export K8S_VERSION="1.34"
export KARPENTER_VERSION="1.8.1"
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
클러스터 생성 완료까지 약 20분 정도의 시간이 소요된다.
```
cat > cluster.yaml <<EOF 
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
```
eksctl create cluster -f cluster.yaml
```

[결과]
```
2025-12-15 04:51:25 [ℹ]  eksctl version 0.220.0
2025-12-15 04:51:25 [ℹ]  using region ap-northeast-2
2025-12-15 04:51:25 [✔]  using existing VPC (vpc-0c1d7c478d72dc9df) and subnets (private:map[ap-northeast-2a:{subnet-015e95a219d74be01 ap-northeast-2a 10.0.4.0/24 0 } ap-northeast-2b:{subnet-02ec412eb8eea970b ap-northeast-2b 10.0.1.0/24 0 } ap-northeast-2c:{subnet-0c3ccee255056c5b7 ap-northeast-2c 10.0.6.0/24 0 } ap-northeast-2d:{subnet-0c57f89f419008c4f ap-northeast-2d 10.0.3.0/24 0 }] public:map[])
2025-12-15 04:51:25 [!]  custom VPC/subnets will be used; if resulting cluster doesn't function as expected, make sure to review the configuration of VPC/subnets
2025-12-15 04:51:25 [ℹ]  nodegroup "ng-arm" will use "" [AmazonLinux2023/1.33]
2025-12-15 04:51:25 [ℹ]  nodegroup "ng-x86" will use "" [AmazonLinux2023/1.33]
2025-12-15 04:51:25 [!]  Auto Mode will be enabled by default in an upcoming release of eksctl. This means managed node groups and managed networking add-ons will no longer be created by default. To maintain current behavior, explicitly set 'autoModeConfig.enabled: false' in your cluster configuration. Learn more: https://eksctl.io/usage/auto-mode/
2025-12-15 04:51:25 [ℹ]  using Kubernetes version 1.33
...
...
2025-12-15 05:09:32 [ℹ]  waiting for CloudFormation stack "eksctl-training-on-eks-addon-iamserviceaccount-karpenter-karpenter"
2025-12-15 05:09:32 [ℹ]  created namespace "karpenter"
2025-12-15 05:09:32 [ℹ]  created serviceaccount "karpenter/karpenter"
2025-12-15 05:09:32 [ℹ]  adding identity "arn:aws:iam::499514681453:role/eksctl-KarpenterNodeRole-training-on-eks" to auth ConfigMap
2025-12-15 05:09:32 [ℹ]  adding Karpenter to cluster training-on-eks
2025-12-15 05:09:52 [ℹ]  kubectl command should work with "/home/ec2-user/.kube/config", try 'kubectl get nodes'
2025-12-15 05:09:52 [✔]  EKS cluster "training-on-eks" in "ap-northeast-2" region is ready
```

### 4. 클러스터 확인 ### 
#### 4.1 컨텍스트 ####
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

### 5. 카펜터 설정 ###
```
cat <<EOF > nodepool-default.yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: topology.kubernetes.io/zone
          operator: Exists
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64", "arm64"]
      nodeClassRef:
        name: default
  limits:
    cpu: 1000
    memory: 1000Gi
  disruption:
    consolidationPolicy: WhenUnderutilized
    expireAfter: 720h
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
spec:
  role: "eksctl-KarpenterNodeRole-${CLUSTER_NAME}"
  amiSelectorTerms:
    # Required; when coupled with a pod that requests NVIDIA GPUs or AWS Neuron
    # devices, Karpenter will select the correct AL2023 accelerated AMI variant
    # see https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/
    - alias: al2023@latest
  subnetSelector:
    - tags:
      karpenter.sh/discovery: ${CLUSTER_NAME}
  securityGroupSelector:
    - tags:
      karpenter.sh/discovery: ${CLUSTER_NAME}
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
EOF
```

```
cat <<EOF > nginx.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 30
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.23.1-alpine
        ports:
        - containerPort: 80
        resources:
          # Karpenter가 새 노드를 준비하도록 리소스 요청 설정
          requests:
            cpu: 100m
            memory: 100Mi
EOF
```
```
kubectl apply -f nginx.yaml
kubectl scale deployment/nginx --replicas=1000
```


## 레퍼런스 ##

* [eksctl 사용 설명서](https://docs.aws.amazon.com/ko_kr/eks/latest/eksctl/what-is-eksctl.html)
* [eksctl EKS 설치 예제](https://www.kubeai.org/installation/eks/)
