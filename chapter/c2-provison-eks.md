## [kubectl 및 eksctl 설치](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/install-kubectl.html#linux_arm64_kubectl) ##

code-server-graviton 코드 서버에 웹으로 접속한 후, 터미널을 열어 kubectl, eksctl, helm 을 설치한다.
별다른 코멘트가 없다면 모든 작업은 code-server-graviton 웹환경의 터미널에서 수행한다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/code-server.png)
 
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

#### 4. k9s 설치 ####
```
ARCH="arm64"
if [ "$(uname -m)" != 'aarch64' ]; then
  ARCH="amd64"
fi
echo ${ARCH}" architecture detected .."
curl --silent --location "https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_${ARCH}.tar.gz" | tar xz -C /tmp
sudo mv /tmp/k9s /usr/local/bin/
k9s version
```

#### 5. eks-node-viewer 설치 ####
```
sudo dnf update -y
sudo dnf install golang -y

# 설치 확인 (v1.11 이상 필요)
go version
go install github.com/awslabs/eks-node-viewer/cmd/eks-node-viewer@latest

echo 'export PATH=$PATH:$(go env GOPATH)/bin' >> ~/.bashrc
source ~/.bashrc
```
go 컴파일 과정에서 다소 시간이 소요된다.

## EKS 클러스터 생성하기 ##

### 1. 환경 설정 ###
```
export AWS_REGION=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export CLUSTER_NAME="training-on-eks"
export K8S_VERSION="1.34"
export KARPENTER_VERSION="1.8.1"
export VPC_ID=$(aws ec2 describe-vpcs --filters Name=tag:Name,Values="${CLUSTER_NAME}" --query "Vpcs[].VpcId" --output text)
```

### 2. 서브넷 식별 ###
클러스터의 데이터 플레인(워커노드 들)은 아래의 프라이빗 서브넷에 위치하게 된다. 
```
aws ec2 describe-subnets \
    --filters "Name=tag:Name,Values=TOE-priv-subnet-*" "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].{ID:SubnetId, AZ:AvailabilityZone, Name:Tags[?Key=='Name']|[0].Value}" \
    --output table

SUBNET_IDS=$(aws ec2 describe-subnets \
    --region "${AWS_REGION}" \
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
클러스터 생성 완료까지 약 20 ~ 30분 정도의 시간이 소요된다.
```
cat > cluster.yaml <<EOF 
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: "${CLUSTER_NAME}"
  version: "${K8S_VERSION}"
  region: "${AWS_REGION}"

vpc:
  id: "${VPC_ID}"                    
  subnets:
    private:                                 # 프라이빗 서브넷에 데이터플레인 설치
$(cat SUBNET_IDS)

addons:
  - name: vpc-cni
    podIdentityAssociations:
      - serviceAccountName: aws-node
        namespace: kube-system
        permissionPolicyARNs: 
          - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
  - name: eks-pod-identity-agent
  - name: metrics-server
  - name: kube-proxy
  - name: coredns
  - name: aws-ebs-csi-driver                 # loki-ng 용  

managedNodeGroups:                           # 관리형 노드 그룹
  - name: ng-arm
    instanceType: c7g.2xlarge
    minSize: 2
    maxSize: 2
    desiredCapacity: 2
    amiFamily: AmazonLinux2023
    privateNetworking: true                  # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다.
    iam:
      withAddonPolicies:
        ebs: true                     		 # EBS CSI 드라이버가 작동하기 위한 IAM 권한 부여

  - name: ng-x86
    instanceType: c6i.2xlarge
    minSize: 2
    maxSize: 2
    desiredCapacity: 2
    amiFamily: AmazonLinux2023
    privateNetworking: true           		 # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다. 
    iam:
      withAddonPolicies:
        ebs: true                     		 # EBS CSI 드라이버가 작동하기 위한 IAM 권한 부여

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

EKS 에서 클러스터 시큐리티 그룹은 컨트롤 플레인과 워커노드 사이의 통신을 가능하게 한다. 컨트롤 플레인은 10250 포트를 통해 노드의 큐블렛과 통신하고 워커노드는 443 포트를 이용하여 컨트롤 플레인의 API 서버에 접근을 시도한다. 아래 명령어는 클러스터 시큐리티 그룹에 "karpenter.sh/discovery=${CLUSTER_NAME}" 태크가 존재하는지 확인하는 스크립트이다. 카펜터가 노드를 생성할때, 이와 동일한 태크를 가진 시큐리티 그룹을 찾아 신규 노드에 할당하게 된다. 시큐리티 그룹 검색에 실패하게 되는 경우, EC2 인스턴스는 생성되지만 EKS 클러스터에 조인하지 못한다.  
```
aws ec2 describe-security-groups \
  --group-ids $(aws eks describe-cluster --name ${CLUSTER_NAME} --query \
					"cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text) \
  --query "SecurityGroups[0].Tags" \
  --output table
```
[결과]
```
----------------------------------------------------------------------------------------
|                                DescribeSecurityGroups                                |
+----------------------------------------+---------------------------------------------+
|                   Key                  |                    Value                    |
+----------------------------------------+---------------------------------------------+
|  kubernetes.io/cluster/training-on-eks |  owned                                      |
|  Name                                  |  eks-cluster-sg-training-on-eks-1860330510  |
|  aws:eks:cluster-name                  |  training-on-eks                            |
|  karpenter.sh/discovery                |  training-on-eks                            |
+----------------------------------------+---------------------------------------------+
```
* 가끔 karpenter.sh/discovery 태그가 누락되는 경우가 발생하는데 이 경우 아래 명령어를 실행하여 추가해 준다.    
```
aws ec2 create-tags \
  --resources $(aws eks describe-cluster --name ${CLUSTER_NAME} --query \
					"cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text) \
  --tags Key=karpenter.sh/discovery,Value=training-on-eks
```


### 추가 정책 설정 ###
클러스터 생성이 완료되면 추가 설정이 필요하다. 카펜터 버전 1.8.1(EKS 1.3.4) 에는 아래와 같은 정책 설정이 누락되어 있어 패치가 필요하다. 
패치를 하지 않는 경우 카펜터가 프러비저닝한 노드가 클러스터에 조인되지 않는다. (노드 describe 시 Not Ready 상태)  

* eksctl-training-on-eks-iamservice-role 에 정책 추가(OIDC 정책 누락)
```
POLICY_JSON=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": "eks:DescribeCluster",
            "Resource": "arn:aws:eks:${AWS_REGION}:${AWS_ACCOUNT_ID}:cluster/${CLUSTER_NAME}"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreateInstanceProfile",
                "iam:DeleteInstanceProfile",
                "iam:GetInstanceProfile",
                "iam:TagInstanceProfile",
                "iam:AddRoleToInstanceProfile",
                "iam:RemoveRoleFromInstanceProfile",
                "iam:ListInstanceProfiles"
            ],
            "Resource": "*"
        }
    ]
}
EOF
)

aws iam put-role-policy \
    --role-name eksctl-training-on-eks-iamservice-role \
    --policy-name EKS_OIDC_Support_Policy \
    --policy-document "$POLICY_JSON"
```


## nginx 실행해 보기 ##
```
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 2 # 2개의 Nginx 파드를 실행합니다.
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      # 만약 기존 노드에 리소스가 부족하다면, 
      # 이 요청량 때문에 Karpenter가 새 노드를 띄울 수 있습니다.
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "1000m"
          limits:
            memory: "256Mi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: nginx
spec:
  selector:
    app: nginx
  type: LoadBalancer # AWS CLB(Classic Load Balancer)를 자동으로 생성합니다.
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
EOF
```

nginx 서비스를 조회한 후, 웹브라우저로 접속해 본다. 
```
kubectl get svc nginx
```
[결과]
```
NAME         TYPE           CLUSTER-IP       EXTERNAL-IP                                                               PORT(S)        AGE
nginx        LoadBalancer   172.20.151.112   a8bef1d582261479aa1eaffae26de2a0-2081456608.us-west-2.elb.amazonaws.com   80:30299/TCP   10s
```


## 클러스터 삭제 ##
#### 1. 카펜터 인스턴스 프로파일 삭제 #### 
```
ROLE_NAME="eksctl-KarpenterNodeRole-training-on-eks"
for p in $(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[*].PolicyArn' --output text); do aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$p"; done
for p in $(aws iam list-role-policies --role-name "$ROLE_NAME" --query 'PolicyNames[*]' --output text); do aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "$p"; done
for i in $(aws iam list-instance-profiles-for-role --role-name "$ROLE_NAME" --query 'InstanceProfiles[*].InstanceProfileName' --output text); do aws iam remove-role-from-instance-profile --instance-profile-name "$i" --role-name "$ROLE_NAME"; aws iam delete-instance-profile --instance-profile-name "$i"; done
aws iam delete-role --role-name "$ROLE_NAME"
```
#### 2. 클러스터 삭제 ####
```
eksctl delete cluster -f cluster.yaml
```

## 레퍼런스 ##

* [eksctl 사용 설명서](https://docs.aws.amazon.com/ko_kr/eks/latest/eksctl/what-is-eksctl.html)
* [Enable an IAM User or IAM Role to access an EKS cluster](https://www.javierinthecloud.com/enable-an-iam-user-or-iam-role-to-access-an-eks-cluster/)
* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)



