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

## EKS 클러스터 생성하기 ##

### 1. 환경 설정 ###
```
export REGION=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)
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
    --region "${REGION}" \
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
  region: "${REGION}"

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

클러스터 생성이 완료되면 1/시큐리티 그룹 태깅과 2/억세스 설정 이 필요하다. 

#### 1. 시큐리티 그룹 태깅 ####
클러스터 생성시 만들어진 시큐리티 그룹에 karpenter.sh/discovery={cluster name} 로 태깅한다.
카펜터가 신규 노드를 프로비저닝 하면, 이 값으로 태킹된 시큐리티 그룹을 찾아 신규 노드에 할당하게 된다.  
노드가 위치하게 되는 서브넷 역시 동일 매커니즘으로 동작하는데, 테라폼에서 이미 karpenter.sh/discovery={cluster name} 태깅을 완료하였다. 
```
NODEGROUP=$(aws eks list-nodegroups --cluster-name "${CLUSTER_NAME}" \
    --query 'nodegroups[0]' --output text)

LAUNCH_TEMPLATE=$(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name "${NODEGROUP}" --query 'nodegroup.launchTemplate.{id:id,version:version}' \
    --output text | tr -s "\t" ",")

SECURITY_GROUPS=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

SECURITY_GROUPS="$(aws ec2 describe-launch-template-versions \
    --launch-template-id "${LAUNCH_TEMPLATE%,*}" --versions "${LAUNCH_TEMPLATE#*,}" \
    --query 'LaunchTemplateVersions[0].LaunchTemplateData.[NetworkInterfaces[0].Groups||SecurityGroupIds]' \
    --output text)"

aws ec2 create-tags \
    --tags "Key=karpenter.sh/discovery,Value=${CLUSTER_NAME}" \
    --resources "${SECURITY_GROUPS}"
```

#### 2. 억세스 설정 ####
카펜터 버전 1.8.1 (EKS 1.3.4) 에는 아래의 두가지 설정이 누락되어 있어서 패치가 필요하다. 패치를 하지 않는 경우 카펜터가 프러비저닝한 노드가 클러스터에 조인되지 않는다.  
* eksctl-training-on-eks-iamservice-role 정책 추가 (OIDC 정책 누락)
``` 
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "VisualEditor0",
			"Effect": "Allow",
			"Action": "eks:DescribeCluster",
			"Resource": "arn:aws:eks:${REGION}:${AWS_ACCOUNT_ID}:cluster/${CLUSTER_NAME}"
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
```

* IAM Identity Mapping (이 부분은 필요한 설정인지 확인이 필요하다) 
```
eksctl create iamidentitymapping \
  --username system:node:{{EC2PrivateDNSName}} \
  --cluster "${CLUSTER_NAME}" \
  --arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER_NAME}-eks-iamservice-role" \
  --group system:bootstrappers \
  --group system:nodes

kubectl describe configmap aws-auth -n kube-system
```

## nginx 실행해 보기 ##
[nginx.yaml]
```
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
```
```
kubectl apply -f nginx.yaml
```

## 레퍼런스 ##

* [eksctl 사용 설명서](https://docs.aws.amazon.com/ko_kr/eks/latest/eksctl/what-is-eksctl.html)
* [eksctl EKS 설치 예제](https://www.kubeai.org/installation/eks/)
* [Enable an IAM User or IAM Role to access an EKS cluster](https://www.javierinthecloud.com/enable-an-iam-user-or-iam-role-to-access-an-eks-cluster/)


 ## todo ##
 * eksctl 로 클러스터를 생성하기 전에 시큐리티 그룹을 만들고, 그것으로 eks 에 붙인다.
 * discovery.sh.. 설정도 넣어야 한다.. 그렇게 하면 태깅이 불필요하게 된다. 


