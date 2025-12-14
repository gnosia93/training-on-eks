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


## 클러스터 생성 ##

그라비톤 인스턴스에서 EKS를 생성할 예정이다. 그라비톤 인스턴스는 EKS 클러스터를 생성하기 위한 권한을 가지고 있어야 하는데 아래 도표는 그라비톤이 가져야 할 최소 권한 리스트이다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/previllege_For_EKS.png)
이 워크샵에서는 TOE_EKS_EC2_ROLE 을 만들어 편의상 AdminFullAccess 권한을 부여하였고, 이를 다시 그라비톤 인스턴스에 부여하였다. (좀더 세부적인 내용은 테라폼 코드를 참조)

또한 클러스터가 생성되는 네트워크상의 위치를 정해 주기위해서 VPC ID 와 서브넷 정보가 필요한데, 보안의 강화하기 위해 EKS 클러스터 워커노드는 프라이빗 서브넷에 위치하게 된다.

#### VPC ID 조회 ####
```
VPC_ID=$(aws ec2 describe-vpcs --filters Name=tag:Name,Values=training-on-eks --query "Vpcs[].VpcId" --output text)
echo ${VPC_ID}
```
[결과]
```
vpc-030b927274aa21417
```

#### 프라이빗 서브넷 리스트 조회 ####
```
aws ec2 describe-subnets \
    --filters "Name=tag:Name,Values=TOE-priv-subnet-*" "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].{ID:SubnetId, AZ:AvailabilityZone, Name:Tags[?Key=='Name']|[0].Value}" \
    --output table
```  
[결과]
```
----------------------------------------------------------------------
|                           DescribeSubnets                          |
+-----------------+----------------------------+---------------------+
|       AZ        |            ID              |        Name         |
+-----------------+----------------------------+---------------------+
|  ap-northeast-2d|  subnet-09b59089486e54bfd  |  TOE-priv-subnet-4  |
|  ap-northeast-2b|  subnet-0e521bd6de96308b8  |  TOE-priv-subnet-2  |
|  ap-northeast-2a|  subnet-099acb450b8051d06  |  TOE-priv-subnet-1  |
|  ap-northeast-2c|  subnet-010db3e6a658817d6  |  TOE-priv-subnet-3  |
+-----------------+----------------------------+---------------------+
```

### 클러스터 생성 ###

아래 YAML 파일에서 VPC ID 와 프라이빗 서브넷 값을 조회된 값으로 수정한다.   

[cluster-config.yaml]
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: training-on-eks
  version: "1.33"
  region: ap-northeast-2

vpc:
  id: vpc-030b927274aa21417           # VPC ID를 여기에 지정해야 합니다 (조회된 값으로 수정)
  subnets:
    private:                          # 프라이빗 서브넷 정보를 지정해야 합니다 (조회된 값으로 수정 - 4개의 서브넷 중 3개만 사용)
      subnet-099acb450b8051d06: { az: ap-northeast-2a }
      subnet-0e521bd6de96308b8: { az: ap-northeast-2b }
      subnet-010db3e6a658817d6: { az: ap-northeast-2c }      

managedNodeGroups:                    # 관리형 노드 그룹을 정의합니다.
  - name: ng-arm
    instanceType: c7g.2xlarge
    minSize: 2
    maxSize: 2
    desiredCapacity: 2
    amiFamily: AmazonLinux2023
    privateNetworking: true           # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다.
   
  - name: ng-x86
    instanceType: c6i.2xlarge
    minSize: 2
    maxSize: 2
    desiredCapacity: 2
    amiFamily: AmazonLinux2023
    privateNetworking: true           # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다. 
```

클러스터를 생성한다. 
```
eksctl create cluster -f cluster-config.yaml 
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
2025-12-13 13:33:20 [ℹ]  using Kubernetes version 1.33
2025-12-13 13:33:20 [ℹ]  creating EKS cluster "training-on-eks" in "ap-northeast-2" region with managed nodes
2025-12-13 13:33:20 [ℹ]  2 nodegroups (ng-arm, ng-x86) were included (based on the include/exclude rules)
2025-12-13 13:33:20 [ℹ]  will create a CloudFormation stack for cluster itself and 2 managed nodegroup stack(s)
2025-12-13 13:33:20 [ℹ]  if you encounter any issues, check CloudFormation console or try 'eksctl utils describe-stacks --region=ap-northeast-2 --cluster=training-on-eks'
2025-12-13 13:33:20 [ℹ]  Kubernetes API endpoint access will use default of {publicAccess=true, privateAccess=false} for cluster "training-on-eks" in "ap-northeast-2"
2025-12-13 13:33:20 [ℹ]  CloudWatch logging will not be enabled for cluster "training-on-eks" in "ap-northeast-2"
2025-12-13 13:33:20 [ℹ]  you can enable it with 'eksctl utils update-cluster-logging --enable-types={SPECIFY-YOUR-LOG-TYPES-HERE (e.g. all)} --region=ap-northeast-2 --cluster=training-on-eks'
2025-12-13 13:33:20 [ℹ]  default addons coredns, metrics-server, vpc-cni, kube-proxy were not specified, will install them as EKS addons
2025-12-13 13:33:20 [ℹ]  
2 sequential tasks: { create cluster control plane "training-on-eks", 
    2 sequential sub-tasks: { 
        2 sequential sub-tasks: { 
            1 task: { create addons },
            wait for control plane to become ready,
        },
        2 parallel sub-tasks: { 
            create managed nodegroup "ng-arm",
            create managed nodegroup "ng-x86",
        },
    } 
}
2025-12-13 13:33:20 [ℹ]  building cluster stack "eksctl-training-on-eks-cluster"
2025-12-13 13:33:20 [!]  1 error(s) occurred and cluster hasn't been created properly, you may wish to check CloudFormation console
2025-12-13 13:33:20 [ℹ]  to cleanup resources, run 'eksctl delete cluster --region=ap-northeast-2 --name=training-on-eks'
2025-12-13 13:33:20 [✖]  creating CloudFormation stack "eksctl-training-on-eks-cluster": operation error CloudFormation: CreateStack, https response error StatusCode: 400, RequestID: 1fb7ff59-938e-48f3-bf06-baf755736b29, AlreadyExistsException: Stack [eksctl-training-on-eks-cluster] already exists
Error: failed to create cluster "training-on-eks"
[ec2-user@ip-10-0-0-60 ~]$ eksctl create cluster -f cluster-config.yaml 
2025-12-13 13:36:04 [ℹ]  eksctl version 0.220.0
2025-12-13 13:36:04 [ℹ]  using region ap-northeast-2
2025-12-13 13:36:05 [✔]  using existing VPC (vpc-030b927274aa21417) and subnets (private:map[subnet-010db3e6a658817d6:{subnet-010db3e6a658817d6 ap-northeast-2c 10.0.6.0/24 0 } subnet-099acb450b8051d06:{subnet-099acb450b8051d06 ap-northeast-2a 10.0.4.0/24 0 } subnet-0e521bd6de96308b8:{subnet-0e521bd6de96308b8 ap-northeast-2b 10.0.5.0/24 0 }] public:map[])
2025-12-13 13:36:05 [!]  custom VPC/subnets will be used; if resulting cluster doesn't function as expected, make sure to review the configuration of VPC/subnets
2025-12-13 13:36:05 [ℹ]  nodegroup "ng-arm" will use "" [AmazonLinux2023/1.33]
2025-12-13 13:36:05 [ℹ]  nodegroup "ng-x86" will use "" [AmazonLinux2023/1.33]
2025-12-13 13:36:05 [!]  Auto Mode will be enabled by default in an upcoming release of eksctl. This means managed node groups and managed networking add-ons will no longer be created by default. To maintain current behavior, explicitly set 'autoModeConfig.enabled: false' in your cluster configuration. Learn more: https://eksctl.io/usage/auto-mode/
2025-12-13 13:36:05 [ℹ]  using Kubernetes version 1.33
2025-12-13 13:36:05 [ℹ]  creating EKS cluster "training-on-eks" in "ap-northeast-2" region with managed nodes
2025-12-13 13:36:05 [ℹ]  2 nodegroups (ng-arm, ng-x86) were included (based on the include/exclude rules)
2025-12-13 13:36:05 [ℹ]  will create a CloudFormation stack for cluster itself and 2 managed nodegroup stack(s)
2025-12-13 13:36:05 [ℹ]  if you encounter any issues, check CloudFormation console or try 'eksctl utils describe-stacks --region=ap-northeast-2 --cluster=training-on-eks'
2025-12-13 13:36:05 [ℹ]  Kubernetes API endpoint access will use default of {publicAccess=true, privateAccess=false} for cluster "training-on-eks" in "ap-northeast-2"
2025-12-13 13:36:05 [ℹ]  CloudWatch logging will not be enabled for cluster "training-on-eks" in "ap-northeast-2"
2025-12-13 13:36:05 [ℹ]  you can enable it with 'eksctl utils update-cluster-logging --enable-types={SPECIFY-YOUR-LOG-TYPES-HERE (e.g. all)} --region=ap-northeast-2 --cluster=training-on-eks'
2025-12-13 13:36:05 [ℹ]  default addons metrics-server, vpc-cni, kube-proxy, coredns were not specified, will install them as EKS addons
2025-12-13 13:36:05 [ℹ]  
2 sequential tasks: { create cluster control plane "training-on-eks", 
    2 sequential sub-tasks: { 
        2 sequential sub-tasks: { 
            1 task: { create addons },
            wait for control plane to become ready,
        },
        2 parallel sub-tasks: { 
            create managed nodegroup "ng-arm",
            create managed nodegroup "ng-x86",
        },
    } 
}
2025-12-13 13:36:05 [ℹ]  building cluster stack "eksctl-training-on-eks-cluster"
2025-12-13 13:36:05 [ℹ]  deploying stack "eksctl-training-on-eks-cluster"
```

### 서브넷 태깅 ### 

EKS 로드 밸런서와 인그레스는 서브넷 태그 정보를 이용하여, 프로비저닝 되는 위치를 정하게 된다. 퍼블릭 서브넷에는 kubernetes.io/role/elb=1 과 kubernetes.io/cluster/{cluster name}=owned 값을 설정하도록 하고 프라이빗 서브넷에는 kubernetes.io/role/internal-elb=1 을 설정하도록 한다.   

#### 퍼블릭 서브넷 리스트 조회 ####
```
aws ec2 describe-subnets \
    --filters "Name=tag:Name,Values=TOE-pub-subnet-*" "Name=vpc-id,Values=${VPC_ID}" \
    --query "Subnets[*].{ID:SubnetId, AZ:AvailabilityZone, Name:Tags[?Key=='Name']|[0].Value}" \
    --output table
```  
[결과]
```
---------------------------------------------------------------------
|                          DescribeSubnets                          |
+-----------------+----------------------------+--------------------+
|       AZ        |            ID              |       Name         |
+-----------------+----------------------------+--------------------+
|  ap-northeast-2a|  subnet-026bdcdeea230b1b3  |  TOE-pub-subnet-1  |
|  ap-northeast-2b|  subnet-0e246ca66e5c239a7  |  TOE-pub-subnet-2  |
|  ap-northeast-2d|  subnet-024c415fd4c7b2ae2  |  TOE-pub-subnet-4  |
|  ap-northeast-2c|  subnet-05cf75c4d41ccc74b  |  TOE-pub-subnet-3  |
+-----------------+----------------------------+--------------------+
```

퍼블릭 서브넷을 태깅한다.
```
aws ec2 create-tags --resources subnet-026bdcdeea230b1b3 subnet-0e246ca66e5c239a7 subnet-05cf75c4d41ccc74b \
  --tags Key=kubernetes.io/role/elb,Value=1 Key=kubernetes.io/cluster/training-on-eks,Value=owned \
  --region ap-northeast-2
```
프라이빗 서브넷을 태깅한다. 
```
aws ec2 create-tags --resources subnet-099acb450b8051d06 subnet-0e521bd6de96308b8 subnet-010db3e6a658817d6 \
  --tags Key=kubernetes.io/role/internal-elb,Value=1 \
  --region ap-northeast-2
```

### 클러스터 확인 ### 
* 현재 컨텍스트(생성된 클러스터)
```
kubectl config current-context
```
[결과]
```
i-048265208fb345ec5@training-on-eks.ap-northeast-2.eksctl.io
```
* 노드그룹
```
eksctl get nodegroup --cluster=training-on-eks --region=ap-northeast-2
```
```
CLUSTER         NODEGROUP       STATUS  CREATED                 MIN SIZE        MAX SIZE        DESIRED CAPACITY        INSTANCE TYPE   IMAGE ID                ASG NAME                                  TYPE
training-on-eks ng-arm          ACTIVE  2025-12-13T13:47:35Z    2               2               2                       c7g.2xlarge     AL2023_ARM_64_STANDARD  eks-ng-arm-a2cd8bfb-ba01-1252-3342-5cabc45b0b0b    managed
training-on-eks ng-x86          ACTIVE  2025-12-13T13:47:34Z    2               2               2                       c6i.2xlarge     AL2023_x86_64_STANDARD  eks-ng-x86-e8cd8bfb-ba1b-0f17-c83a-0db24ba49f87    managed
```
* 노드 리스트
```
kubectl get nodes -o wide 
```
```
NAME                                            STATUS   ROLES    AGE     VERSION               INTERNAL-IP   EXTERNAL-IP   OS-IMAGE                       KERNEL-VERSION                    CONTAINER-RUNTIME
ip-10-0-4-148.ap-northeast-2.compute.internal   Ready    <none>   7m27s   v1.33.5-eks-ecaa3a6   10.0.4.148    <none>        Amazon Linux 2023.9.20251208   6.12.58-82.121.amzn2023.aarch64   containerd://2.1.5
ip-10-0-4-191.ap-northeast-2.compute.internal   Ready    <none>   7m31s   v1.33.5-eks-ecaa3a6   10.0.4.191    <none>        Amazon Linux 2023.9.20251208   6.12.58-82.121.amzn2023.x86_64    containerd://2.1.5
ip-10-0-6-140.ap-northeast-2.compute.internal   Ready    <none>   7m32s   v1.33.5-eks-ecaa3a6   10.0.6.140    <none>        Amazon Linux 2023.9.20251208   6.12.58-82.121.amzn2023.x86_64    containerd://2.1.5
ip-10-0-6-224.ap-northeast-2.compute.internal   Ready    <none>   7m27s   v1.33.5-eks-ecaa3a6   10.0.6.224    <none>        Amazon Linux 2023.9.20251208   6.12.58-82.121.amzn2023.aarch64   containerd://2.1.5
```


### nginx 배포해 보기 ### 

* TBD








