카펜터는 스케줄링 되지 않는 파드가 클러스터 이벤트 로그에 발견되면, 이를 해결하기 위해 신규 노드를 자동으로 프로비저닝 한다. 현재 클러스터 노드 그룹은 2개로 (ng_x86 과 ng_grav) GPU 인스턴스를 스케줄링 할수 없는 그룹들이다. 
이를 해결하기 위해서는 GPU 설정을 가지고 있는 신규 노드 그룹을 만들거나, 카펜터를 이용하여 동적으로 인스턴스를 프러비저닝 해 줘야 한다. 
현재의 설정으로는 GPU 리소스를 요청하는 파드는 pending 상태에 빠지게 된다.  

## [카펜터 설치하기](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/) ##

본 워크샵에서 사용하는 EKS 클러스터의 버전은 1.33 으로 아래의 명령어를 통해서 확인할 수 있다. 
```
aws eks describe-cluster --name training-on-eks --query cluster.version
```
카펜터를 설치하기 전에 가장 먼저해야 하는 일은 EKS 클러스터와의 호환성을 확인하는 것으로, 아래 호환성 메트릭을 보면 버전 1.5 이상의 카펜터가 필요한 것을 알 수 있다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/comp%20matrix.png)


#### 1. 환경변수 설정 및 OIDC 등록 ####

OIDC_ENDPOINT는 EKS 클러스터가 발급하는 모든 임시 자격 증명이 유효한지 검증할 수 있는 URL로, 쿠버네티스의 서비스 어카운트가 AWS 리소스에 안전하게 접근할 수 있게 해준다.
```
export KARPENTER_NAMESPACE=karpenter
export CLUSTER_NAME=training-on-eks
export AWS_PARTITION="aws" 
export AWS_REGION="$(aws configure list | grep region | tr -s " " | cut -d" " -f3)"
export OIDC_ENDPOINT="$(aws eks describe-cluster --name "${CLUSTER_NAME}" \
    --query "cluster.identity.oidc.issuer" --output text)"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export K8S_VERSION=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --query "cluster.version" --output text)

kubectl create ns ${KARPENTER_NAMESPACE}                                            # karpenter 네임스페이스 생성 
eksctl utils associate-iam-oidc-provider --cluster ${CLUSTER_NAME} --approve        # AWS IAM에서 OIDC 공급자 등록
```

#### 2. 카펜터 노드 및 컨트롤러 IAM Role 설정 ####
카펜터 노드 Role 은 카펜터 컨트롤러가 AWS 환경 내에서 사용자를 대신해 실제 컴퓨팅 자원(EC2 인스턴스)을 생성, 관리, 그리고 종료하는 일련의 작업을 수행할 때 사용된다. 이 Role은 카펜터가 클라우드 환경에서 노드의 수명 주기를 완벽하게 제어할 수 있도록 하는 필수적인 역할을 한다.   
```
curl -s https://raw.githubusercontent.com/gnosia93/training-on-eks/refs/heads/main/karpenter/KarpenterNodeRole.sh | sh
```

카펜터 컨트롤러는 EKS 클러스터에서 노드의 자동 생성, 관리, 종료를 전담하는 핵심 소프트웨어, 대기 중인 파드(Unschedulable Pods)가 있는지 계속 감시하고, 신규 파드의 CPU, 메모리, 특정 하드웨어(GPU 등) 요구 사항을 분석하여 노드 필요성 판단한다. 또한 새 노드가 준비되면, 대기 중이던 파드를 새로 생성된 노드에 직접 할당하거나, 더 이상 사용되지 않아 유휴 상태이거나 비효율적인 노드를 감지하면 해당 노드를 안전하게 종료하기도 한다. 
여기서는 카펜터 컨트롤러가 신규 인스턴스를 프로비저닝하는 데 필요한 IAM Role을 생성하는데, 카펜터 컨트롤러는 서비스 어카운트용 IAM 역할(IRSA)을 사용하며 OIDC 엔드포인트와 통신한다. 카펜터 컨트롤러는 OIDC 엔드포인트를 통해 발급받은 신뢰할 수 있는 신분증(ID 토큰)을 사용하여 AWS에 자신의 신분을 증명하며, 이를 통해 IRSA에 정의된 필요한 권한만을 안전하게 위임받아 작업을 수행한다.
```
curl -s https://raw.githubusercontent.com/gnosia93/training-on-eks/refs/heads/main/karpenter/KarpenterControllerRole.sh | sh
```

#### 3. 서브넷 및 시큐리티 그룹 태깅 ####

카펜터가 EC2 인스턴스를 새로 생성할때, 해당 인스턴스가 어느 네트워크(서브넷)에 위치해야 하고 어떤 네트워크 규칙(시큐리티 그룹)을 따라야 하는지 알고 있어야 한다. 
카펜터는 기존 노드그룹(ng-arm, ng-x86)의 서브넷과 시큐리티 그룹을 그대로 사용하게 되는데, 이를 위해 karpenter.sh/discovery={cluster name} 태깅을 기존 서브넷과 시큐리티 그룹에 할당한다.  

```
##### 서븐넷 태깅
for NODEGROUP in $(aws eks list-nodegroups --cluster-name "${CLUSTER_NAME}" --query 'nodegroups' --output text); do
    aws ec2 create-tags \
        --tags "Key=karpenter.sh/discovery,Value=${CLUSTER_NAME}" \
        --resources $(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" \
                        --nodegroup-name "${NODEGROUP}" --query 'nodegroup.subnets' --output text)
done

##### 시큐리티 그룹 태깅
NODEGROUP=$(aws eks list-nodegroups --cluster-name "${CLUSTER_NAME}" \
    --query 'nodegroups[0]' --output text)

LAUNCH_TEMPLATE=$(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name "${NODEGROUP}" --query 'nodegroup.launchTemplate.{id:id,version:version}' \
    --output text | tr -s "\t" ",")

# If your EKS setup is configured to use only Cluster security group, then please execute -
SECURITY_GROUPS=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

# If your setup uses the security groups in the Launch template of a managed node group, then :
SECURITY_GROUPS="$(aws ec2 describe-launch-template-versions \
    --launch-template-id "${LAUNCH_TEMPLATE%,*}" --versions "${LAUNCH_TEMPLATE#*,}" \
    --query 'LaunchTemplateVersions[0].LaunchTemplateData.[NetworkInterfaces[0].Groups||SecurityGroupIds]' \
    --output text)"

aws ec2 create-tags \
    --tags "Key=karpenter.sh/discovery,Value=${CLUSTER_NAME}" \
    --resources "${SECURITY_GROUPS}"
```

#### 4. aws-auth 컨피드맵 수정 ####

새로운 워커 노드가 시작되면, 해당 노드는 EKS 컨트롤 플레인에 자기 자신을 등록하는 데, 이때 사용하는 것이 바로 우리가 생성한 카펜터 노드 IAM Role 이다. EKS 컨트롤 플레인은 aws-auth ConfigMap을 확인하여, 해당 IAM Role ARN이 mapRoles 목록에 있는지 확인하게 된다. 즉 IAM 역할이나 사용자가 쿠버네티스의 RBAC 에 매핑되어 있는지를 확인하는 것이다.
이 과정이 실패하면 노드는 클러스터에 정상적으로 합류하지 못하고 'NotReady' 상태로 머무르게 된다.
```
eksctl create iamidentitymapping \
  --username system:node:{{EC2PrivateDNSName}} \
  --cluster "${CLUSTER_NAME}" \
  --arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/KarpenterNodeRole-${CLUSTER_NAME}" \
  --group system:bootstrappers \
  --group system:nodes
```

```
kubectl describe configmap aws-auth -n kube-system
```
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/aws-auth.png)


#### 5. 카펜터 배포하기 ####
헬름의 template 옵션을 이용하여 karpenter.yaml 파일을 만들고 카펜터 디플로이먼트의 nodeAffinity 를 아래와 같이 수정한다. 
```
export KARPENTER_VERSION="1.8.3"

helm template karpenter oci://public.ecr.aws/karpenter/karpenter --version "${KARPENTER_VERSION}" --namespace "${KARPENTER_NAMESPACE}" \
    --set "settings.clusterName=${CLUSTER_NAME}" \
    --set "settings.interruptionQueue=${CLUSTER_NAME}" \
    --set "serviceAccount.annotations.eks\.amazonaws\.com/role-arn=arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:role/KarpenterControllerRole-${CLUSTER_NAME}" \
    --set controller.resources.requests.cpu=1 \
    --set controller.resources.requests.memory=1Gi \
    --set controller.resources.limits.cpu=1 \
    --set controller.resources.limits.memory=1Gi > karpenter.yaml
```

karpenter.yaml의 카펜터 디플로이먼트를 수정하여 카펜터 컨트롤러가 이미 존재하는 노드그룹의 노드에 스케줄링되도록 nodeAffinity 를 수정한다.  
yaml 의 경우 탭과 Identation 에 민감하니 수정시 주의를 요한다.  

#### karpenter.yaml 수정전 ####
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/karpenter.png)

#### karpenter.yaml 에 추가할 내용 ####
```
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: karpenter.sh/nodepool
          operator: DoesNotExist
        - key: eks.amazonaws.com/nodegroup         # 여기서 부터 
          operator: In
          values:
          - ng-arm
          - ng-x86                                 # 여기까지 karpenter.yaml 의 420번 라인 이후로 추가 
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - topologyKey: "kubernetes.io/hostname"
```
#### 수정후 #### 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/karpenter-after.png)

아래 스크립트로 CRD(노드풀, 노드 클래스, 노드 클레임)와 카펜터 오브젝트들을 설치한다. 
```
kubectl create -f \
    "https://raw.githubusercontent.com/aws/karpenter-provider-aws/v${KARPENTER_VERSION}/pkg/apis/crds/karpenter.sh_nodepools.yaml"
kubectl create -f \
    "https://raw.githubusercontent.com/aws/karpenter-provider-aws/v${KARPENTER_VERSION}/pkg/apis/crds/karpenter.k8s.aws_ec2nodeclasses.yaml"
kubectl create -f \
    "https://raw.githubusercontent.com/aws/karpenter-provider-aws/v${KARPENTER_VERSION}/pkg/apis/crds/karpenter.sh_nodeclaims.yaml"
kubectl apply -f karpenter.yaml
```

#### 6. 카펜터 동작 확인 ####
```
kubectl get all -n karpenter
```
[결과]
```
NAME                             READY   STATUS    RESTARTS   AGE
pod/karpenter-565db98b46-4d9km   1/1     Running   0          31s
pod/karpenter-565db98b46-pmt99   1/1     Running   0          31s

NAME                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
service/karpenter   ClusterIP   172.20.216.254   <none>        8080/TCP   19m

NAME                        READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/karpenter   2/2     2            2           12m

NAME                                   DESIRED   CURRENT   READY   AGE
replicaset.apps/karpenter-565db98b46   2         2         2       31s
```

* 카펜터 출력 로그 확인
```
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter
```



## GPU 노드풀 준비 ##

아래 조회 결과에서 볼수 있는 것처럼 현재 클러스터에는 GPU를 스케줄링 할수 있는 카펜터 노드풀이 존재하지 않는다.
클러스터 생성시 만들어진 노드그룹(ng-arm, ng-x86) 역시 CPU 만으로 구성되어져 있다.  
```
kubectl get nodepools -n karpenter
```
[결과]
```
No resources found
```

CRD 는 사용자 정의 리소스 정의(Custom Resource Definition) 를 의미하는 것으로, 파드/서비스/디폴로이먼트와 같은 내장된 리소스 객체이외에 
사용자가 원하는 형태의 새로운 리소스 타입을 쿠버네티스 API로 추가할 수 있게 해주는 강력한 기능이다. 
CRD 로 필요한 리소스 타입을 정의하고, Operator 즉 쿠버네티스 컨트롤러를 사용자가 직접 그 기능을 구현하면 된다. 카펜터 역시 CRD 의 한 유형이다.   

GPU 노드풀을 만들기 전에, 먼저 카펜터 CRD를 조회하여 해당 API 의 도메인를 확인하도록 한다. 
노드 클래스는 karpenter.k8s.aws 사용하고, 노드 클레임과 노드풀은 karpenter.sh 도메인을 사용하고 있는 것을 확인할 수 있다. 
참고로 EKS Auto 모드의 경우 오픈소스 카펜터와는 별도의 CRD 를 사용하고 있으며 API 도메인 역시 동일하지 않다. (다른 CRD 임)    
```
kubectl get crd -o wide | grep karpenter
```
[결과]
```
ec2nodeclasses.karpenter.k8s.aws                2025-12-14T04:26:24Z
nodeclaims.karpenter.sh                         2025-12-14T04:26:25Z
nodepools.karpenter.sh                          2025-12-14T04:26:23Z
```

#### 1. DLAMI 조회 ####
```
export PYTORCH_DLAMI=$(aws ec2 describe-images --region ${AWS_REGION} --owners amazon \
    --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Amazon Linux 2023) ????????' \
    'Name=state,Values=available' --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' \
    --output text)

echo ${PYTORCH_DLAMI}
```
[결과]
```
ami-00aec23bbe1278680
```

#### 2. GPU 노드풀 생성하기 #### 
```
cat <<EOF | envsubst | kubectl apply -f -
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-pool
  namespace: karpenter
spec:
  template:
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]            # X86 만 설정  
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]        # 온디맨드 인스턴스 사용        
        - key: eks.amazonaws.com/instance-category 
          operator: In
          values: ["g", "p"]           # GPU 인스턴스 사용  
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: default
      taints:                          # GPU 노드임을 명시하는 Taint 추가 (GPU Pod만 스케줄링되도록 유도)
        - key: "gpu-workload"
          effect: "NoSchedule"
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 1m
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
  namespace: karpenter
spec:
  role: "KarpenterNodeRole-${CLUSTER_NAME}"
  amiFamily: AL2023                                 # AMI 계열을 명시         
  amiSelectorTerms:
    - id: "${PYTORCH_DLAMI}"
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER_NAME}"   
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER_NAME}"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 100Gi                           # root 보륨 크기 - 40GB 이상 권장
        volumeType: gp3
        deleteOnTermination: true
EOF
```
GPU 파드를 실행할 수 있는 노드풀을 생성한다.   
```
kubectl get nodepool
```
[결과]
```
NAME       NODECLASS   NODES   READY   AGE
gpu-pool   default     0       False   5s
```

### 부연설명 ###

컨테이너 환경에서 GPU 워크로드(AI/ML)를 실행할 때, 컨테이너 이미지(Container Image)와 AMI (Amazon Machine Image)는 명확하게 역할이 구분된다
핵심 원칙은 AMI는 하드웨어와 쿠버네티스 환경을 준비하고, 컨테이너 이미지는 애플리케이션 실행 환경을 준비한다. 

#### 1. AMI 에서 담당하는 부분 ###
* 운영 체제 (OS): Amazon Linux, Ubuntu 등 기본 OS 커널.
* NVIDIA GPU 드라이버: 가장 중요합니다. GPU 하드웨어를 직접 제어하는 낮은 수준의 드라이버는 호스트 OS(AMI) 레벨에 설치되어야 합니다. 컨테이너 내부에서는 하드웨어에 직접 접근할 수 없기 때문입니다.
* CUDA 런타임 라이브러리 (선택적/기본): 드라이버와 함께 설치되는 기본 CUDA 라이브러리 일부가 포함됩니다.
* 컨테이너 런타임: Docker, containerd 등 컨테이너를 실행하는 엔진.
* 쿠버네티스 에이전트: kubelet, aws-node (CNI), nvidia-container-runtime (Docker 대신 GPU 사용을 위한 런타임) 등 클러스터 조인을 위한 필수 소프트웨어.

GPU를 인식하고 사용할 수 있도록 하드웨어 접근 권한을 설정하는 것이 AMI의 역할입니다.

#### 2. 컨테이너 이미지의 역할: 애플리케이션 및 종속성 준비 ####
컨테이너 이미지는 애플리케이션 코드와 그 실행에 필요한 모든 라이브러리 및 환경 변수를 포함하는 격리된 패키지입니다.
* ML 프레임워크: PyTorch, TensorFlow, MXNet 등 실제 AI/ML 프레임워크.
* CUDA 툴킷 및 호환 라이브러리: 애플리케이션이 빌드될 때 사용된 특정 버전의 cuda-toolkit, cuDNN, NCCL 라이브러리. (이 버전은 AMI의 드라이버 버전과 호환되어야 합니다.)
* 애플리케이션 코드: 학습 스크립트 (train.py 등).
* 파이썬 및 종속성: Python 인터프리터, pip install로 설치된 라이브러리들 (numpy, pandas 등).

AMI가 제공하는 드라이버를 활용하여 애플리케이션을 실행하는 것이 컨테이너 이미지의 역할입니다.

#### 3. EKS 최적화 AMI vs DLAMI ####
* EKS 최적화 AMI: 쿠버네티스 구동에 필요한 kubelet, aws-node CNI 등이 설치되어 있지만, GPU 드라이버나 ML 프레임워크는 포함되어 있지 않습니다.
* Deep Learning AMI: GPU 드라이버, CUDA가 완벽하게 설정되어 있어 ML 워크로드에 최적입니다. Karpenter는 이 AMI를 사용하여 새 노드를 띄운 후, 자동으로 EKS 클러스터에 조인하는 부트스트랩 스크립트를 실행합니다.


## GPU 파드 스케줄링 ##

[도커허브 nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) 로 가서 nvidia-smi 가 설치되어 있는 컨테이너 이미지를 확인한다. 해당 페이지에서 아래로 스크롤하면 최신 컨테이너 이미지 정보를 확인할 수 있다.    
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/cuda-container.png)
이번 워크샵에서는 13.0.2-runtime-ubuntu22.04 도커 이미지로 nvidia-smi 를 실행할 예정이다.   

[gpu-pod.yaml]
```
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:13.0.2-runtime-ubuntu22.04    # runtime 이미지 사용
      command: ["nvidia-smi"]                          # 컨테이너 시작 시 실행할 프로그램
      resources:
        limits:
          nvidia.com/gpu: 1
  tolerations:                                             
    - key: "gpu-workload"                              # GPU 노드풀에 파드를 스케줄링하기 위해서 toleration 을 설정한다.        
      operator: "Exists"
      effect: "NoSchedule"
```

파드를 생성하고 nvidia-smi 가 동작하는 확인한다.  
```
kubectl apply -f gpu-pod.yaml
kubectl describe pod gpu-pod
kubectl logs gpu-pod
```
[출력]
```
Node-Selectors:              <none>
Tolerations:                 gpu-workload:NoSchedule op=Exists
                             node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:NoSchedule op=Exists
Events:
  Type     Reason            Age                  From                   Message
  ----     ------            ----                 ----                   -------
  Warning  FailedScheduling  5m12s                default-scheduler      0/2 nodes are available: 1 node(s) had untolerated taint {CriticalAddonsOnly: }, 1 node(s) had untolerated taint {karpenter.sh/disrupted: }. preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
  Normal   Scheduled         4m25s                default-scheduler      Successfully assigned default/gpu-pod to i-0b2714b5d7951c695
  Normal   Nominated         5m12s                eks-auto-mode/compute  Pod should schedule on: nodeclaim/gpu-pool-6csft
  Normal   Pulling           4m20s                kubelet                Pulling image "nvidia/cuda:13.0.2-runtime-ubuntu22.04"
  Normal   Pulled            3m21s                kubelet                Successfully pulled image "nvidia/cuda:13.0.2-runtime-ubuntu22.04" in 58.965s (58.965s including waiting). Image size: 1766399024 bytes.
  Normal   Created           12s (x6 over 3m21s)  kubelet                Created container: cuda-container
  Normal   Started           12s (x6 over 3m21s)  kubelet                Started container cuda-container
  Normal   Pulled            12s (x5 over 3m10s)  kubelet                Container image "nvidia/cuda:13.0.2-runtime-ubuntu22.04" already present on machine
  Warning  BackOff           0s (x15 over 3m9s)   kubelet                Back-off restarting failed container cuda-container in pod gpu-pod_default(4b7846fb-2e8f-4ba7-9d6d-ee5711363436)


Wed Dec 10 06:44:46 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA T4G                     On  |   00000000:00:1F.0 Off |                    0 |
| N/A   49C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
(참고) describe 의 출력 결과중 마지막 라인의 Warning BackOff 의 경우 컨테이너는 종료하였으나 파드가 살아있기 때문에 발생하는 메시지이다. 즉 nvidia-smi 는 실행을 종료하였으나 파드는 살아있다.

## 레퍼런스 ##

* https://pinetree0308.tistory.com/204


