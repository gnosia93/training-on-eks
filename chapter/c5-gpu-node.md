카펜터는 스케줄링 되지 않는 파드가 클러스터 이벤트 로그에 발견되면, 이를 해결하기 위해 신규 노드를 자동으로 프로비저닝 한다. 현재 클러스터 노드 그룹은 2개로 (ng_x86 과 ng_grav) GPU 인스턴스를 스케줄링 할수 없는 그룹들이다. 
이를 해결하기 위해서는 GPU 설정을 가지고 있는 신규 노드 그룹을 만들거나, 카펜터를 이용하여 동적으로 인스턴스를 프러비저닝 해 줘야 한다. 
현재의 클러스터 설정으로는 파드에서 GPU 리소스를 요청하는 경우 해당 파드는 pending 상태에 빠지게 된다.  

## [카펜터 설치하기](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/) ##

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/comp%20matrix.png)

```
aws eks describe-cluster --name training-on-eks --query cluster.version
```
[결과]
```
"1.33"
```


```
export CLUSTER_NAME="YOUR_EKS_CLUSTER_NAME" # EKS 클러스터 이름으로 변경 (예: my-eks-cluster)
export AWS_REGION="ap-northeast-2"         # AWS 리전으로 변경
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export KARPENTER_VERSION="v0.35.2"          # 사용하려는 Karpenter 버전
export KARPENTER_NAMESPACE="karpenter"

# OIDC는 Karpenter가 EKS와 통신하기 위한 필수 요구사항입니다.
eksctl utils associate-iam-oidc-provider --cluster $CLUSTER_NAME --approve --region $AWS_REGION

2단계: Karpenter IAM 역할 및 정책 수동 설정 (가장 중요)
Karpenter 컨트롤러가 EC2 인스턴스를 생성/삭제할 수 있는 권한을 부여해야 합니다. 이 단계는 스크립트로 완전히 자동화하기 어렵기 때문에 수동으로 진행합니다.

# karpenter-policy.json 파일을 다운로드합니다.
curl -fsSL raw.githubusercontent.com{KARPENTER_VERSION}/website/content/en/docs/getting-started/getting-started-with-eks/cloudformation.yaml | jq -r '.Resources.KarpenterControllerPolicy.Properties.PolicyDocument' > karpenter-policy.json

# IAM 정책을 AWS에 생성합니다.
aws iam create-policy --policy-name KarpenterControllerPolicy-${CLUSTER_NAME} --policy-document file://karpenter-policy.json

# Trust policy 파일 다운로드 (EKS 클러스터 OIDC URL 사용)
OIDC_PROVIDER=$(aws eks describe-cluster --name ${CLUSTER_NAME} --query "cluster.identity.oidc.issuer" --output text | sed -e 's/^https:\/\///')

cat <<EOF > trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_PROVIDER}:aud": "sts.amazonaws.com",
          "${OIDC_PROVIDER}:sub": "system:serviceaccount:${KARPENTER_NAMESPACE}:karpenter"
        }
      }
    }
  ]
}
EOF

# IAM 역할 생성
aws iam create-role --role-name KarpenterControllerRole-${CLUSTER_NAME} --assume-role-policy-document file://trust-policy.json

# 역할에 정책 연결
aws iam attach-role-policy --role-name KarpenterControllerRole-${CLUSTER_NAME} --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/KarpenterControllerPolicy-${CLUSTER_NAME}


# 환경 변수 다시 설정 (2단계에서 사용한 변수 유지 시 생략 가능)
export CLUSTER_NAME="YOUR_EKS_CLUSTER_NAME"
export AWS_REGION="ap-northeast-2"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export KARPENTER_VERSION="v0.35.2"
export KARPENTER_NAMESPACE="karpenter"

# 1. 네임스페이스 생성
kubectl create namespace ${KARPENTER_NAMESPACE}

# 2. Helm repository 추가
helm repo add karpenter charts.karpenter.sh
helm repo update

# 3. Helm 설치 (ServiceAccount에 2단계에서 만든 Role ARN 연결)
helm upgrade --install karpenter karpenter/karpenter --namespace ${KARPENTER_NAMESPACE} \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::${ACCOUNT_ID}:role/KarpenterControllerRole-${CLUSTER_NAME} \
  --set settings.clusterName=${CLUSTER_NAME} \
  --set defaultProvisioner.spec.cluster.name=${CLUSTER_NAME} \
  --set settings.interruptionQueueName=${CLUSTER_NAME} \
  --version ${KARPENTER_VERSION}

```
#### 설치확인 ####
```
kubectl get pods -n karpenter
```



## gpu 노드풀 준비 ##
EKS 오토모드에서 아래와 같이 두개의 노드풀이 자동으로 생성되지만, gpu 파드를 스케줄링 할 수는 없다. 노드풀의 세부 설정을 describe 해 보면
C, M, R 인스턴스 타입(CPU) 만을 가지고 있어 CPU를 사용하는 파드만 스케줄링이 가능하다.
```
kubectl get nodepools -n karpenter
```
[결과]
```
NAME              NODECLASS   NODES   READY   AGE
general-purpose   default     0       True    8h
system            default     1       True    8h
```
[참고] 아래는 카펜터 노드풀의 상세 설정을 조회할 수 있는 명령어 셋이다.
```
kubectl describe nodepool system -n karpenter
kubectl describe nodepool general-purpose -n karpenter
```   

또한 카펜터에서 제공하는 CRD와는 다른 별도의 CRD 를 사용하고, 사용할 수 있는 레이블 역시 오픈 소스 카펜터와는 다른 것을 사용한다. 예를들어 
기존의 karpenter.k8s.aws/instance-category 레이블은 오토모드에서 eks.amazonaws.com/instance-category 으로 변경되었다 (자세한 내용은 https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html 참조) 
```
kubectl get crd -o wide
```
[결과]
```
NAME                                            CREATED AT
applicationnetworkpolicies.networking.k8s.aws   2025-12-09T17:19:41Z
clusternetworkpolicies.networking.k8s.aws       2025-12-09T17:19:41Z
clusterpolicyendpoints.networking.k8s.aws       2025-12-09T17:19:42Z
cninodes.eks.amazonaws.com                      2025-12-09T17:21:32Z
cninodes.vpcresources.k8s.aws                   2025-12-09T17:19:42Z
ingressclassparams.eks.amazonaws.com            2025-12-09T17:21:32Z
nodeclaims.karpenter.sh                         2025-12-09T17:21:32Z
nodeclasses.eks.amazonaws.com                   2025-12-09T17:21:32Z
nodediagnostics.eks.amazonaws.com               2025-12-09T17:21:32Z
nodepools.karpenter.sh                          2025-12-09T17:21:32Z
policyendpoints.networking.k8s.aws              2025-12-09T17:19:41Z
securitygrouppolicies.vpcresources.k8s.aws      2025-12-09T17:19:42Z
targetgroupbindings.eks.amazonaws.com           2025-12-09T17:21:32Z
```
crd 를 조회해 보면 api 가 변경된 것을 확인할 수 있다. 노드풀의 경우 karpenter.sh 에 있으나, 노드 클래스의 경우 eks.amazonaws.com 으로 변경되었다.




### [gpu 노드풀 생성](https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html) ###

여기서는 gpu-pool 을 신규로 생성 예정인데, 노드 클래스의 경우 EKS 오토모드에서 기본으로 제공하는 default 클래스를 사용할 것이다. 노드 클래스 역시 기존의 EC2NodeClass 와 비교해서 상당 부분이 바뀌였다. (세부적인 내용은 https://docs.aws.amazon.com/eks/latest/userguide/create-node-class.html 에서 참조)

[gpu-nodepool.yaml] 
```
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-pool
spec:
  template:
    spec:
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: default
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
        
        # 특정 세대(예: 4세대 이상) 또는 특정 타입만 허용할 수 있습니다.
        # - key: karpenter.k8s.aws/instance-type
        #   operator: In
        #   values: ["g5.xlarge", "p3.2xlarge"]
        
      # GPU 노드임을 명시하는 Taint 추가 (GPU Pod만 스케줄링되도록 유도)
      taints:
        - key: "gpu-workload"
          effect: "NoSchedule"
```

GPU 파드를 실행할 수 있는 노드풀을 생성하고 READY 상태를 확인한다.   
```
kubectl apply -f gpu-nodepool.yaml
kubectl get nodepool
```
[결과]
```
NAME              NODECLASS   NODES   READY   AGE
general-purpose   default     0       True    12h
gpu-pool          default     0       True    8s
system            default     1       True    12h
```

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


