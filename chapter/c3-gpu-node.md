## [디바이스 플러그인 설치](https://docs.aws.amazon.com/eks/latest/userguide/ml-eks-k8s-device-plugin.html) ##

쿠버네티스는 CPU 및 메모리 같은 일반적인 리소스만 관리할 수 있고, GPU의 존재에 대해서는 알지 못한다.
Nvidia 디바이스 플러그인은 각 노드의 GPU를 감지하고, GPU에 대한 정보를 쿠버네티스 컨트롤 플레인에게 전달한다.
디바이스 플러그인은 GPU 드라이버 및 NVIDIA 컨테이너 런타임(예: nvidia-container-runtime)과 연동하여, 컨테이너가 호스트의 GPU 하드웨어에 직접 접근할 수 있도록  해준다.

```
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm search repo nvdp --devel
```
[결과]
```
NAME                            CHART VERSION   APP VERSION     DESCRIPTION                                       
nvdp/gpu-feature-discovery      0.18.0          0.18.0          A Helm chart for gpu-feature-discovery on Kuber...
nvdp/nvidia-device-plugin       0.18.0          0.18.0          A Helm chart for the nvidia-device-plugin on Ku...
```
```
helm install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia \
  --create-namespace \
  --version 0.18.0 \
  --set gfd.enabled=true

kubectl get daemonset -n nvidia
```

## 카펜터 확인 ##
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

#### GPU 노드풀 생성하기 #### 
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
      taints:                          
        - key: "nvidia.com"        # 새로 생성되는 노드에 GPU Taint를 적용합니다.
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
  amiSelectorTerms:
    # Required; when coupled with a pod that requests NVIDIA GPUs or AWS Neuron
    # devices, Karpenter will select the correct AL2023 accelerated AMI variant
    # see https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/
    - alias: al2023@latest
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER_NAME}"   
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER_NAME}"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 100Gi                           # root 볼륨 크기 - 40GB 이상 권장
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

## nvidia-smi 파드 스케줄링 ##

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
    - key: "nvidia.com"
      operator: "Exists"
      effect: "NoSchedule"                             # GPU를 요청하는 Pod만 스케줄되도록 강제합니다.
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

* [Karpenter Workshop](https://catalog.workshops.aws/karpenter/en-US)
* [Amazon EKS optimized Amazon Linux 2023 accelerated AMIs now available](https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/)
* https://sirzzang.github.io/dev/Dev-Kubernetes-GPU-Setting/
* https://www.youtube.com/watch?v=dS6UIovSXpA 
