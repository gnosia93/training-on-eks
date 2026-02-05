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
nvdp/gpu-feature-discovery      0.18.2          0.18.2          A Helm chart for gpu-feature-discovery on Kuber...
nvdp/nvidia-device-plugin       0.18.2          0.18.2          A Helm chart for the nvidia-device-plugin on Ku...
```
```
helm install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia \
  --create-namespace \
  --version 0.18.2 \
  --set gfd.enabled=true

kubectl get daemonset -n nvidia
```
[결과]
```
NAME                                              DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                 AGE
nvdp-node-feature-discovery-worker                4         4         4       4            4           <none>                        7d15h
nvdp-nvidia-device-plugin                         0         0         0       0            0           <none>                        7d15h
nvdp-nvidia-device-plugin-gpu-feature-discovery   0         0         0       0            0           <none>                        7d15h
nvdp-nvidia-device-plugin-mps-control-daemon      0         0         0       0            0           nvidia.com/mps.capable=true   7d15h
```

## GPU 노드풀 생성 ##
```
cat <<EOF > nodepool-gpu.yaml 
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu
spec:
  template:
    metadata:
      labels:
        nodeType: "nvidia" 
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["g", "p"]
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu
      expireAfter: 720h # 30 * 24h = 720h
      taints:
      - key: "nvidia.com/gpu"            # nvidia-device-plugin 데몬은 nvidia.com/gpu=present:NoSchedule 테인트를 Tolerate 한다. 
        value: "present"                 # value 값으로 present 와 다른값을 설정하면 nvidia-device-plugin 이 동작하지 않는다 (GPU를 찾을 수 없다)   
        effect: NoSchedule               # nvidia-device-plugin 이 GPU 를 찾으면 Nvidia GPU 관련 각종 테인트와 레이블 등을 노드에 할당한다.  
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmpty       # 이전 설정값은 WhenEmptyOrUnderutilized / 노드의 잦은 Not Ready 상태로의 변경으로 인해 수정  
    consolidateAfter: 20m
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu
spec:
  role: "eksctl-KarpenterNodeRole-training-on-eks"
  amiSelectorTerms:
    # Required; when coupled with a pod that requests NVIDIA GPUs or AWS Neuron
    # devices, Karpenter will select the correct AL2023 accelerated AMI variant
    # see https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/
    # EKS GPU Optimized AMI: NVIDIA 드라이버와 CUDA 런타임만 포함된 가벼운 이미지 (Karpenter가 자동으로 선택 가능) 가 설치됨.
    # 특정 DLAMI 가 필요한 경우 - name : 필드에 정의해야 함. 
    - alias: al2023@latest
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks" 
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks" 
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
EOF
```
ec2nodeclass 와 nodepol 을 생성한다.
```
kubectl apply -f nodepool-gpu.yaml
```
ec2nodeclass 와 nodepol 의 READY 필드의 값이 True 임것을 확인한다.
```
kubectl get ec2nodeclass,nodepool
```
[결과]
```
NAME                                 READY   AGE
ec2nodeclass.karpenter.k8s.aws/gpu   True    17s

NAME                        NODECLASS   NODES   READY   AGE
nodepool.karpenter.sh/gpu   gpu         0       True    17s
```

## nvidia-smi 파드 스케줄링 ##
```
cat <<EOF | kubectl apply -f - 
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  restartPolicy: Never                                # 재시작 정책을 Never로 설정 (실행 완료 후 다시 시작하지 않음)
  containers:                                         # 기본값은 Always - 컨테이너가 성공적으로 종료(exit 0)되든, 에러로 종료(exit nonzero)되든 상관없이 항상 재시작
    - name: cuda-container                            # nvidia-smi만 실행하고 끝나는 파드에 이 정책이 적용되면, 종료 후 다시 실행을 반복하다가 결국 CrashLoopBackOff 상태가 됨.
      image: nvidia/cuda:13.0.2-runtime-ubuntu22.04    
      command: ["/bin/sh", "-c"]
      args: ["nvidia-smi && sleep 300"]                # nvidia-smi 실행 후 300초(5분) 동안 대기
      resources:
        limits:
          nvidia.com/gpu: 1
  tolerations:                                             
    - key: "nvidia.com/gpu"
      operator: "Exists"                      # 노드의 테인트는 nvidia.com/gpu=present:NoSchedule 이나,   
      effect: "NoSchedule"                    # Exists 연산자로 nvidia.com/gpu 키만 체크         
EOF
```

파드를 생성하고 nvidia-smi 가 동작하는 확인한다.  
```
kubectl get pods
kubectl logs gpu-pod
```
[출력]
```
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
eks-node-viewer 로 신규 이스턴스가 프로비저닝 된것을 확인한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/eks-node-viewer.png)

## 참고 - Role List ##

### 1. eksctl 클러스터 생성시 만들어 지는 전체 Role ###
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/training-on-eks-roles.png)

### 2. eksctl-training-on-eks-iamservice-role (관리자/컨트롤러) ###
* 주체: EKS 클러스터 안에서 실행 중인 Karpenter 포드(Pod)가 사용.
* 용도: Karpenter가 AWS API를 호출하여 "노드를 만들고, 삭제하고, 인스턴스 프로파일을 생성"할 수 있게 해주는 권한.
* 핵심 권한: ec2:RunInstances, iam:PassRole, iam:CreateInstanceProfile 등.
* 특징: IRSA(IAM Roles for Service Accounts)를 통해 Karpenter 서비스 어카운트에 연결.

### 3. eksctl-KarpenterNodeRole-training-on-eks (카펜터 노드 롤/Worker Node) ###
* 주체: Karpenter에 의해 새롭게 생성된 EC2 인스턴스(노드) 자체가 사용.
* 용도: 생성된 노드가 EKS 클러스터에 접속하고, ECR에서 이미지를 풀(Pull)하거나, VPC CNI와 통신하는 등 "K8s 워커 노드로서 동작"하기 위해 필요한 권한.
* 핵심 권한: AmazonEKSWorkerNodePolicy, AmazonEKS_CNI_Policy, AmazonEC2ContainerRegistryReadOnly 등.
* 특징: EC2NodeClass의 spec.role 부분에 명시되는 이름.

#### 카펜터 노드 롤 확인 ####
```
aws iam list-roles --query 'Roles[?contains(RoleName, `KarpenterNodeRole`)].RoleName'
```  



## 레퍼런스 ##

* [Amazon EKS optimized Amazon Linux 2023 accelerated AMIs now available](https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/)
* https://catalog.workshops.aws/karpenter/en-US
* [Kubernetes 환경에서 NVIDIA GPU 사용하기 - NVIDIA Device Plugin](https://sirzzang.github.io/dev/Dev-Kubernetes-GPU-Setting/#google_vignette)
