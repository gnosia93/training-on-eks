<< fault injection 이 동작하지 않는다. 왜 일까? >> 

* 설치방법 
```
eksctl create addon --cluster ${CLUSTER_NAME} --name eks-node-monitoring-agent --version latest

   #\
   # --service-account-role-arn arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME} --force
```
ROLE 설정은 불필요하다.. 

* 설치후 로그 확인 
```
kubectl logs -f -l app.kubernetes.io/name=eks-node-monitoring-agent -n kube-system
```
* 폴트 주입
```
echo "<3>NVRM: Xid (PCI:00000000:00:1F.0): 31, Ch 00000001, failed to allocate GPU memory" | sudo tee /dev/kmsg

```
  



## 오류 메시지 ##
```
[Errno 2] No such file or directory: '/var/run/secrets/eks.amazonaws.com/serviceaccount/token'
panic: exit status 255

goroutine 1 [running]:
main.main()
        /local/p4clients/pkgbuild-const/workspace/src/EKSNodeMonitoringAgent/cmd/chroot/main.go:17 +0xec
{"level":"error","ts":"2025-12-25T03:01:17Z","msg":"failed to report managed conditions","hostname":"ip-10-0-4-234.ap-northeast-2.compute.internal","error":"failed to get server groups:
Get \"https://F1DD89A1F08A8008F71A4721C30B0435.sk1.ap-northeast-2.eks.amazonaws.com/api\":
getting credentials: exec: executable /opt/amazon/bin/chroot failed with exit code 2","stacktrace":"golang.a2z.com/EKSNodeMonitoringAgent/internal/manager.(*nodeExporter).run\n\t/local/p4clients/pkgbuild-const/workspace/src/EKSNodeMonitoringAgent/internal/manager/node_exporter.go:150\ngolang.a2z.com/EKSNodeMonitoringAgent/internal/manager.(*nodeExporter).Run\n\t/local/p4clients/pkgbuild-const/workspace/src/EKSNodeMonitoringAgent/internal/manager/node_exporter.go:137"}
^C[ec2-user@ip-10-0-0-122 ~]$ 
```


## raw 설치 ##
```
export CLUSTER_NAME="training-on-eks"
export AWS_REGION="ap-northeast-2"
export ROLE_NAME="eks-nma-role"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export K8S_VERSION="1.34"

echo "--- 1. 클러스터 OIDC 정보 확인 및 신뢰 정책 생성 ---"
# 클러스터의 OIDC ID 추출
OIDC_ENDPOINT=$(aws eks describe-cluster --name ${CLUSTER_NAME} --query "cluster.identity.oidc.issuer" --output text | sed 's/https:\/\///')


cat <<EOF > nma-trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_ENDPOINT}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_ENDPOINT}:sub": "system:serviceaccount:kube-system:eks-node-monitoring-agent",
          "${OIDC_ENDPOINT}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF



echo "--- 2. IAM Role 생성 및 정책 연결 ---"
# 1) IAM Role 생성
aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document file://nma-trust-policy.json || true

# 2) 필수 권한 정책 연결 (노드 모니터링에 필요한 표준 정책)
aws iam attach-role-policy --role-name ${ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy

echo "--- 3. EKS 애드온 설치 (Role 지정) ---"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# 최신 NMA 버전 가져오기
NMA_VERSION=$(aws eks describe-addon-versions \
    --kubernetes-version ${K8S_VERSION} \
    --addon-name eks-node-monitoring-agent \
    --query 'addons[0].addonVersions[?compatibilities[0].defaultVersion==`true`].addonVersion' \
    --output text)

aws eks create-addon \
    --cluster-name ${CLUSTER_NAME} \
    --addon-name eks-node-monitoring-agent \
    --addon-version ${NMA_VERSION} \
    --service-account-role-arn ${ROLE_ARN} \
    --resolve-conflicts OVERWRITE

echo "--- 4. RBAC 권한 패치 (노드 상태 수정 권한) ---"
# 애드온이 생성될 때까지 대기 후 실행 (혹은 명령어가 성공할 때까지 반복)
sleep 30
kubectl patch clusterrole eks-node-monitoring-agent --type='json' -p='[{"op": "add", "path": "/rules/-", "value": {"apiGroups": [""], "resources": ["nodes", "nodes/status"], "verbs": ["get", "patch", "update", "list", "watch"]}}]'

echo "설치가 완료되었습니다."


aws eks update-addon \
    --cluster-name training-on-eks \
    --addon-name eks-node-monitoring-agent \
    --service-account-role-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/eks-nma-role \
    --resolve-conflicts OVERWRITE


# 1. ServiceAccount에 IAM Role 주석 강제 삽입
 kubectl get sa eks-node-monitoring-agent -n kube-system -o yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::499514681453:role/eks-nma-role
  creationTimestamp: "2025-12-25T03:31:24Z"
  labels:
    app.kubernetes.io/component: controller
    app.kubernetes.io/instance: eks-node-monitoring-agent
    app.kubernetes.io/managed-by: EKS
    app.kubernetes.io/name: eks-node-monitoring-agent
    app.kubernetes.io/version: 1.4.3
  name: eks-node-monitoring-agent
  namespace: kube-system
  resourceVersion: "1107140"
  uid: cae74315-f9a8-4cc6-84f3-d9def0193462


# 2. 파드 강제 삭제 (새로운 토큰 주입 유도)
kubectl delete pods -n kube-system -l app.kubernetes.io/name=eks-node-monitoring-agent



kubectl get sa -n kube-system eks-node-monitoring-agent -o yaml
```
---
## IRSA 방식 ##
```
export CLUSTER_NAME="training-on-eks"
export REGION="ap-northeast-2"
export ROLE_NAME="eks-nma-irsa-role"

# eksctl을 사용하면 OIDC 신뢰 관계를 자동으로 설정하여 매우 간편합니다.
eksctl create iamserviceaccount \
    --name eks-node-monitoring-agent \
    --namespace kube-system \
    --cluster ${CLUSTER_NAME} \
    --role-name ${ROLE_NAME} \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy \
    --approve \
    --override-existing-serviceaccounts
```




### pod identity 방식 ###
```
# 1. 환경 변수 설정 (본인의 환경에 맞게 수정)
export CLUSTER_NAME="training-on-eks"
export REGION="ap-northeast-2"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ROLE_NAME="eks-nma-pod-identity-role"
export K8S_VERSION="1.34"

echo "--- 1. Pod Identity용 IAM Role 생성 ---"
# Pod Identity 서비스가 이 역할을 맡을 수 있도록 신뢰 정책 생성
cat <<EOF > nma-pod-identity-trust-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "pods.eks.amazonaws.com"
            },
            "Action": [
                "sts:AssumeRole",
                "sts:TagSession"
            ]
        }
    ]
}
EOF

# Role 생성 및 필수 정책(WorkerNodePolicy) 연결
aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document file://nma-pod-identity-trust-policy.json || true
aws iam attach-role-policy --role-name ${ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy

echo "--- 2. Pod Identity 연결(Association) 생성 ---"
# 클러스터, 네임스페이스, SA 이름, IAM Role을 하나로 묶음
# (SA는 애드온이 생성할 것이므로 미리 정의만 함)
aws eks create-pod-identity-association \
    --cluster-name ${CLUSTER_NAME} \
    --namespace kube-system \
    --service-account eks-node-monitoring-agent \
    --role-arn arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME} \
    --region ${REGION} || true

echo "--- 3. EKS NMA 애드온 설치 ---"
# 최신 버전 확인 (2025년 기준)
NMA_VERSION=$(aws eks describe-addon-versions \
    --kubernetes-version ${K8S_VERSION} \
    --addon-name eks-node-monitoring-agent \
    --query 'addons[0].addonVersions[?compatibilities[0].defaultVersion==`true`].addonVersion' \
    --output text)
echo "Node Monitoring Agent Version: "${NMA_VERSION}

aws eks create-addon \
    --cluster-name ${CLUSTER_NAME} \
    --addon-name eks-node-monitoring-agent \
    --addon-version ${NMA_VERSION} \
    --resolve-conflicts OVERWRITE

echo "애드온 배포 대기 중 (약 30초)..."
sleep 30
aws eks describe-addon --cluster-name ${CLUSTER_NAME} \
    --addon-name eks-node-monitoring-agent --query "addon.status" --output text


echo "--- 4. 쿠버네티스 RBAC 권한 패치 (중요!) ---"
# NMA가 노드 상태(nodes/status)를 수정할 수 있도록 내부 권한 부여
kubectl patch clusterrole eks-node-monitoring-agent --type='json' -p='[{"op": "add", "path": "/rules/-", "value": {"apiGroups": [""], "resources": ["nodes", "nodes/status"], "verbs": ["get", "patch", "update", "list", "watch"]}}]'

echo "--- 설치 완료! ---"
echo "이제 dcgmi 결함 주입 테스트를 진행하셔도 좋습니다."
```


## 폴트 주입 ##
```
export DCGM_POD=$(kubectl get pods -n dcgm -l app.kubernetes.io/name=dcgm-exporter -o jsonpath='{.items[0].metadata.name}')
echo ${DCGM_POD}
kubectl exec -n dcgm ${DCGM_POD} -- dcgmi test --inject --gpuid 0 -f 319 -v 4

```

## 노드 확인 ##
```
TARGET_NODE=$(kubectl get pod -n dcgm -l app.kubernetes.io/name=dcgm-exporter -o jsonpath='{.items[0].spec.nodeName}')
echo "Target Node: $TARGET_NODE"

# 해당 노드만 콕 집어서 상태 확인
kubectl get node $TARGET_NODE -o json | jq '.status.conditions[] | select(.type=="AcceleratedHardwareReady")'

NMA_POD=$(kubectl get pod -n kube-system -l app.kubernetes.io/name=eks-node-monitoring-agent --field-selector spec.nodeName=$TARGET_NODE -o jsonpath='{.items[0].metadata.name}')

kubectl logs -n kube-system $NMA_POD -f
```














## 1. Node Monitoring Agent 설치 ##
클러스터 설정에서 eks-node-monitoring-agent 애드온을 추가하여 설치해야 합니다. 이 에이전트가 노드의 로그를 분석하여 장애를 감지하는 역할을 합니다.
```
export CLUSTER_NAME="training-on-eks"
export K8S_VERSION="1.34"

# AWS가 권장하는 최적의 애드온 버전을 찾는다.
NMA_VERSION=$(aws eks describe-addon-versions \
    --kubernetes-version ${K8S_VERSION} \
    --addon-name eks-node-monitoring-agent \
    --query 'addons[0].addonVersions[?compatibilities[0].defaultVersion==`true`].addonVersion' \
    --output text)
echo "Node Monitoring Agent Version: "${NMA_VERSION}

# 해당 애드온을 설치한다. 
aws eks create-addon \
    --cluster-name ${CLUSTER_NAME} --addon-name eks-node-monitoring-agent \
    --addon-version ${NMA_VERSION}  

# 설치된 파드 확인
kubectl get pods -n kube-system | grep node-monitoring-agent
```
[결과]
```
eks-node-monitoring-agent-pm4kc   1/1     Running     0          44h
eks-node-monitoring-agent-rbgxp   1/1     Running     0          44h
```
Node Monitoring Agent는 GPU 노드의 상태를 정밀하게 진단하기 위해 내부적으로 NVIDIA의 핵심 라이브러리인 NVML(NVIDIA Management Library)과 DCGM을 활용한다.


## 2. 노드 타입별 설정 ##

### 2-1. 매니지드 노드 ###
관리형 노드 그룹에서는 오토 모드와 달리 사용자가 직접 기능을 활성화해야 한다.  
NMA 에이전트가 노드의 커널, 네트워크, 스토리지, GPU 상태를 모니터링하다가 문제가 발견되면 Kubernetes NodeCondition을 업데이트한다.
상태가 나빠진 노드를 발견하면 EKS 컨트롤 플래인이 해당 노드를 자동으로 격리(Cordon) 및 비우기(Drain)한 후, 새로운 인스턴스로 교체한다. 

#### 노드 그룹의 'Node Repair' 활성화 ####
에이전트만 설치한다고 복구가 자동으로 수행되지 않습니다. 매니지드 노드 그룹 설정에서 node-repair 기능을 Enabled로 변경해야 한다.
```
aws eks update-nodegroup-config --cluster-name <클러스터명> \
  --nodegroup-name <노드그룹명> \
  --node-repair-config enabled=true
```

### 2-2. 카펜터 노드 ###

카펜터는 기본적으로 NotReady 상태인 노드를 감지하여 교체한다. Node Monitoring Agent를 설치하면 NotReady가 되기 전의 미세한 하드웨어 장애(GPU 오류, 디스크 읽기 전용 등) 단계에서 더 빠르게 대응할 수 있다. 매니지드 노드 그룹에서는 AWS 백엔드가 복구를 수행하지만, 카펜터 환경에서는 Node Monitoring Agent 가 문제를 감지하고 카펜터가 노드를 교체하는 협업 구조로 작동한다. 
Node Monitoring Agent는 노드 장애를 발견되면, 해당 노드의 NodeCondition을 업데이트 한다. (예: StorageReady=False).
카펜터의 Disruption(중단) 컨트롤러가 이를 감지하고, 자동으로 교체(Terminate & Provision)해 준다. 

#### Disruption 컨트롤러의 역할 ####
Disruption 컨트롤러 클러스터의 상태를 계속 감지하면서 "지금 이 노드를 삭제하거나 교체해야 하는가?"를 결정하는데, 다음 3가지의 방식이 존재한다. 
* Drift (드리프트): 사용자가 NodePool 설정을 변경하여 실제 노드 사양과 설정이 달라졌을 때.
* Consolidation (최적화): 노드가 비어 있거나 더 저렴한 인스턴스로 합칠 수 있을 때.
* Repair (복구 - Node Monitoring Agent 연동): Node Monitoring Agent가 노드 장애를 보고하여 NodeCondition이 나빠졌을 때.

#### Node Repair 기능 활성화 ####
카펜터 버전 v1.1.0 이상부터는 기본적으로 Node Auto Repair 기능이 포함되어져 있어 별도의 설정이 필요하지 않다. 
다만, 카펜터가 노드를 너무 공격적으로 노드를 교체하지 않도록 NodePool 리소스에 Disruption Budgets를 설정하는 것을 고려해 볼 수도 있다.
```
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    budgets:
      - nodes: "10%" # 한 번에 교체될 수 있는 최대 노드 비율
```
아래 명령어를 이용하여 카펜터가 노드를 스케일링 하는 로그를 관찰할 수 있다. 
```
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter
```


## 에이전트 로그 실시간 모니터링 ##

에이전트가 노드의 커널 메시지나 시스템 로그를 제대로 파싱하고 있는지 확인한다.
```
kubectl logs -f -n kube-system -l app.kubernetes.io/instance=eks-node-monitoring-agent
```
[결과]
```
{"level":"info","ts":"2025-12-24T01:53:06Z","msg":"reported node conditions","hostname":"ip-10-0-6-164.ap-northeast-2.compute.internal"}
{"level":"info","ts":"2025-12-24T01:58:06Z","msg":"reporting managed conditions","hostname":"ip-10-0-6-164.ap-northeast-2.compute.internal"}
{"level":"info","ts":"2025-12-24T01:58:06Z","msg":"Skipping MAC address policy check - not needed for this OS","hostname":"ip-10-0-6-164.ap-northeast-2.compute.internal","monitor":"networking"}
```

## 장애 시뮬레이션 ##

#### 1. GPU 노드 할당 받기 ####
```
cat <<EOF > nvidia-smi.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:13.0.2-runtime-ubuntu22.04    # runtime 이미지 사용
      command: ["/bin/sh", "-c"]
      args: ["nvidia-smi && sleep infinity"]
      resources:
        limits:
          nvidia.com/gpu: 1
  tolerations:                                             
    - key: "nvidia.com/gpu"
      operator: "Exists"               # 노드의 테인트는 nvidia.com/gpu=present:NoSchedule 이나, Exists 연산자로 nvidia.com/gpu 키만 체크
      effect: "NoSchedule"
EOF
kubectl apply -f nvidia-smi.yaml && kubectl logs nvidia-smi
```

[결과]
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA T4G                     On  |   00000000:00:1F.0 Off |                    0 |
| N/A   58C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
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
Bus-Id 값이 00000000:00:1F.0 임을 확인할 수 있다.

#### 2. 노드 확인 ####
```
kubectl get nodes -o custom-columns="NAME:.metadata.name,STATUS:.status.conditions[-1].type,\
GPU_TOTAL:.status.capacity.nvidia\.com/gpu,\
GPU_AVAIL:.status.allocatable.nvidia\.com/gpu,\
INSTANCE:.metadata.labels.node\.kubernetes\.io/instance-type"
```
[결과]
```
NAME                                            STATUS                     GPU_TOTAL   GPU_AVAIL   INSTANCE
ip-10-0-4-115.ap-northeast-2.compute.internal   KernelReady                <none>      <none>      c7g.2xlarge
ip-10-0-4-138.ap-northeast-2.compute.internal   AcceleratedHardwareReady   1           1           g5g.xlarge
ip-10-0-6-164.ap-northeast-2.compute.internal   NetworkingReady            <none>      <none>      c6i.2xlarge
```


#### 3. GPU 오류 주입 ####


```
export NMA_NAME=$(kubectl get pod eks-node-monitoring-agent-zwsc9 -n kube-system -o jsonpath='{.spec.containers[*].name}')
echo ${NMA_NAME}
kubectl debug -it eks-node-monitoring-agent-zwsc9 -n kube-system \
    --image=nvidia/dcgm-exporter:4.4.2-4.7.1-ubuntu22.04 --target=${NMA_NAME} \
    --profile=sysadmin -- bash

nv-hostengine &
dcgmi discovery -l
```
[결과]
```
1 GPU found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA T4G                                                     |
|        | PCI Bus ID: 00000000:00:1F.0                                         |
|        | Device UUID: GPU-9ca02968-f496-c0c4-01d9-509cb6c32f5f                |
+--------+----------------------------------------------------------------------+
```
```
# 1. DCGM 그룹 생성 (이미 있으면 에러가 나지만 무시됨)
dcgmi group -c nma_test_group -a 0

# 2. 헬스 체크 시스템 활성화 (메모리, PCIe, NVLink 감시 시작)
# 이 명령어가 실행되어야 주입된 값이 '장애'로 판정됩니다.
dcgmi health -g $(dcgmi group -l | grep nma_test_group | awk '{print $6}') -s mpi

dcgmi health -f -g 5

# 3. 에러 주입 (가장 강력한 트리거들 조합)
echo "Injecting errors to GPU 0..."

# (A) XID 48: Double-bit ECC 에러 (NMA가 즉각 반응하는 Critical 에러)
dcgmi test --inject --gpuid 0 -f 150 -v 48

# (B) ECC Uncorrectable Volatile Total: 999개 (수치상 장애 유도)
dcgmi test --inject --gpuid 0 -f 111 -v 999

# 4. DCGM 자체 상태 확인
echo "Checking DCGM Health Status..."
sleep 5
dcgmi health -g $(dcgmi group -l | grep nma_test_group | awk '{print $2}') -c
dcgmi health -g 5 -c
```

#### 4. NodeCondition 변화 확인 ####
```
kubectl describe node ${NODE_NAME} | grep -A 15 Conditions
```
[결과]
```
Conditions:
  Type                       Status  LastHeartbeatTime                 LastTransitionTime                Reason                             Message
  ----                       ------  -----------------                 ------------------                ------                             -------
  MemoryPressure             False   Wed, 24 Dec 2025 03:04:59 +0000   Wed, 24 Dec 2025 02:27:47 +0000   KubeletHasSufficientMemory         kubelet has sufficient memory available
  DiskPressure               False   Wed, 24 Dec 2025 03:04:59 +0000   Wed, 24 Dec 2025 02:27:47 +0000   KubeletHasNoDiskPressure           kubelet has no disk pressure
  PIDPressure                False   Wed, 24 Dec 2025 03:04:59 +0000   Wed, 24 Dec 2025 02:27:47 +0000   KubeletHasSufficientPID            kubelet has sufficient PID available
  Ready                      True    Wed, 24 Dec 2025 03:04:59 +0000   Wed, 24 Dec 2025 02:32:20 +0000   KubeletReady                       kubelet is posting ready status
  ContainerRuntimeReady      True    Wed, 24 Dec 2025 02:32:26 +0000   Wed, 24 Dec 2025 02:32:26 +0000   ContainerRuntimeIsReady            Monitoring for the ContainerRuntime system is active
  StorageReady               True    Wed, 24 Dec 2025 02:32:26 +0000   Wed, 24 Dec 2025 02:32:26 +0000   DiskIsReady                        Monitoring for the Disk system is active
  NetworkingReady            True    Wed, 24 Dec 2025 02:32:26 +0000   Wed, 24 Dec 2025 02:32:26 +0000   NetworkingIsReady                  Monitoring for the Networking system is active
  KernelReady                True    Wed, 24 Dec 2025 02:32:26 +0000   Wed, 24 Dec 2025 02:32:26 +0000   KernelIsReady                      Monitoring for the Kernel system is active
  AcceleratedHardwareReady   True    Wed, 24 Dec 2025 02:32:26 +0000   Wed, 24 Dec 2025 02:32:26 +0000   NvidiaAcceleratedHardwareIsReady   Monitoring for the Nvidia AcceleratedHardware system is active
Addresses:
  InternalIP:   10.0.4.138
  InternalDNS:  ip-10-0-4-138.ap-northeast-2.compute.internal
  Hostname:     ip-10-0-4-138.ap-northeast-2.compute.internal
```
오류 주입후 AcceleratedHardwareReady 값이 False 변경되는지 확인하고, 수 분 내에 노드가 Cordon(스케줄링 중단) 상태가 되고 새로운 노드가 프로비저닝되는지 확인한다. 


## 레퍼런스 ##

* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/
* https://dev.to/aws-builders/node-health-monitoring-and-auto-repair-for-amazon-eks-3eja
* [Chaos Mesh](https://kmaster.tistory.com/12#google_vignette)
