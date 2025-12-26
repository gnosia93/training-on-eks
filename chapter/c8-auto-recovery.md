NPD(Node Problem Detector) 같은 모니터링 에이전트가 없으면, 카펜터(Karpenter)는 "GPU가 물리적으로 고장 났다"는 사실을 스스로 알아낼 방법이 없습니다.
그 이유와 결과를 2025년 기준 시스템 동작 원리로 설명해 드립니다.

#### 1. 카펜터가 "모르는" 이유 ####
카펜터는 주로 쿠버네티스 API 서버의 정보를 보고 판단합니다.
카펜터의 시야: 노드가 Ready 상태인지, CPU/Memory 자원이 여유가 있는지, 인스턴스가 활성화(Running) 상태인지만 확인합니다.
GPU 장애의 특성: GPU가 물리적으로 고장 나거나 드라이버가 깨져도(Xid 에러 등), 리눅스 커널이나 CPU, 네트워크는 멀쩡한 경우가 많습니다. 따라서 쿠버네티스 입장에서는 노드가 여전히 Ready 상태로 보입니다.



## 버그 ##
* https://github.com/aws/containers-roadmap/issues/2555
  
현재 EKS 노드 모니터링 에이전트에서는 dcgm exporter 로 연결하지 못해서 노드의 AcceleratedHardwareReady 값이 False 로 변경되는 버그를 가지고 있어 GPU Auto Recovery 테스트는 불가능하다. 


## 1. Node Monitoring Agent 설치 ##
eks-node-monitoring-agent 애드온을 설치한다. 이 에이전트가 노드의 로그(/dev/kmsg)를 분석하여 장애를 감지하는 역할 한다.
Node Monitoring Agent는 GPU 의 상태를 확인하기 위해 DCGM exporter(nv-hostengine)와 unix domain 소켓으로 연결하여 관련 정보를 수집한다.
다른 애드온 들과는 달리 Pod identity 나 OIDC 와 연관된 Role 를 설정하지 않는다. 
```
export CLUSTER_NAME="training-on-eks"
eksctl create addon --cluster ${CLUSTER_NAME} --name eks-node-monitoring-agent --version latest
```
[결과]
```
2025-12-25 21:07:49 [ℹ]  Kubernetes version "1.34" in use by cluster "training-on-eks"
2025-12-25 21:07:49 [ℹ]  creating addon: eks-node-monitoring-agent
2025-12-25 21:08:11 [ℹ]  addon "eks-node-monitoring-agent" active
```
아래 명령어로 로그를 확인할 수 있다. 
```
kubectl logs -f -n kube-system -l app.kubernetes.io/name=eks-node-monitoring-agent
```
[결과]
```
{"level":"info","ts":"2025-12-24T01:53:06Z","msg":"reported node conditions","hostname":"ip-10-0-6-164.ap-northeast-2.compute.internal"}
{"level":"info","ts":"2025-12-24T01:58:06Z","msg":"reporting managed conditions","hostname":"ip-10-0-6-164.ap-northeast-2.compute.internal"}
{"level":"info","ts":"2025-12-24T01:58:06Z","msg":"Skipping MAC address policy check - not needed for this OS","hostname":"ip-10-0-6-164.ap-northeast-2.compute.internal","monitor":"networking"}
```

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
카펜터의 경우 매니지드 노드 그룹과 달리 추가적인 설정이 존재하지 않는다. 카펜터는 기본적으로 NotReady 상태인 노드를 감지하여 교체하기 때문이다. 

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

* 방법1 
```
export DCGM_POD=$(kubectl get pods -n dcgm -l app.kubernetes.io/name=dcgm-exporter -o jsonpath='{.items[0].metadata.name}')
echo ${DCGM_POD}
kubectl exec -n dcgm ${DCGM_POD} -- dcgmi test --inject --gpuid 0 -f 319 -v 4
```

* 방법2 
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

#### 4. Node Condition 확인 ####
```
kubectl describe node ${NODE_NAME} | grep -A 15 Conditions
```
[결과]
```
Conditions:
  Type                       Status  LastHeartbeatTime                 LastTransitionTime                Reason                             Message
  ----                       ------  -----------------                 ------------------                ------                             -------
  ...
  AcceleratedHardwareReady   True    Wed, 24 Dec 2025 02:32:26 +0000   Wed, 24 Dec 2025 02:32:26 +0000   NvidiaAcceleratedHardwareIsReady   Monitoring for the Nvidia AcceleratedHardware system is active

```
오류 주입후 AcceleratedHardwareReady 값이 False 변경되는지 확인하고, 수 분 내에 노드가 Cordon(스케줄링 중단) 상태가 되고 새로운 노드가 프로비저닝되는지 확인한다. 


## 레퍼런스 ##

* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/
* https://dev.to/aws-builders/node-health-monitoring-and-auto-repair-for-amazon-eks-3eja
* [Chaos Mesh](https://kmaster.tistory.com/12#google_vignette)
