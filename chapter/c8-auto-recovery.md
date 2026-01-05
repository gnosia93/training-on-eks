## [NPD(Node Problem Detector)](https://github.com/kubernetes/node-problem-detector) ##

2026년 대규모 분산 학습 인프라에서 p5.48xlarge나 g6e.48xlarge와 같은 멀티 GPU 인스턴스를 운영할 때, 단 하나의 GPU 하드웨어 결함만으로도 전체 작업이 중단되는 리스크를 방지하기 위해 Node Problem Detector(NPD) 도입은 필수적이다. 카펜터(Karpenter)는 인스턴스의 프로비저닝 상태는 파악하지만 GPU 내부의 물리적 장애를 스스로 인지할 수 없는데, 이때 NPD가 커널 로그(dmesg)와 시스템 로그(journald)를 실시간 모니터링하여 NVIDIA XID 에러나 NVLink 통신 장애와 같은 하드웨어 인터럽트 신호를 쿠버네티스가 이해할 수 있는 'Node Condition'으로 변환해 주는 인터페이스 역할을 수행하기 때문이다. NPD는 시스템 로그를 분석하는 것이 고유 기능이므로 별도의 복잡한 플러그인이나 바이너리 설치 없이 NVIDIA GPU 전용 로그 패턴이 정의된 ConfigMap 설정만 연결해주면 즉시 하드웨어 결함을 식별할 수 있다. 결과적으로 NPD가 장애 신호를 감지하여 노드 상태를 업데이트하면, 카펜터가 해당 노드를 불건전(Unhealthy) 상태로 판단해 즉시 폐기하고 신규 물리 서버로 교체하는 자가 치유(Self-healing) 아키텍처를 완성함으로써 대규모 연산 작업의 가용성을 극대화한다.


### 1. NPD 가 식별하는 장애 유형 ###

#### 1-1. 하드웨어 및 시스템 장애 (System Log Monitor) ####
커널 로그(dmesg, journald)를 실시간으로 감시하여 하드웨어 차원의 심각한 결함을 찾아냅니다.
* CPU/Memory: CPU 스택 정지(Stuck), 메모리 ECC 에러(데이터 손상), Read-only 파일시스템 전환.
* 디스크: 디스크 응답 없음, 파일 시스템 손상.
* GPU
  * 하드웨어 결함: GPU 하드웨어 자체의 오류 또는 고장
  * 드라이버 문제: GPU 드라이버 응답 없음(Hang) 또는 드라이버 로드 실패
  * 통신 오류: NVLink 연결 및 통신 오류 또는 XID Critical Error (NVIDIA 기준) 
  * 기타 시스템 오류: GPU 온도 이상 또는 메모리 오류(ECC Error) 등 

#### 1-2. 커스텀 장애 식별 (Custom Plugin Monitor) ####
사용자가 정의한 스크립트를 주기적으로 실행하여 특정 리소스의 상태를 체크합니다. GPU 장애 감지가 이 영역에 해당합니다.
* GPU 상태: nvidia-smi 응답 여부, 드라이버 좀비 프로세스 확인.
* 네트워크: 특정 게이트웨이 핑(Ping) 테스트, DNS 확인 실패.
* 런타임: Docker/Containerd 데몬 응답 지연 [2].

#### 1-3. 일시적 이벤트 보고 (Temporary Events) ####
노드 상태(Condition)를 영구적으로 바꾸지는 않지만, 장애가 발생했던 기록을 이벤트로 남깁니다.
* OOM Kill: 특정 프로세스가 메모리 부족으로 강제 종료된 기록.
* 프로세스 중단: 주요 시스템 서비스의 일시적인 재시작 기록.

### 2. 자동 복구 작동 원리 (NPD + Karpenter) ###
* 장애 감지: NPD가 GPU 장애를 발견하고 노드 Condition에 GPUProblem=True를 기록합니다.
* 노드 오염(Tainting): NPD의 설정이나 별도의 컨트롤러(예: Node Problem Detector의 NodeProblemHosts)가 해당 상태를 기반으로 노드에 NoSchedule 또는 NoExecute Taint(왜곡)를 추가합니다.
* 카펜터의 개입: 카펜터는 노드가 더 이상 정상적으로 파드를 수용할 수 없거나(Tainted), 정의된 건강 기준에서 벗어났다고 판단하면 해당 노드를 불건전(Unhealthy) 노드로 간주합니다.
* 자동 교체: 카펜터는 장애 노드를 비우고(Drain) 제거한 뒤, 즉시 새로운 정상 GPU 노드를 프로비저닝(Provisioning)하여 교체합니다.

#### 카펜터의 한계 ####
카펜터는 주로 쿠버네티스 API 서버의 정보를 보고 판단한다. 카펜터의 시야는 노드가 Ready 상태인지, CPU/Memory 자원이 여유가 있는지, 인스턴스가 활성화(Running) 상태인지만 확인하게 된다. 
GPU가 물리적으로 고장 나거나 드라이버가 깨져도(Xid 에러 등), 리눅스 커널이나 CPU, 네트워크는 멀쩡한 경우가 많다. 따라서 쿠버네티스 입장에서는 노드가 여전히 Ready 상태로 보이게 된다.
NPD가 없어도 카펜터가 노드를 교체하는 경우가 한가지 있는데, 장애가 너무 심각해서 노드 자체가 완전히 멈추거나(Kernel Panic), AWS 상태 검사(Status Check)가 실패하여 노드가 NotReady 상태가 될 때 뿐이다. 하지만 대부분의 GPU 에러는 노드 자체는 살려두고 GPU만 먹통으로 만들기 때문에, NPD 없이 카펜터만으로는 대응이 불가능하게 된다.

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/npd.png)


### 3. NPD 설치하기 ###

```
cat <<EOF > npd-values.yaml
image:
  repository: registry.k8s.io/node-problem-detector/node-problem-detector
  tag: v1.35.1
  pullPolicy: IfNotPresent

settings:
  log_monitors:
    # 커널 로그에서 XID 등 GPU 에러를 감시하는 기본 설정 파일 경로 (기본적으로 해당 경로에 제공됨) 
    - /config/kernel-monitor.json
    # NVIDIA 툴킷 및 드라이버 로그를 상세 감시하는 설정
    - /custom-config/nvidia-toolkit-monitor.json
EOF
```

```
kubectl create configmap npd-node-problem-detector-custom-config \
  --namespace kube-system \
  --from-literal=nvidia-toolkit-monitor.json='{
    "plugin": "journald",
    "pluginConfig": {
        "source": "kernel"
    },
    "logPath": "/var/log/journal",
    "lookback": "5m",
    "bufferSize": 10,
    "source": "gpu-monitor",
    "conditions": [
        {
            "type": "GPUProblem",
            "reason": "GPUHealthy",
            "message": "GPU is operating normally"
        }
    ],
    "rules": [
        {
            "type": "permanent",
            "condition": "GPUProblem",
            "reason": "GPUHardwareError",
            "pattern": "NVRM: Xid.*"
        },
        {
            "type": "permanent",
            "condition": "GPUProblem",
            "reason": "NVLinkError",
            "pattern": ".*NVLink Error.*"
        }
    ]
}' --dry-run=client -o yaml | kubectl apply -f -
```


```
helm repo add deliveryhero https://charts.deliveryhero.io/
helm repo update

helm install npd deliveryhero/node-problem-detector \
  -f npd-values.yaml --namespace kube-system
```


### 참고 - nvidia-validator 설치 ###
```
# 1. NVIDIA Helm 저장소 추가
helm repo add nvidia helm.ngc.nvidia.com
helm repo update

# 2. Validator만 활성화하여 설치
helm install nvidia-validator nvidia/gpu-operator \
  -n gpu-operator --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set devicePlugin.enabled=false \
  --set dcgmExporter.enabled=false \
  --set gfd.enabled=false \
  --set operator.enabled=false \
  --set validator.enabled=true
```

----
## AWS NMA ##
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
