2025년 기준, Kubernetes와 PyTorchJob 환경에서 특정 노드나 특정 GPU ID를 배제하는 방법은 다음과 같습니다.


### 1. 특정 노드 배제 (Node Exclusion) ###
가장 표준적인 방법은 nodeAffinity를 사용하는 것입니다. 특정 노드에 장애가 있거나 성능이 떨어질 때 해당 노드를 제외하고 스케줄링하도록 설정할 수 있습니다.
```
spec:
  pytorchReplicaSpecs:
    Worker:
      template:
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
                    - bad-node-01  # 배제하고 싶은 노드 이름
                    - bad-node-02
```

### 2. 특정 GPU ID 배제 (GPU Index Exclusion) ###
Kubernetes 수준에서 특정 번호의 GPU(예: 0번은 출력용이라 배제 등)만 콕 집어 제외하는 설정은 공식적으로 지원되지 않습니다. 대신 환경 변수를 통해 컨테이너 내부의 PyTorch가 인식하는 GPU를 제한해야 합니다.
CUDA_VISIBLE_DEVICES 환경 변수를 사용하면 컨테이너는 할당받은 GPU들 중 지정된 인덱스만 사용하게 됩니다.
```
spec:
  pytorchReplicaSpecs:
    Worker:
      template:
        spec:
          containers:
          - name: pytorch
            image: your-image
            env:
            - name: CUDA_VISIBLE_DEVICES
              value: "1,2,3" # 0번 GPU를 제외하고 1, 2, 3번만 사용
            resources:
              limits:
                nvidia.com: 3 # 할당량과 환경 변수 숫자를 맞춰야 함
```

### 3. 노드 오염(Taints)과 용인(Tolerations) 활용 ###
특정 노드가 상태가 불안정하여 모든 학습 작업에서 배제하고 싶다면, 해당 노드 자체에 Taint를 거는 것이 가장 깔끔합니다.
이렇게 설정하면 별도의 tolerations가 없는 PyTorchJob은 해당 노드에 절대 배정되지 않습니다.
```
kubectl taint nodes bad-node-01 hardware=unstable:NoSchedule
```

### 특정 노드의 GPU UUID와 시리얼 번호 조회 ###
특정 노드에 직접 접속하지 않고도, 해당 노드의 GPU 시리얼과 UUID를 가장 정확하게 확인할 수 있는 방법입니다. --format=csv 옵션을 사용하여 필요한 정보만 추출할 수 있습니다.
특정 GPU가 계속 에러를 발생시킨다면, 위에서 찾은 UUID를 기반으로 PyTorchJob 실행 시 특정 GPU를 배제하거나 노드 격리(Cordon)를 진행하는 것이 좋습니다.
```
kubectl run gpu-info-check --rm -it --restart=Never --image=nvidia/cuda:12.0-base-ubuntu22.04 --overrides='
{
  "spec": {
    "nodeName": "<확인할-노드-이름>",
    "containers": [{
      "name": "gpu-info",
      "image": "nvidia/cuda:12.0-base-ubuntu22.04",
      "args": ["nvidia-smi", "--query-gpu=index,name,uuid,serial", "--format=csv"],
      "resources": { "limits": { "nvidia.com": "1" } }
    }]
  }
}'
```

## 노드 격리 및 배제 ##

### 1. 특정 노드 격리 (Cordon & Drain) ####
노드 전체의 GPU 상태가 불안정하거나 점검이 필요할 때 사용합니다.
* Cordon (스케줄링 중단): 해당 노드에 새로운 Pod가 배치되지 않도록 막습니다.
```
kubectl cordon <노드-이름>
```
* Drain (기존 Pod 배출): 현재 실행 중인 PyTorchJob 워커들을 다른 노드로 강제 이동시킵니다.
```
kubectl drain <노드-이름> --ignore-daemonsets --delete-emptydir-data
```

### 2. PyTorchJob 내 특정 GPU 배제 (UUID 기반) ###
한 노드에 여러 개의 GPU(예: 8개)가 있는데, 그중 특정 ID(UUID)의 GPU만 고장이 났을 경우입니다. 2025년 최신 nvidia-device-plugin 환경에서는 환경 변수를 통해 이를 제어합니다.
PyTorchJob의 YAML 설정에 아래 환경 변수를 추가하세요:

```
spec:
  pytorchReplicaSpecs:
    Worker:
      template:
        spec:
          containers:
          - name: pytorch
            env:
            # 특정 UUID의 GPU만 사용하도록 명시 (나머지는 배제됨)
            # nvidia-smi -L 로 확인한 UUID 입력
            - name: NVIDIA_VISIBLE_DEVICES
              value: "GPU-8e3d6f21-...,GPU-f2a3b4c5-..." 
            # 또는 특정 GPU 인덱스만 제외하고 싶을 때 (예: 0번 제외)
            # - name: CUDA_VISIBLE_DEVICES
            #   value: "1,2,3"
```

### 3. 노드 선택기(NodeSelector)를 통한 배제 ###
특정 노드에 faulty-gpu=true와 같은 레이블을 붙여두고, PyTorchJob이 해당 레이블이 없는 노드만 찾아가게 설정할 수 있습니다. 

```
spec:
  pytorchReplicaSpecs:
    Worker:
      template:
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: gpu-status
                    operator: NotIn
                    values:
                    - faulty
```

