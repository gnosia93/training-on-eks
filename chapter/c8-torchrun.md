## [트레이닝 오퍼레이터(V2) 설치](https://www.kubeflow.org/docs/components/trainer/operator-guides/installation/) ##

V2 버전의 큐브 플로우 트레이닝 오퍼레이터와 런터임 설치한다. 
```
sudo dnf install git -y

export VERSION=v2.1.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"

kubectl get pods -n kubeflow-system
kubectl get clustertrainingruntimes

```
[결과]
```
NAME                                                   READY   STATUS    RESTARTS   AGE
jobset-controller-manager-58555b47c7-ltrck             1/1     Running   0          2m55s
kubeflow-trainer-controller-manager-5b7b978fbf-r24kr   1/1     Running   0          2m55s

NAME                     AGE
deepspeed-distributed    114s
mlx-distributed          114s
torch-distributed        114s
torchtune-llama3.2-1b    114s
torchtune-llama3.2-3b    114s
torchtune-qwen2.5-1.5b   114s
```

## CRD 조회 ##
```
kubectl explain trainjob.spec
kubectl explain trainjob.spec.podTemplateOverrides.spec
```

## runtme 조회 ##
```
kubectl get clustertrainingruntime torch-distributed -o yaml
```
[결과]
```
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  creationTimestamp: "2025-12-25T17:37:32Z"
  generation: 1
  labels:
    trainer.kubeflow.org/framework: torch
  name: torch-distributed
  resourceVersion: "1310309"
  uid: 2067ff23-511e-4b9c-b37e-b4d873f43c85
spec:
  mlPolicy:
    numNodes: 1
    torch:
      numProcPerNode: auto
  template:
    spec:
      replicatedJobs:
      - groupName: default
        name: node
        replicas: 1
        template:
          metadata:
            labels:
              trainer.kubeflow.org/trainjob-ancestor-step: trainer
          spec:
            template:
              spec:
                containers:
                - image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
                  name: node
```
```
kubectl explain ClusterTrainingRuntime.spec.template.spec.failurePolicy.maxRestarts
```
```
GROUP:      trainer.kubeflow.org
KIND:       ClusterTrainingRuntime
VERSION:    v1alpha1

FIELD: maxRestarts <integer>


DESCRIPTION:
    MaxRestarts defines the limit on the number of JobSet restarts.
    A restart is achieved by recreating all active child jobs.
```
### 재시도 횟수 수정 ###
#### 1. ClusterTrainingRuntime 직접 수정 (필수) ####
TrainJob 파일에서 failurePolicy를 빼고, 대신 런타임 자체를 수정해야 합니다.
```
kubectl edit clustertrainingruntime torch-distributed
```

에디터가 열리면 아래 구조를 찾아 failurePolicy를 추가합니다.
```
spec:
  template:
    spec:
      # 여기에 추가 (replicatedJobs와 같은 레벨)
      failurePolicy:
        maxRestarts: 5
      replicatedJobs:
      - name: node
        # ...  
```
* failurePolicy.maxRestarts를 써야 하는 이유
  * 전체 재시작 (Clean Slate): maxRestarts는 문제가 발생하면 관련된 모든 파드(전체 워커들)를 한꺼번에 삭제하고 새로 띄웁니다.
  * 랑데뷰 초기화: 모든 노드가 동시에 새로 뜨기 때문에 랑데뷰 포인트에서 다시 깔끔하게 모여 학습을 재개할 수 있습니다. 이것이 분산 학습에서 훨씬 안정적인 복구 방식입니다.


## 트레이닝 작업 실행 ##

#### 1. TrainJob 만들기 #### 
```
cat <<EOF > t5-large.yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: t5-large
spec:
  podTemplateOverrides:
    - targetJobs:
        - name: node                                                  # ClusterTrainingRuntime 에 있는 runtime job template
      spec:
        nodeSelector:
          node.kubernetes.io/instance-type: g6e.48xlarge              # https://instances.vantage.sh/aws/ec2/g6e.48xlarge?currency=USD
          topology.kubernetes.io/zone: ap-northeast-2a                # AZ 설정, 노드 간 통신 지연을 최소화 

  runtimeRef:
    name: torch-distributed                   # torch 분산 백엔드 사용 (관련 파이썬 패키지 묶음)

  trainer:
    numNodes: 2                               # 노드수 설정
    numProcPerNode: auto                      # 노드별 프로세스 갯수                                                                               
    image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0

    # 랑데뷰 포인트를 명시적으로 기술해 준다(rdzv_id, rdzv_backend, rdzv_endpoint)
    # --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} 에서 \$ 함으로써 쉘이 해당 변수를 해석하지 않도록 함.
    # ${MASTER_ADDR} 와 ${MASTER_PORT} 환경변수는 TrainJob 오퍼레이터가 잡 실행시 채워주는 값이다.  
    command:
      - /bin/bash
      - -c
      - |
        git clone https://github.com/gnosia93/training-on-eks /workspace/code
        cd /workspace/code/samples/fsdp
        pip install -r requirements.txt
        echo "=== Launching Distributed Training ==="
        export MASTER_ADDR=\${PET_MASTER_ADDR}
        export MASTER_PORT=\${PET_MASTER_PORT:-29500}
        echo "Master Address: \${MASTER_ADDR}"
        echo "Master Port: \${MASTER_PORT}"
        echo "=================================="
        torchrun \
          --nproc_per_node=8 \
          --rdzv_id=elastic-job \
          --rdzv_backend=c10d \
          --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
          t5-fsdp.py
    resourcesPerNode:
      limits:
        nvidia.com/gpu: "8"
      requests:
        nvidia.com/gpu: "8"
EOF
```
* Placement Group (가용 영역 지정):
nodeSelector에 topology.kubernetes.io/zone을 명시하면, 분산 학습시 노드들이 동일한 AZ 내에 배치되어 NCCL 통신 레이턴시가 크게 줄어든다.

#### 2. 잡 실행하기 ####
트레이닝 작업을 시작하고 로그를 확인한다. 
```
kubectl apply -f t5-large.yaml

kubectl get trainjob

kubectl get pods

kubectl logs -f -l trainjob-name=t5-large
```
* Job 삭제
```
kubectl delete trainjob t5-large
``` 


#### 3. 노드 리스트 출력하기 ####
본 워크삽에서는 카펜터를 이용하여 GPU 노드를 프로비저닝 하므로, 트레이닝 잡을 실행 후 GPU 노드가 프로비저닝 될때 까지 1분 이상의 시간이 소요된다. 아래 명령어는 쿠버네티스 클러스터에 조인된 노드 정보를 출력하는 명령어이다.
```
kubectl get nodes -o custom-columns="NAME:.metadata.name, \
   INSTANCE:.metadata.labels['node\.kubernetes\.io/instance-type'], \
   ARCH:.status.nodeInfo.architecture, \
   OS:.status.nodeInfo.osImage, \
   GPU:.status.capacity['nvidia\.com/gpu']"
```
[결과]
```
NAME                                                INSTANCE       ARCH       OS                             GPU
ip-10-0-4-115.ap-northeast-2.compute.internal   c7g.2xlarge    arm64      Amazon Linux 2023.9.20251208   <none>
ip-10-0-4-210.ap-northeast-2.compute.internal   g6e.48xlarge   amd64      Amazon Linux 2023.9.20251208   8
ip-10-0-4-89.ap-northeast-2.compute.internal    g6e.48xlarge   amd64      Amazon Linux 2023.9.20251208   8
ip-10-0-6-164.ap-northeast-2.compute.internal   c6i.2xlarge    amd64      Amazon Linux 2023.9.20251208   <none>
```

## 장애 발생 시 복구 프로세스 ##
노드 1개가 죽었을 때, 일반적인 NCCL 훈련과 달리 torchrun은 다음과 같이 행동합니다.

* 장애 감지: 특정 Pod가 죽으면 NCCL 통신이 깨집니다. 이때 살아있는 나머지 Pod의 torchrun 프로세스가 이를 감지하고 자신의 로컬 프로세스들을 모두 종료(Terminate)시킵니다. (전체 작업은 잠시 멈춥니다.)
* 쿠버네티스 재스케줄링: 쿠버네티스의 Job 컨트롤러나 ReplicaSet이 죽은 Pod를 감지하고, 새로운 Pod를 자동으로 다시 생성합니다.
* 새로운 랑데부: 새로 뜬 Pod와 기존에 살아있던 Pod들이 다시 랑데부 서버에 모입니다.
* World 재구성: 랑데부 서버는 "자, 다시 8명이 모였으니 새로 시작하자"라고 신호를 보냅니다. 이때 바뀐 IP 정보 등을 NCCL에 다시 전파하여 통신 그룹을 재형성(Re-init)합니다.
* 학습 재개: 개발자가 짠 코드 내의 load_checkpoint 로직에 의해 공유 스토리지에서 마지막 상태를 불러와 학습을 이어갑니다.

#### 요약: 마스터 파드 재시작 시 시나리오 ####
* Operator가 파드 재살림 (IP가 바뀌어도 서비스 이름으로 연결 유지).
* PyTorch Elastic이 랑데뷰 광장을 새로 개설.
* 모든 워커가 다시 모여서 그룹 구성 (처음부터 다시 시작).
* (중요) 코드에 체크포인트 로드 로직이 있다면 끊긴 지점부터 학습 재개, 없다면 0 에폭부터 다시 시작.

## 랑데뷰 포인트 ##
* c10d (권장): 추가 인프라가 필요 없어 가장 가볍습니다. 포드가 재시작되어도 쿠버네티스 서비스 이름은 유지되므로 torchrun이 다시 랑데뷰하는 데 문제가 없습니다.
* etcd: 수백 개 이상의 노드를 사용하는 대규모 클러스터에서 랑데뷰의 안정성을 극한으로 높여야 할 때 사용합니다. 일반적인 5~10개 노드 규모에서는 c10d로도 충분합니다.





--------------------

## 체크포인트 ##
모든 노드가 최신 체크포인트 파일에 접근할 수 있어야 합니다. 이를 위해 현업에서는 크게 두 가지 방법을 사용합니다.
#### 1. 공유 스토리지 사용 (가장 권장됨) ####
모든 노드가 NFS, AWS FSx, Google Cloud Filestore와 같은 공유 네트워크 스토리지를 동일한 경로에 마운트하는 방식입니다.
* 장점: 특정 노드가 완전히 사라져도 데이터가 안전하며, 모든 노드가 같은 경로(/mnt/nfs/checkpoint.pt)를 바라보기만 하면 됩니다.
* 방식: 0번 마스터 노드가 체크포인트를 저장하면 나머지 노드들이 재시작 시 해당 파일을 읽어옵니다.

#### 2. 로컬 스토리지 + 복제 ####
각 노드의 로컬 디스크(SSD)에 저장하는 방식입니다.
* 단점: 노드 자체가 물리적으로 고장 나면 해당 노드에 있던 체크포인트는 유실됩니다.
* 방식: 이를 해결하려면 학습 중 주기적으로 체크포인트를 S3 같은 클라우드 스토리지로 업로드하거나, 모든 노드가 각자 자기 디스크에 동일한 복사본을 저장하도록 설계해야 합니다.

#### 💡 torchrun 재시작 시 코드 구현 핵심 ####
torchrun은 프로세스를 다시 띄워줄 뿐, 체크포인트를 불러오는 코드는 직접 작성해야 합니다. 보통 다음과 같은 로직을 사용합니다.
```
import torch
import os

def main():
    # 1. 체크포인트 경로 설정 (공유 스토리지 권장)
    ckpt_path = "/shared/storage/model_latest.pt"

    # 2. 만약 기존 체크포인트가 있다면 로드
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # 3. 학습 루프 중 주기적 저장 (Rank 0번만 저장)
    if dist.get_rank() == 0:
        torch.save({...}, ckpt_path)
```
* 공유 저장소(NFS 등)를 쓰는 것이 가장 안전하고 편리합니다. 
* 노드가 재시작될 때 자동으로 ckpt_path를 확인하여 load_state_dict를 수행하는 로직이 코드에 포함되어야 합니다.
* 만약 공유 저장소가 없다면, 노드 장애 시 해당 노드에 있던 데이터는 못 쓰게 되므로 외부 클라우드 저장소(S3 등)에 백업하는 절차가 필요합니다.


## 레퍼런스 ##

* https://github.com/kubeflow/trainer
* https://www.kubeflow.org/docs/components/trainer/operator-guides/migration/
* https://blog.kubeflow.org/trainer/intro/
