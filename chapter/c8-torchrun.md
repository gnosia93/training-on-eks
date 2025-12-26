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


## 트레이닝 작업 실행 ##
TrainJob 오퍼레이터는 backoffLimit 라는 필드를 이용하여 작업 복구 매커니즘을 제공한다. 작업이 실패 했을때 다시 시작하는 기능으로, 이 예제에서는 3번까지 트레이닝 작업을 재 시작 하도록 설정 하였다.  
```
cat <<EOF > t5-large.yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: t5-large
spec:
  backoffLimit: 3                             # 작업 실패시 재시도 횟수
  restartPolicy: OnFailure

  podTemplateOverrides:
    - targetJobs:
        - name: trainer
      spec:
        nodeSelector:
          node.kubernetes.io/instance-type: g6e.48xlarge
          topology.kubernetes.io/zone: ap-northeast-2                # 특정 가용 영역(AZ) 내 배치를 강제하여 노드 간 통신 지연을 최소화 

  runtimeRef:
    name: torch-distributed                   # torch 분산 백엔드 사용 (관련 파이썬 패키지 묶음)

  trainer:
    numNodes: 2                            # 노드수 설정
    numProcPerNode: auto                   # 노드별 프로세스 갯수                                                                               
    image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
    command: 
      - git clone https://github.com/gnosia93/training-on-eks /workspace/code
      - cd /workspace/code/samples/fsdp
      - echo "working directory: "$(pwd)
      - pip install -r requirements.txt
      - torchrun --nproc_per_node 8 --rdzv_id=elastic-job --rdzv_backend=c10d t5-fsdp.py
    resourcesPerNode:
      limits:
        nvidia.com: "8"
      requests:
        nvidia.com: "8"
EOF
```
트레이닝 작업을 시작하고 로그를 확인한다. 
```
kubectl apply -f t5-large.yaml

kubectl logs -f -l trainjob-name=t5-large
```

* Placement Group (가용 영역 지정):
nodeSelector에 topology.kubernetes.io/zone을 명시하면, 분산 학습에 참여하는 노드들이 동일한 데이터 센터 내에 배치되어 NCCL 통신 레이턴시가 크게 줄어듭니다.
* Scheduling Policy (Gang Scheduling):
schedulingPolicy를 사용하면 2개의 노드가 동시에 할당될 때만 학습을 시작합니다. 이는 하나는 확보되고 하나는 대기 상태일 때 발생하는 자원 낭비와 통신 비효율을 방지합니다.

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
