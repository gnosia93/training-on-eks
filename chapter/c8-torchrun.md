## [트레이닝 오퍼레이터 설치](https://www.kubeflow.org/docs/components/trainer/operator-guides/installation/) ##

큐브 플로우 트레이닝 오퍼레이터와 런터임 설치한다. 
```
export VERSION=v2.1.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"

kubectl get pods -n kubeflow-system
```
[결과]
```

```



---
```
# 모든 노드(1번~5번)에서 동일하게 실행 (endpoint는 1번 노드 IP로 통일)
torchrun --nnodes=2 \
         --nproc_per_node=8 \
         --rdzv_id=job_1 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=10.0.0.1:2379 \
         train.py
```
를 쿠버네티스 환경에서 실행하는 경우 

```
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "multi-node-train"
spec:
  # [중요] Master 섹션 없이 Worker만 설정
  pytorchReplicaSpecs:
    Worker:
      replicas: 2  # --nnodes=2 에 해당
      template:
        spec:
          containers:
            - name: pytorch
              image: your-training-image:latest
              command:
                - torchrun                           # rdzv 관련 인자를 직접 쓰지 않아도 컨트롤러가 주입함
                - "--nproc_per_node=8"               # 노드당 GPU 8개
                - "train.py"
                - "--your-arg=value"
              resources:
                limits:
                  nvidia.com: 8                      # 실제 GPU 할당
  
  elasticPolicy:                                     # torchrun을 위한 정책 설정
    minReplicas: 2
    maxReplicas: 2
    rdzvBackend: c10d                                # 외부 etcd 없이 내부 통신 사용 (c10d 소켓 통신)
    maxRestarts: 3                                   # 노드 장애 시 재시도 횟수
```

* Master 섹션이 없는 이유: torchrun은 모든 노드를 대등한 워커로 취급하며, 랑데뷰 시스템을 통해 실행 시점에 Rank 0을 자동으로 선출합니다. Master 스펙을 별도로 정의하면 오히려 랑데뷰 주소가 꼬일 수 있습니다. 
* rdzv_endpoint 생략: PyTorchJob 컨트롤러가 각 포드에 PET_RDZV_ENDPOINT 환경 변수를 자동으로 넣어줍니다. 보통 첫 번째 워커 포드(worker-0)의 주소를 사용하도록 자동 설정됩니다.
* rdzv_id 생략: 컨트롤러가 해당 Job의 고유 UID를 ID로 자동 주입하여 다른 Job과 섞이지 않게 해줍니다.

#### 주입되는 환경 변수 (자동 설정됨) ####
YAML을 실행하면 쿠버네티스는 각 워커 포드에 다음과 같은 환경 변수를 자동으로 설정하여 torchrun이 이를 읽게 합니다.
* PET_NNODES: "5:5"
* PET_NPROC_PER_NODE: "8"
* PET_RDZV_BACKEND: "c10d"
* PET_RDZV_ENDPOINT: "multi-node-train-worker-0:2379"

#### 주의사항 ####
* 공유 저장소 마운트: 5개 노드가 모두 동일한 체크포인트 파일에 접근해야 하므로, spec.template.spec.volumes 설정에 NFS나 AWS FSx 같은 공유 파일 시스템을 마운트하는 설정을 반드시 추가해야 합니다.
* 리소스 할당: replicas: 5와 nvidia.com: 8을 설정하면 클러스터에 최소 40개의 GPU 여유 자원이 있어야 학습이 시작됩니다.

이 방식이 현재 쿠버네티스 환경에서 torchrun을 사용하는 표준(Best Practice)입니다.


---
#### 장애 발생 시 복구 프로세스 (Step-by-Step) ####
노드 1개가 죽었을 때, 일반적인 NCCL 훈련과 달리 torchrun은 다음과 같이 행동합니다.

장애 감지: 특정 Pod가 죽으면 NCCL 통신이 깨집니다. 이때 살아있는 나머지 Pod의 torchrun 프로세스가 이를 감지하고 자신의 로컬 프로세스들을 모두 종료(Terminate)시킵니다. (전체 작업은 잠시 멈춥니다.)
쿠버네티스 재스케줄링: 쿠버네티스의 Job 컨트롤러나 ReplicaSet이 죽은 Pod를 감지하고, 새로운 Pod를 자동으로 다시 생성합니다.
새로운 랑데부: 새로 뜬 Pod와 기존에 살아있던 Pod들이 다시 랑데부 서버에 모입니다.
World 재구성: 랑데부 서버는 "자, 다시 8명이 모였으니 새로 시작하자"라고 신호를 보냅니다. 이때 바뀐 IP 정보 등을 NCCL에 다시 전파하여 통신 그룹을 재형성(Re-init)합니다.
학습 재개: 개발자가 짠 코드 내의 load_checkpoint 로직에 의해 공유 스토리지에서 마지막 상태를 불러와 학습을 이어갑니다.









--------------------
## 토치런 랑데뷰 ##
torchrun (PyTorch Elastic) 환경에서 하나의 노드에 장애가 발생하면, 나머지 노드들은 무한정 기다리는 것이 아니라 현재 작업을 중단하고 새로운 랑데뷰(Rendezvous)를 통해 재구성을 시도합니다.
구체적인 동작 과정은 다음과 같습니다.

#### 1. 장애 감지 및 전송 (Stop) ####
한 노드가 다운되거나 프로세스가 종료되면, 연결되어 있던 다른 노드들은 통신 타임아웃을 통해 이를 감지합니다. 이 시점에서 모든 생존 노드의 학습 프로세스는 즉시 중단(Panic/Exit)됩니다.

#### 2. 재학습을 위한 재집결 (Rendezvous) ####
torchrun은 설정된 min_nodes와 max_nodes 범위 내에서 다시 랑데뷰를 시도합니다.
* 생존 노드들: 다시 랑데뷰 포인트에 모여 새로운 구성을 기다립니다.
* 장애 노드: 만약 쿠버네티스(Kubernetes) 같은 오케스트레이터가 노드를 복구시킨다면, 해당 노드도 다시 랑데뷰에 참여합니다.
* 대기 시간: --rdzv_timeout 설정값 동안 기다리며, min_nodes 조건이 충족되면 즉시 다음 단계로 넘어갑니다.

#### 3. 재시작 (Restart) ####
새로운 멤버 구성이 완료되면(예: 4개 노드 중 1개가 죽어 3개만 모인 경우), torchrun은 모든 노드의 프로세스를 새로운 RANK와 WORLD_SIZE로 다시 실행합니다.

#### 4. 주의사항: 체크포인트 로드 ####
torchrun은 노드 구성만 자동으로 다시 잡아줄 뿐, 학습 데이터의 상태를 자동으로 복구해주지는 않습니다.
장애 복구 후 이전 단계부터 이어서 학습하려면, 코드 내에 torch.save와 torch.load를 이용한 체크포인트 저장 및 로드 로직이 반드시 구현되어 있어야 합니다.
보통 load_state_dict를 통해 가장 최근에 저장된 모델 파일부터 다시 시작하도록 코드를 짭니다.

#### 요약 ####
노드 장애 시 나머지 노드들은 기다리기만 하는 것이 아니라, 다 같이 멈췄다가 가능한 노드들끼리 모여서 처음부터(또는 마지막 체크포인트부터) 다시 시작합니다.
이것이 python -m torch.distributed.launch와 같은 과거 방식(전체 종료 후 수동 재시작 필요)과 torchrun의 가장 큰 차이점입니다

## 재시도 ##
torchrun에서 노드 장애 시 재시도 횟수는 실행 명령 시 사용하는 --max_restarts 옵션으로 결정됩니다. PyTorch Elastic Documentation

#### 1. 기본 재시도 횟수 ####
기본값: 별도로 설정하지 않으면 0입니다. 즉, 한 번이라도 노드 장애나 프로세스 오류가 발생하면 재시도 없이 전체 작업이 종료됩니다.
설정 방법: torchrun --max_restarts=3 ...과 같이 실행 시 명시적으로 횟수를 지정해야 합니다.

#### 2. 무한 재시도 설정 ####
만약 노드 복구 속도가 느리거나, 쿠버네티스 환경에서 노드가 계속 교체되는 상황이라 무한히 재시도하고 싶다면 -1 또는 매우 큰 값을 설정할 수 있습니다.
예: torchrun --max_restarts=999 ... (사실상의 무한 재시도)

#### 3. 재시도 메커니즘의 특징 ####
카운트 기준: --max_restarts는 작업 전체(Job)가 실패로 간주되기 전까지 허용되는 재결합(Rendezvous) 및 재시작 총 횟수를 의미합니다.
부분 복구: 설정된 횟수 안에서라면, 일부 노드가 죽어도 min_nodes 조건만 만족되면 남은 노드들끼리 다시 랑데뷰하여 학습을 이어갑니다.
횟수 초과 시: 만약 재시도 횟수를 모두 소모하면 torchrun은 최종적으로 실패(Fail) 상태가 되며 종료됩니다.


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

## 장애 복구 시나리오 (Kubernetes + PyTorchJob) ##
* 장애 발생: 한 개의 워커 포드가 노드 문제로 종료됩니다.
* 랑데뷰 중단: 나머지 노드들의 torchrun 프로세스가 장애를 감지하고 중단됩니다.
* 포드 재생성: PyTorchJob 컨트롤러가 죽은 포드를 확인하고 새로운 포드를 띄웁니다.
* 재결합(Rendezvous): 모든 포드(신규 포드 포함)가 다시 랑데뷰 포인트에 모여 WORLD_SIZE를 확인합니다.
* 학습 재개: 코드가 마지막 체크포인트를 로드하여 학습을 이어갑니다. (공유 스토리지 필수)

```
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "elastic-train"
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 4  # 전체 노드 수
  elasticPolicy:
    minReplicas: 4 # min/max를 동일하게 설정하여 고정 규모 유지
    maxReplicas: 4
    rdzvBackend: c10d # Kubernetes 내부 서비스 이름을 rdzvEndpoint로 사용
    maxRestarts: 3
```

## worker 노드만 구성 ##
PyTorchJob에서 Master와 Worker를 구분하여 설정하는 방식은 과거의 mpirun이나 구형 분산 학습 방식의 유산이며, torchrun과 ElasticPolicy를 사용하는 현대적인 방식에서는 모든 노드를 Worker로만 구성하는 것이 권장됩니다.
그 이유는 다음과 같습니다.

#### 1. torchrun의 랑데뷰 시스템 때문 ####
torchrun 기반의 탄력적 학습(Elastic Training)에서는 특정 노드가 고정된 마스터 역할을 수행하지 않습니다. 대신, 랑데뷰 백엔드(c10d 또는 etcd)를 통해 어떤 노드가 'Rank 0'이 될지 동적으로 결정합니다.
Worker 5개로 구성 시: 5개의 포드가 모두 동일한 권한을 가지며, 그중 먼저 랑데뷰에 도달하거나 선정된 노드가 자동으로 Rank 0 역할을 수행합니다.
* 이점: 특정 '마스터 포드'가 죽었을 때 전체 작업이 불능 상태에 빠지는 것을 방지하고, 어떤 포드가 죽더라도 남은 포드들끼리 재결합이 가능합니다.

#### 2. PyTorchJob 설정 가이드 ####
Kubeflow 공식 문서에 따르면, elasticPolicy를 사용할 때는 다음과 같이 구성하는 것이 정석입니다.
* Master 섹션: 생략합니다.
* Worker 섹션: 전체 노드 수(replicas: 5)를 입력합니다.
* 결과: Kubernetes는 동일한 사양의 포드 5개를 생성하며, 이들은 torchrun을 통해 하나의 그룹으로 묶입니다.

```
spec:
  pytorchReplicaSpecs:
    Worker:              # Master 없이 Worker만 정의
      replicas: 5        # 총 5개의 노드 사용
      template:
        spec:
          containers:
            - name: pytorch
              image: your-training-image
              command: ["torchrun"] # torchrun이 내부적으로 Rank 할당
  elasticPolicy:
    minReplicas: 5
    maxReplicas: 5
    rdzvBackend: c10d     # Pod 간 통신을 위해 Kubernetes Service 활용
    maxRestarts: 3
```
* c10d (권장): 추가 인프라가 필요 없어 가장 가볍습니다. 포드가 재시작되어도 쿠버네티스 서비스 이름은 유지되므로 torchrun이 다시 랑데뷰하는 데 문제가 없습니다.
* etcd: 수백 개 이상의 노드를 사용하는 대규모 클러스터에서 랑데뷰의 안정성을 극한으로 높여야 할 때 사용합니다. 일반적인 5~10개 노드 규모에서는 c10d로도 충분합니다.

## 레퍼런스 ##

* https://github.com/kubeflow/trainer
* https://www.kubeflow.org/docs/components/trainer/operator-guides/migration/
