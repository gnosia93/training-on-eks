큐브 플로우에서 제공해주는 TrainJob 오퍼레이터(트레이닝 오퍼레이터 V2)는 작업에 대한 복구 기능을 기본적으로 제공해 준다. 하드웨어 장애, 소프트웨어 설정 오류, GPU 장애, 트레이닝 코드 버그 및 NCCL 오류 등 여러가지 이유로 분산 트레이닝 잡이 실패했을때 이를 복구하기 위해 문제가 발생한 Pod를 다시 실행해 준다.   

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

        containers:
          - name: node                                                # -name: node 은 상당히 중요한 설정값 / ClusterTrainingRuntime 에 있는 컨테이너 이름이 node 이다.
            volumeMounts:                                             # 이 값을 잘못 설정하는 경우 TrainJob 이 시작되지 않는다. 
              - mountPath: /dev/shm
                name: dshm
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "64Gi"               # shared memory 공간을 기본값(64Mi) 에서 64Gi 로 설정 / num_workers 와 연관됨.

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
          t5-fsdp.py --model_id="google-t5/t5-large" --epochs=10
    resourcesPerNode:
      limits:
        nvidia.com/gpu: "8"
      requests:
        nvidia.com/gpu: "8"
EOF
```
* google-t5/t5-large는 약 770M 파라미터 모델, epochs는 10회 수행 
* Placement Group (가용 영역 지정): 이와 관련된 필드 설정은 존재하지 않는다. 노드 라벨을 이용하여 비슷한 효과를 내게해야 한다. 
nodeSelector에 topology.kubernetes.io/zone을 명시하면, 분산 학습시 노드들이 동일한 AZ 내에 배치되어 NCCL 통신 레이턴시가 크게 줄어든다.
* CPU/메모리 자원 할당:
분산 학습(특히 FSDP)은 데이터 전처리나 체크포인트 저장시 일시적으로 많은 CPU와 메모리를 사용한다. 쿠버네티스 리소스 리미트를 주지 않으면 이러한 작업이 병목 없이 빠르게 처리된다.
* Kubeflow Training Operator가 분산 학습을 위해 쿠버네티스 헤드리스 서비스를 생성하면, 각 파드 내의 PyTorch Elastic Training(PET) 모듈이 이 서비스 주소를 참조하여 PET_MASTER_ADDR 및 PET_MASTER_PORT 환경 변수를 채워준다. 이를 통해 모든 랭크(Rank)들이 마스터 노드를 식별하고 랑데뷰(Rendezvous)를 수행한다 

#### 2. 잡 실행하기 ####
트레이닝 작업을 시작하고 로그를 확인한다. 
```
kubectl apply -f t5-large.yaml

kubectl get trainjob

kubectl get pods

kubectl logs -f <pod name> 
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
   GPU:.status.capacity['nvidia\.com/gpu'], \
   CAPACITY:.metadata.labels['karpenter\.sh/capacity-type']"
```
[결과]
```
NAME                                                INSTANCE       ARCH       OS                             GPU
ip-10-0-4-115.ap-northeast-2.compute.internal   c7g.2xlarge    arm64      Amazon Linux 2023.9.20251208   <none>
ip-10-0-4-210.ap-northeast-2.compute.internal   g6e.48xlarge   amd64      Amazon Linux 2023.9.20251208   8
ip-10-0-4-89.ap-northeast-2.compute.internal    g6e.48xlarge   amd64      Amazon Linux 2023.9.20251208   8
ip-10-0-6-164.ap-northeast-2.compute.internal   c6i.2xlarge    amd64      Amazon Linux 2023.9.20251208   <none>
```

## 복원력 설정 (재시도 횟수 설정) ##

큐브플루우의 TrainingJob 오퍼레이터는 기본적으로 6회 까지의 잡 재시작 기능을 제공하고 있다. 이 값을 더 크게 늘리기 위해서는 ClusterTrainingRuntime 의 maxRestarts 필드값을 수정해 줘야한다. 대규모 학습에서는 10~20 정도로 넉넉하게 설정하여, 밤사이에 노드에 문제가 발생하더라도 시스템이 작업을 좀 더 많이 자동 재시도 하도록 하는 것이 좋다.  AWS Spot 인스턴스를 쓴다면 노드 회수가 빈번할 수 있고, GPU 노드가 10대면 1대일 때보다 하드웨어 장애(GPU 에러 등)가 발생할 확률이 더 높기 때문에 기본값 6 으로는 부족하다.
```
kubectl edit clustertrainingruntime torch-distributed
```
에디터가 열리면 아래 구조를 찾아 failurePolicy를 추가한다.
```
spec:
  template:
    spec:
      # 여기에 추가 (replicatedJobs와 같은 레벨)
      failurePolicy:
        maxRestarts: 10
      replicatedJobs:
      - name: node
        # ...  
```


#### 잡 재시작 시나리오 ####
* TrainingJob 오퍼레이터가 장애를 감지하고 파드를 재시작한다. 
* torchrun 에 의해 랑데뷰 포인트가 만들어 진다.
* 모든 워커가 다시 모여서 그룹 구성 (처음부터 다시 시작).
* 훈련 코드에 체크포인트 로드 로직이 있다면 끊긴 지점부터 학습 재개, 없다면 0 에폭부터 다시 시작.


## 복원력 적용 샘플 ##
```
cat <<EOF > t5-large-resilient.yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: t5-large
spec:
  podTemplateOverrides:
    - targetJobs: ["node"]
      spec:
        nodeSelector:
          node.kubernetes.io/instance-type: g6e.48xlarge
          topology.kubernetes.io/zone: ap-northeast-2a
        containers:
          - name: node
            env:
              # [타임아웃] 노드가 많을수록 랑데뷰 시간을 넉넉히 (30분)
              - name: NCCL_ASYNC_ERROR_HANDLING
                value: "1"
              - name: NCCL_NETWORK_TIMEOUT
                value: "1800"
              # [EFA] AWS g6e 인스턴스 성능 최적화 (EFA 사용 시) - 명시적으로 EFA GPUDirect RDMA 활성화
              - name: FI_EFA_SET_DEFAULT_GDR
                value: "1"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: "64Gi"

  runtimeRef:
    name: torch-distributed

  trainer:
    numNodes: 10                               # 10개 노드(파드)로 확장
    numProcPerNode: 8                          # g6e.48xlarge는 GPU가 8개임
    image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
    
    command:
      - /bin/bash
      - -c
      - |
        git clone github.com /workspace/code
        cd /workspace/code/samples/fsdp
        pip install -r requirements.txt
        
        export MASTER_ADDR=\${PET_MASTER_ADDR}
        export MASTER_PORT=\${PET_MASTER_PORT:-29500}
        
        # [복원력 핵심] torchrun에 max-restarts 추가
        torchrun \
          --nnodes=10 \
          --nproc_per_node=8 \
          --max-restarts=3 \                   # torchrun 재시작 횟수 설정 (파드 재시작 없이)  
          --rdzv_id=t5_elastic_job \
          --rdzv_backend=c10d \
          --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
          t5-fsdp.py --model_id="google-t5/t5-large" --epochs=10
    resourcesPerNode:
      limits:
        nvidia.com: "8"
      requests:
        nvidia.com: "8"
EOF
```
* torchrun --max-restarts=3 : 10개 노드 중 한 곳에서 네트워크 장애가 발생하면, 파드를 새로 띄우지 않고 그 자리에서 즉시 프로세스를 다시 띄운다. 이 설정이 없으면 노드 하나만 문제가 발생하더라도 전체 Job이 죽게 된다. Job 이 죽으면 트레이닝 오퍼레이터가 Job의 스펙을 보고 새 파드(Container)를 다시 띄우게 되는데, 인프라 복구에 해당하는 것으로 완전히 새로운 Pod 를 스케줄링 하기 때문에 GPU 할당, 이미지 풀링(Pulling), 컨테이너 기동 등에 꽤 오랜 시간이 걸린다 (최소 수 분)
* NCCL_NETWORK_TIMEOUT="1800": 10대나 되는 노드들이 서로 통신 준비를 마칠 때까지 기다려주는 시간으로, 기본값이 짧으면 "동기화 실패"로 오인하고 학습이 종료된다. 
* NCCL_ASYNC_ERROR_HANDLING=1 : 분산 학습 중 발생하는 통신 에러를 실시간(비동기)으로 감지하고 프로세스에 알려주는 '경보 시스템' 으로 설정이다. 이 설정을 하지 않으면 한 노드에서 GPU 에러나 네트워크 장애가 발생하더라도 나머지 9개 노드는 그 사실을 모른 채 하염없이 기다린다(Timeout 될때 까지). 설정을 켰을 때 (=1) 한 노드에서 통신 에러가 나면, NCCL이 즉시 "에러 발생!" 메시지를 모든 노드에게 비동기적으로 뿌리게 된다. 모든 노드가 즉시 에러를 인지하고 프로세스를 종료한다. 이때 설정한 torchrun --max-restarts가 작동하여, 모든 노드가 동시에 깔끔하게 재시작(Rendezvous)할 수 있다.

### 각각의 역할 ###

#### 1. training operator ####
Training Operator는 쿠버네티스 클러스터 수준에서 인프라와 자원 관리를 담당하는데 사용자가 제출한 YAML 파일을 해석하여 실제 실행 환경을 구축하고 관리한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/resiliency-operator-role.png)

#### 2. torchrun ####
torchrun은 파드 내부에서 실행되는 애플리케이션 수준의 도구로, 실제 학습 프로세스의 실행과 동기화를 담당한다. PyTorch 코드가 분산 환경에서 원활하게 작동하도록 조율한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/resiliency-torchrun-role.png)

### 노드 10개 중 하나가 일시적으로 문제를 일으켰을 때 ###
* torchrun은 --max-restarts를 통해 빠르게 프로세스 재시도를 하여 대응.
* 만약 노드가 완전히 다운되면 Training Operator가 개입하여 새로운 파드를 생성.
* 새 파드가 뜨면 torchrun이 다시 랑데부를 통해 나머지 파드들과 학습을 재개.

### 장애 감지 시나리오 ###
* 프로세스 충돌: 노드 내 프로세스가 죽으면 에이전트가 즉시 인지하고 다른 노드들에 알립니다. (1초 내외)
* 노드 물리적 고장: 하트비트가 끊기면 랑데부 서버가 이를 감지하는 데 약 30초~1분이 소요됩니다.
* 네트워크 지연: 이때 설정하신 NCCL_NETWORK_TIMEOUT이 작동합니다. 30분(1800초) 동안 패킷이 안 오면 그때서야 "이건 죽은 거다"라고 판정하고 터뜨립니다.


## 레퍼런스 ##

* https://github.com/kubeflow/trainer
* https://www.kubeflow.org/docs/components/trainer/operator-guides/migration/
* https://blog.kubeflow.org/trainer/intro/
