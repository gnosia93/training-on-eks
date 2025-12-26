큐브 플로우에서 제공해주는 TrainJob 오퍼레이터는 작업에 대한 복구 기능을 제공해 준다. 하드웨어 장애, 소프트웨어 설정 오류, GPU 장애, 트레이닝 코드 버그 등 여러가지 이유로 분산 트레이닝 잡이 실패했을때 이를 복구 하는 기능으로 문제가 발생한 Pod 를 다시 북구해 준다.   

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

## 복원력 설정 ##

큐브플루우의 TrainingJob 오퍼레이터는 기본적으로 6회 까지의 잡 재시작 기능을 제공하고 있다. 이 값을 더 크게 늘리기 위해서는 ClusterTrainingRuntime 의 maxRestarts 필드값을 수정해 줘야한다.
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


## 레퍼런스 ##

* https://github.com/kubeflow/trainer
* https://www.kubeflow.org/docs/components/trainer/operator-guides/migration/
* https://blog.kubeflow.org/trainer/intro/
