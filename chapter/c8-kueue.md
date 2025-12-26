## 갱 스케줄러의 이해 ##

쿠버네티스는 Pod 단위로 작업을 스케줄링 한다. 2대의 노드가 필요한 분산 학습에서 1대의 자원만 확보했다면, 확보된 노드에 먼저 Pod를 스케줄링하고 노드 부족으로 인해 스케줄링 되지 못한 Pod는 자원이 확보될떄 까지 무기한 대기하게 된다. 
하지만 이는 단지 쿠버네티스 스케줄러만의 입장이고, Pod가 실행하는 NCCL 라이브러리는 랑데뷰를 무기한 대기하지 않고 어느 정도의 시간이 지난후 타임아웃과 함께 해당 프로세스는 종료 시키게 된다. (랑데뷰 타임아웃 - 기본 10분 대기) 

이처럼 분산 트레닝 작업시 필요한 전체 자원을 확보하지 못한 상태에서 잡이 실행되는 경우 예기치 못한 데드락 또는 오랜 시간동안의 대기 및 재시작 과정을 거친후 전체 작업이 실패할 수 위험이 존재하게 된다.
분산 환경에서 이러한 문제를 사전에 방지하기 위해서 All or Nothing 방식의 자원 할당 매커니즘이 필요한데 이를 가능하게 하는 것이 바로 갱 스케줄러이다. 

* Kueue (현재 Kubeflow 표준 권장): 2025년 기준 Kubeflow와 가장 긴밀하게 통합되는 작업 큐 관리자입니다.
TrainJob의 spec.schedulingPolicy.queue 필드는 주로 Kueue의 로컬 큐 이름을 참조합니다.

* Volcano:
전통적인 고성능 컴퓨팅(HPC) 및 배치 작업용 스케줄러입니다.
spec.schedulingPolicy.schedulerName: volcano 형식으로 지정하여 사용합니다.


## Kueue ##
Kueue 자체는 '언제(When)' 작업을 실행할지를 결정하는 쿼터 관리 시스템이며, '어디서(Where)' 실행할지를 결정하는 노드/GPU 레벨의 세부 스케줄링은 쿠버네티스 기본 스케줄러(kube-scheduler) 및 노드 설정과 연동하여 처리합니다
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/Kueue-Arch.svg)

### 1. 설치 ###
```
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.15.2 \
  --namespace  kueue-system \
  --create-namespace \
  --wait --timeout 300s
```
* 삭제
```
helm uninstall kueue --namespace kueue-system 
```

### 2. Kueue가 설정 ###
PyTorchJob을 실행하기 전에 Kueue가 해당 작업을 인식하고 리소스를 할당할 수 있도록 ResourceFlavor, ClusterQueue, 그리고 LocalQueue 세 가지 핵심 리소스가 설정되어 있어야 합니다

#### 1. 리소스 플레이버 정의 ####
Kueue에서 리소스 플레이버는 클러스터 내 노드들의 다양한 물리적/논리적 특성(사양)을 정의하는 API 객체로, 단순히 CPU나 메모리 양을 의미하는 것이 아니라 특정 노드 그룹이 가진 고유한 성격(예: GPU 모델, 인스턴스 유형, 가용성 정책 등)을 구분하는 역할을 한다. 클러스터에 존재하는 실제 리소스(여기서는 GPU)의 종류와 레이블을 정의하면 된다. 
```
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: flavor-gpu-nvidia
spec:
  nodeSelector:
    nodeType: "nvidia"                         # 카펜터가 생성한 gpu 노드풀의 레이블 
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: flavor-gpu-nvidia-efa
spec:
  nodeSelector:
    nodeType: "nvidia-efa"                    # 카펜터가 생성한 gpu-efa 노드풀의 레이블
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

#### 2. 클러스터큐 정의 (cluster-queue.yaml) #### 
클러스터 전체의 리소스 할당량과 정책을 정의한다. cluster-queue-gpu 라는 이름으로 GPU를 최대 100 장까지 할당 받을 수 있도록 설정한다. 
```
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: cluster-queue-gpu
spec:
  resourceGroups:
  - flavors:
    - flavor-gpu-nvidia
    - flavor-gpu-nvidia-efa
    resources:
    - name: "nvidia.com/gpu"              # 실제 리소스 명칭 / 카펜터에서 관리되고 있는 리소스 명칭과 동일해야 한다. 
      nominalQuota: 100                   # 전체 GPU 쿼타 설정
```

#### 3. 로컬큐 정의 (local-queue.yaml) ####
특정 네임스페이스(team-a)내 사용자들이 작업을 제출하는 통로를 정의하는 것으로 로컬큐는 클러스터큐를 참조한다. 이 예제에서는 default 네임스페이스에 default-queue 라는 로컬큐를 만들었다.
team-a 라는 네임스페이스를 만들게되면 해당 네임스페이스안에 로컬큐를 별도로 만들어야 한다. 
```
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: default-queue
  namespace: default
spec:
  # 위에서 정의한 ClusterQueue 이름
  clusterQueue: cluster-queue-gpu
```

## 사용 방법 ##
```
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: t5-large
  namespace: default
  labels:
    kueue.x-k8s.io/queue-name: default-queue                 # 이전에 정의한 로컬큐 이름
spec:
    # ... (생략)
```
#### Kueue에서의 작동 방식 ####

Kueue는 TrainJob 이 요구하는 리소스 합계를 계산하고(여기서는 gpu 갯수), cluster-queue-gpu 리소스 풀에 리소스가 남아있는지 확인한다.
* 리소스가 모두 있으면: 모든 파드를 동시에 실행(Admit) 상태로 변경하고
* 리소스가 하나라도 부족하면: 어떤 파드도 생성되지 않도록 큐에서 대기시킨다.
이 방식을 통해 일부 파드만 먼저 실행되어 자원을 점유한 채 나머지를 기다리는 데드락(Deadlock) 현상을 방지할 수 있다.

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/operator-guides/job-scheduling/
* https://kueue.sigs.k8s.io/docs/overview/
* https://thenewstack.io/kueue-can-now-schedule-kubernetes-batch-jobs-across-clusters/
