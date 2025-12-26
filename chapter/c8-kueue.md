***이 챕터는 갱 스케줄러의 개념과 구현체인 Kueue 설정에 대해서 다룬다. 하지만 GPU 리소스 부족으로 실제 워크샵에서 갱 스케줄링을 테스트 하기에는 어려움이 있다.***     

## 갱 스케줄러의 이해 ##

쿠버네티스는 Pod 단위로 작업을 스케줄링 한다. 2대의 노드가 필요한 분산 학습에서 1대의 자원만 확보했다면, 확보된 노드에 먼저 Pod를 스케줄링하고 노드 부족으로 인해 스케줄링 되지 못한 Pod는 자원이 확보될떄 까지 무기한 대기하게 된다. 하지만 이는 단지 쿠버네티스 스케줄러만의 입장이고, Pod가 실행하는 NCCL 라이브러리는 랑데뷰를 무기한 대기하지 않고 어느 정도의 시간이 지난후 타임아웃과 함께 해당 프로세스는 종료 시키게 된다. (랑데뷰 타임아웃 - 기본 10분 대기) 

이처럼 분산 훈련 작업시 필요한 전체 자원을 확보하지 못한 상태에서 잡이 실행되는 경우, 예기치 못한 데드락 또는 오랜 시간동안의 대기 및 재시작 과정을 거친 후 전체 작업이 실패할 수 있는 위험이 존재한다.
분산 환경에서 이러한 문제를 사전에 방지하기 위해서 All or Nothing 방식의 자원 할당 매커니즘이 필요한데 이를 가능하게 하는 것이 바로 갱 스케줄러이다. 

* Kueue (현재 Kubeflow 표준 권장): 2025년 기준 Kubeflow와 가장 긴밀하게 통합되는 작업 큐 관리자.
* Volcano: 전통적인 고성능 컴퓨팅(HPC) 및 배치 작업용 스케줄러.


## Kueue ##
Kueue 자체는 '언제(When)' 작업을 실행할 지를 결정하는 쿼터 관리 시스템이다. '어디서(Where)' 실행할 지를 결정하는 노드/GPU 레벨의 세부 스케줄링은 쿠버네티스 기본 스케줄러(kube-scheduler) 또는 카펜터가 처리하게 된다. 
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


## 카펜터 통합 전략 ##
Kueue가 할당량(Quota) 관점에서는 갱 스케줄링을 보장하지만, 실제 노드 프로비저닝(Karpenter)과 파드 배치(Kube-scheduler) 단계에서 일부 파드만 먼저 생성되어 '노드 대기' 상태에 빠지는 현상이 종종 발생한다.
이 문제를 해결하고 Karpenter 환경에서 완전한 갱 스케줄링(All-or-Nothing)을 구현하기 위한 핵심 전략은 다음과 같다.

#### 1. Kueue의 'Admit' 메커니즘 활용 ####
Kueue는 리소스 풀에 자리가 생기기 전까지는 TrainJob 아래의 실제 Pod들을 생성조차 하지 못하게(Suspended) 막아둡니다.
Kueue가 승인(Admit)을 내리는 순간, Karpenter는 필요한 전체 노드 개수를 한꺼번에 인지하게 됩니다.
하지만 Karpenter가 노드를 하나씩 띄우는 속도 차이 때문에, 먼저 뜬 노드에 일부 파드가 먼저 배치되는 현상은 여전히 발생합니다.

#### 2. PodGroup 기반 Coscheduling 적용 (필수) ####
Karpenter가 노드를 준비하는 동안 발생할 수 있는 "일부만 실행되는 현상"을 막으려면, 쿠버네티스 스케줄러 레벨에서 Scheduler Plugins의 Coscheduling을 함께 사용해야 합니다.
* PodGroup 정의: TrainJob의 모든 파드가 하나의 PodGroup에 속하게 합니다.
* MinMember 설정: 예를 들어 GPU 파드 4개가 모두 준비되어야만 스케줄링을 시작하도록 minMember: 4를 설정합니다.
* 동작 방식:
  * Karpenter가 노드 2개를 먼저 띄워도, 스케줄러는 minMember가 충족되지 않았으므로 이 노드들에 파드를 배치하지 않고 대기시킵니다.
  * 나머지 노드 2개가 마저 떠서 전체 4개의 자리가 확보되는 순간, 모든 파드를 동시에 노드에 바인딩합니다.

#### 3. Karpenter의 통합 최적화 ####
Karpenter가 갱 스케줄링에 최적화되도록 하려면 NodePool(또는 구 버전의 Provisioner)에서 다음 설정을 고려해야 합니다.
* Batching Window: Karpenter는 짧은 시간 동안 발생하는 파드 요청을 모아서 처리합니다. TrainJob의 모든 파드 요청이 Karpenter에게 한꺼번에 전달되도록 Kueue가 제어하므로, Karpenter는 한 번의 작업으로 필요한 노드들을 동시에 프로비저닝하기 시작합니다.
* Capacity-type: 갱 스케줄링 시 스팟(Spot) 인스턴스를 쓰면 일부 노드만 확보되고 나머지는 재고 부족으로 실패할 위험이 큽니다. 안정적인 갱 스케줄링을 원한다면 해당 작업에 한해 온디맨드(on-demand)를 권장합니다.

#### 4. 설정 요약 예시 ####
TrainJob의 각 리플리카 템플릿에 아래 내용을 포함해야 합니다.
```
spec:
  template:
    spec:
      schedulerName: coscheduling               # 스케줄러 플러그인 지정
      priorityClassName: high-priority          # Kueue와 연동된 우선순위
      containers:
      - ...
```

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/operator-guides/job-scheduling/
* https://kueue.sigs.k8s.io/docs/overview/
* https://thenewstack.io/kueue-can-now-schedule-kubernetes-batch-jobs-across-clusters/
