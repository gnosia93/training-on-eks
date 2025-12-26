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

#### 1. ResourceFlavor 정의 ####
클러스터에 존재하는 실제 리소스(여기서는 GPU)의 종류와 레이블을 정의합니다. 
```
# gpu 노드풀 예시
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu
spec:
  template:
    metadata:
      labels:
        nvidia-type: "standard" # 이 레이블로 구분
# ... 생략
---
# gpu-efa 노드풀 예시
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-efa
spec:
  template:
    metadata:
      labels:
        nvidia-type: "efa" # 이 레이블로 구분
# ... 생략
```

```
# 1. 일반 GPU Flavor
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: flavor-gpu-standard
spec:
  nodeSelector:
    nvidia-type: "standard"
  tolerations:
  - key: "nvidia.com"
    operator: "Exists"
    effect: "NoSchedule"
---
# 2. EFA GPU Flavor
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: flavor-gpu-efa
spec:
  nodeSelector:
    nvidia-type: "efa"
  tolerations:
  - key: "nvidia.com"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "vpc.amazonaws.com" # EFA 전용 테인트가 있다면 추가
    operator: "Exists"
    effect: "NoSchedule"
```

#### 2. ClusterQueue 정의 (cluster-queue.yaml) #### 
클러스터 전체의 리소스 할당량과 정책을 정의합니다. queue-a라는 이름으로 총 GPU 10개까지 할당할 수 있도록 설정합니다.
```
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: cluster-queue-a
spec:
  resourceGroups:
    - flavors:
        - name: nvidia-gpu-flavor
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 10  # 총 GPU 10개 할당 가능
            - name: cpu
              nominalQuota: 40
            - name: memory
              nominalQuota: 100Gi
      # 갱 스케줄링(Gang Scheduling)을 위한 설정
      # 모든 파드가 준비될 때까지 기다릴지 여부 (선택 사항, v0.3.0 이후)
      # waitForPodsReady:
      #   enable: true
  # 이 ClusterQueue를 사용할 수 있는 LocalQueue의 네임스페이스 제약 (선택 사항)
  namespaceSelector: {} 
```

#### 3. LocalQueue 정의 (local-queue.yaml) ####
특정 네임스페이스(team-a) 내 사용자들이 작업을 제출하는 통로를 정의합니다. 이 LocalQueue가 위의 ClusterQueue를 참조합니다.
```
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: team-a-queue
  namespace: team-a
spec:
  # 위에서 정의한 ClusterQueue 이름
  clusterQueue: cluster-queue-a
```

#### 4. 주의해야 할 점 (매칭 확인) ####
설정이 꼬이지 않으려면 다음 사항만 일치시키면 됩니다.
* 카펜터의 NodePool 설정: 카펜터의 NodePool (또는 Provisioner)이 Kueue Flavor에서 지정한 레이블(예: GPU 타입, 인스턴스 타입)을 생성할 수 있는 권한과 범위 내에 있어야 합니다.
* 레이블 일치: Flavor에서 nvidia-a100을 지정했는데 카펜터 설정에는 g4dn.xlarge(T4 GPU)만 허용되어 있다면, 카펜터가 노드를 띄우지 못해 포드가 Pending 상태로 남게 됩니다.


## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/operator-guides/job-scheduling/
* https://kueue.sigs.k8s.io/docs/overview/
* https://thenewstack.io/kueue-can-now-schedule-kubernetes-batch-jobs-across-clusters/
