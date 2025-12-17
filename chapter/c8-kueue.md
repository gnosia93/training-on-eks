## Kueue ##

### 1. 설치 ###
```
cat <<EOF > kueue-values.yaml
controllerManager:
  manager:
    configuration:
      integrations:
        frameworks:
        - "batch/job"
        - "kubeflow.org/pytorchjob"
#        - "kubeflow.org/tfjob"   
#        - "ray.io/rayjob"      
EOF

helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --namespace kueue-system \
  --create-namespace \
  -f kueue-values.yaml
```

### 2. Kueue가 설정 ###
PyTorchJob을 실행하기 전에 Kueue가 해당 작업을 인식하고 리소스를 할당할 수 있도록 ResourceFlavor, ClusterQueue, 그리고 LocalQueue 세 가지 핵심 리소스가 설정되어 있어야 합니다

#### 1. ResourceFlavor 정의 ####
클러스터에 존재하는 실제 리소스(여기서는 GPU)의 종류와 레이블을 정의합니다. 
```
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: nvidia-gpu-flavor
spec:
  nodeSelector:
    # 이 레이블은 GPU 노드에 실제로 있어야 합니다.
    # 예: "cloud.provider.com": "nvidia-a100"
    # 또는 간단한 예시로 "karpenter.sh/capacity-type": "on-demand"
    kueue.x-k8s.io/default-flavor: "true" 
  tolerations:
  - key: "kueue.x-k8s.io/gpu"
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


## PytorchJob 수정 및 실행 ##

```
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: pytorch-dist-mnist
  namespace: team-a  # LocalQueue가 생성된 네임스페이스
  labels:
    # 1. 이 레이블이 있어야 Kueue가 작업을 관리합니다.
    kueue.x-k8s.io/queue-name: team-a-queue
spec:
  # 2. 제출 시점에 true로 설정해야 Kueue가 쿼터를 확인한 뒤 실행합니다.
  runPolicy:
    suspend: true
    cleanPodPolicy: Running
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        ...chJob 제출
kubectl apply -f pytorch-job.yaml

# 2. Kueue 워크로드 상태 확인 (Admitted: true가 되면 실행 시작)
kubectl get workloads -n team-a

# 3. PyTorchJob 상태 확인 (ALL-OR-NOTHING 확인)
kubectl get pytorchjob pytorch-dist-mnist -n team-a
```



### PytorchJob 고려사항 ###

#### 1. 갱 스케줄링 (Gang Scheduling) 보장 ####
PyTorchJob과 같은 분산 학습은 모든 워커(Worker)가 동시에 실행되지 않으면 학습이 진행되지 않고 자원만 점유하는 상황이 발생합니다.
* Kueue의 역할: Kueue는 All-or-nothing 방식을 지원합니다. 설정된 모든 레플리카(Master + Workers)를 수용할 수 있는 자원이 있을 때만 작업을 승인(Admit)합니다.
* 주의: 만약 Kueue 없이 일반 스케줄러만 사용하면, 일부 워커만 생성되어 GPU 자원을 점유한 채 나머지 워커를 무한정 기다리는 '교착 상태(Deadlock)'에 빠질 수 있습니다.

#### 2. Suspend 모드 필수 사용 ####
Kueue가 작업을 제어하게 하려면 제출 시점에 작업이 중단된 상태여야 합니다.
* 설정: PyTorchJob의 spec.runPolicy.suspend 필드를 true로 설정하여 제출하세요.
* 동작: Kueue가 자원을 할당하는 순간 이 값을 false로 바꿔주며 학습이 시작됩니다. 만약 false로 제출하면 Kueue의 관리 기능을 우회하여 즉시 실행을 시도하므로 쿼터 관리가 무너집니다.

#### 3. Resource Flavor와 노드 선택 (카펜터 연동 시) ####
분산 학습 시 노드 간 통신 속도가 매우 중요합니다.
* Flavor 설정: ResourceFlavor를 사용하여 특정 노드 그룹(예: NVIDIA A100 등)이나 가용 영역(AZ)을 지정할 수 있습니다.
* 카펜터 최적화: 카펜터(Karpenter)를 함께 사용한다면, 분산 학습 포드들이 가급적 동일한 서브넷이나 물리적 위치에 배치되도록 topologySpreadConstraints나 nodeSelector를 PyTorchJob 템플릿 내에 설정하는 것이 네트워크 지연 시간을 줄이는 데 도움이 됩니다.

#### 4. 선점(Preemption) 정책 설계 ####
중요도가 낮은 학습이 자원을 점유하고 있을 때, 우선순위가 높은 학습이 들어오는 경우를 대비해야 합니다.
* PriorityClass: PyTorchJob에 적절한 priorityClassName을 부여하세요.
* Kueue 설정: ClusterQueue의 preemption 정책을 통해 낮은 우선순위의 작업을 중단시키고 고우선순위 작업을 먼저 실행할지 결정해야 합니다. (단, 학습 중단 시 체크포인트가 저장되도록 코드가 작성되어 있어야 합니다.)







---

### 2. 리소스 관리 구조 설정 ###
Kueue를 작동시키려면 리소스의 종류와 사용량을 정의하는 3가지 설정 파일이 필요합니다. 
* ResourceFlavor: 노드 풀의 특성(예: GPU 종류, Spot 인스턴스 여부)을 정의합니다.
* ClusterQueue: 클러스터 전체에서 팀들이 공유할 수 있는 리소스 할당량(Quota)을 정의합니다.
* LocalQueue: 특정 네임스페이스 사용자가 작업을 제출할 수 있는 통로입니다.


### 주요 특징 및 작동 방식 ###
* 작업 큐잉 및 할당량 관리
Kueue는 작업 수준 관리자 역할을 합니다. 리소스가 충분하지 않을 경우 작업을 대기열에 넣고, 리소스가 확보되면 작업을 클러스터에 제출하여 Pod가 생성되도록 합니다.
* 리소스 공유 및 공정성
여러 팀이나 사용자가 클러스터 리소스를 공정하게 공유할 수 있도록 계층적 할당량 관리를 제공합니다. 특정 네임스페이스에 할당된 리소스가 모두 사용되지 않을 경우, 다른 네임스페이스의 작업이 해당 유휴 리소스를 빌려 사용할 수 있도록 할 수 있습니다.
* 정책 기반 예약
리소스 가용성, 작업 우선 순위, 할당량 정책 등을 기반으로 작업 실행 여부를 결정합니다.

### Kueue의 구성 요소 ###
Kueue는 주로 다음 두 가지 핵심 사용자 정의 리소스(CRD)를 중심으로 작동합니다: 
* ClusterQueue: 클러스터 전체에서 공유되는 리소스 풀(예: CPU, 메모리, GPU 할당량)을 나타냅니다. 클러스터 관리자가 정의합니다.
* LocalQueue: 특정 네임스페이스 내에서 사용자가 작업을 제출하는 데 사용하는 큐입니다. 각 LocalQueue는 작업을 할당받을 ClusterQueue에 연결됩니다.

사용자가 LocalQueue에 작업을 제출하면, Kueue는 ClusterQueue의 할당량을 확인하여 작업 실행 가능 여부를 판단합니다. 이를 통해 리소스 부족으로 인한 데드락을 방지하고 클러스터 활용도를 최적화할 수 있습니다

### 카펜터와의 협업 ###
Kueue가 "언제 작업을 실행할지" 결정하는 작업 관리(Queueing) 역할을 한다면, 카펜터는 그 결정에 따라 "필요한 노드를 즉시 생성"하는 인프라 프로비저닝(Autoscaling) 역할을 수행합니다. 

#### 함께 사용할 때의 작동 방식 ####
* 작업 제출 및 대기: 사용자가 작업을 제출하면 Kueue가 이를 가로채서 대기열(LocalQueue)에 넣습니다.
* 리소스 승인 (Admission): Kueue가 설정된 쿼터(Quota)를 확인하고 작업을 승인하면, 작업의 suspend 상태를 해제합니다.
* 포드 생성: 작업이 승인되어 포드(Pod)들이 생성되지만, 처음에는 실행할 노드가 없어 Pending 상태가 됩니다.
* 카펜터의 노드 프로비저닝: 카펜터가 이 Pending 포드들을 감지하고, 포드의 요구사항(GPU, CPU 등)에 딱 맞는 최적의 노드를 즉시 프로비저닝합니다.
* 작업 실행: 노드가 준비되면 포드들이 배치되어 작업을 시작합니다. 

## 레퍼런스 ##
* https://kueue.sigs.k8s.io/docs/overview/
* https://www.redhat.com/ko/blog/openshift-joining-kueue
