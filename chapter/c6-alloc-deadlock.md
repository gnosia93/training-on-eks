```
$ kubectl describe pytorchjob pytorch-dist-job
   
Name:         pytorch-dist-job
Namespace:    default
Labels:       <none>
Annotations:  <none>
API Version:  kubeflow.org/v1
Kind:         PyTorchJob
Metadata:
  Creation Timestamp:  2025-12-19T08:46:16Z
  Generation:          1
  Resource Version:    1582888
  UID:                 cfa77767-9da2-4d6a-ab67-f149ad7acdd8
Spec:
  Pytorch Replica Specs:
    Master:
      Replicas:        1
      Restart Policy:  OnFailure
      Template:
        Spec:
          Affinity:
            Pod Affinity:
              Required During Scheduling Ignored During Execution:
                Label Selector:
                  Match Expressions:
                    Key:       training.kubeflow.org/job-name
                    Operator:  In
                    Values:
                      pytorch-dist-job
                Topology Key:  kubernetes.io/hostname
          Containers:
            Args:
              git clone https://github.com/gnosia93/training-on-eks /workspace/code
cd /workspace/code/samples/fsdp
echo "working directory: "$(pwd)
pip install -r requirements.txt
torchrun --nnodes 4 --nproc_per_node 1 t5-fsdp.py
            Command:
              /bin/bash
              -c
            Image:  public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
            Name:   pytorch
            Resources:
              Limits:
                nvidia.com/gpu:  1
              Requests:
                nvidia.com/gpu:  1
          Node Selector:
            karpenter.sh/nodepool:  gpu
          Tolerations:
            Effect:    NoSchedule
            Key:       nvidia.com/gpu
            Operator:  Exists
    Worker:
      Replicas:        3
      Restart Policy:  OnFailure
      Template:
        Spec:
          Affinity:
            Pod Affinity:
              Required During Scheduling Ignored During Execution:
                Label Selector:
                  Match Expressions:
                    Key:       training.kubeflow.org/job-name
                    Operator:  In
                    Values:
                      pytorch-dist-job
                Topology Key:  kubernetes.io/hostname
          Containers:
            Args:
              git clone https://github.com/gnosia93/training-on-eks /workspace/code
cd /workspace/code/samples/fsdp
echo "working directory: "$(pwd)
pip install -r requirements.txt
torchrun --nnodes 4 --nproc_per_node 1 t5-fsdp.py

            Command:
              /bin/bash
              -c
            Image:  public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
            Name:   pytorch
            Resources:
              Limits:
                nvidia.com/gpu:  1
              Requests:
                nvidia.com/gpu:  1
          Node Selector:
            karpenter.sh/nodepool:  gpu
          Tolerations:
            Effect:    NoSchedule
            Key:       nvidia.com/gpu
            Operator:  Exists
  Run Policy:
    Clean Pod Policy:  Running
    Suspend:           false
Status:
  Completion Time:  2025-12-19T08:56:33Z
  Conditions:
    Last Transition Time:  2025-12-19T08:46:16Z
    Last Update Time:      2025-12-19T08:46:16Z
    Message:               PyTorchJob pytorch-dist-job is created.
    Reason:                PyTorchJobCreated
    Status:                True
    Type:                  Created
    Last Transition Time:  2025-12-19T08:55:08Z
    Last Update Time:      2025-12-19T08:55:08Z
    Message:               PyTorchJob pytorch-dist-job is running.
    Reason:                PyTorchJobRunning
    Status:                False
    Type:                  Running
    Last Transition Time:  2025-12-19T08:56:33Z
    Last Update Time:      2025-12-19T08:56:33Z
    Message:               PyTorchJob pytorch-dist-job is successfully completed.
    Reason:                PyTorchJobSucceeded
    Status:                True
    Type:                  Succeeded
  Replica Statuses:
    Master:
      Selector:   training.kubeflow.org/job-name=pytorch-dist-job,training.kubeflow.org/operator-name=pytorchjob-controller,training.kubeflow.org/replica-type=master
      Succeeded:  1
    Worker:
      Succeeded:  3
  Start Time:     2025-12-19T08:46:16Z
Events:
  Type     Reason                   Age                From                   Message
  ----     ------                   ----               ----                   -------
  Normal   SuccessfulCreatePod      13m                pytorchjob-controller  Created pod: pytorch-dist-job-master-0
  Normal   SuccessfulCreateService  13m                pytorchjob-controller  Created service: pytorch-dist-job-master-0
  Normal   SuccessfulCreatePod      13m                pytorchjob-controller  Created pod: pytorch-dist-job-worker-0
  Normal   SuccessfulCreatePod      13m                pytorchjob-controller  Created pod: pytorch-dist-job-worker-1
  Normal   SuccessfulCreatePod      13m                pytorchjob-controller  Created pod: pytorch-dist-job-worker-2
  Normal   SuccessfulCreateService  13m                pytorchjob-controller  Created service: pytorch-dist-job-worker-0
  Normal   SuccessfulCreateService  13m                pytorchjob-controller  Created service: pytorch-dist-job-worker-1
  Normal   SuccessfulCreateService  13m                pytorchjob-controller  Created service: pytorch-dist-job-worker-2
  Warning  Unschedulable            13m                pytorchjob-controller  Error pod pytorch-dist-job-master-0 condition message: 0/2 nodes are available: 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
  Warning  Unschedulable            12m (x2 over 13m)  pytorchjob-controller  Error pod pytorch-dist-job-worker-0 condition message: 0/2 nodes are available: 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
  Warning  Unschedulable            12m (x2 over 13m)  pytorchjob-controller  Error pod pytorch-dist-job-worker-1 condition message: 0/2 nodes are available: 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
  Warning  Unschedulable            12m (x4 over 13m)  pytorchjob-controller  Error pod pytorch-dist-job-worker-2 condition message: 0/2 nodes are available: 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
  Warning  Unschedulable            12m (x4 over 12m)  pytorchjob-controller  Error pod pytorch-dist-job-master-0 condition message: 0/3 nodes are available: 1 node(s) had untolerated taint {karpenter.sh/unregistered: }, 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/3 nodes are available: 3 Preemption is not helpful for scheduling.
  Warning  Unschedulable            12m (x2 over 12m)  pytorchjob-controller  Error pod pytorch-dist-job-worker-1 condition message: 0/3 nodes are available: 1 node(s) had untolerated taint {karpenter.sh/unregistered: }, 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/3 nodes are available: 3 Preemption is not helpful for scheduling.
  Warning  Unschedulable            12m (x2 over 12m)  pytorchjob-controller  Error pod pytorch-dist-job-worker-0 condition message: 0/3 nodes are available: 1 node(s) had untolerated taint {karpenter.sh/unregistered: }, 2 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/3 nodes are available: 3 Preemption is not helpful for scheduling.
  Normal   SuccessfulDeletePod      5m43s              pytorchjob-controller  Deleted pod: pytorch-dist-job-worker-0
  Normal   ExitedWithCode           3m10s              pytorchjob-controller  Pod: default.pytorch-dist-job-worker-0 exited with code 0
```

이 로그는 강력한 포드 어피니티(Pod Affinity)와 카펜터(Karpenter)의 노드 프로비저닝 속도가 충돌하여 발생하는 전형적인 스케줄링 실패 상황을 보여줍니다. 2025년 기준 Kubernetes 클러스터에서 자주 발생하는 현상입니다. [1], [2]

### 1. 로그 분석: 왜 실패했나? ###
* 교착 상태 (Deadlock): 설정한 podAffinity는 "모든 Pod가 동일한 호스트에 있어야 한다"고 강제합니다. 하지만 노드가 한 대도 없는 상태에서 Pod들이 생성되면, 스케줄러는 "기준이 될 첫 번째 Pod가 어디에도 없으니 나머지 Pod도 배치할 수 없다"고 판단하여 아무 노드에도 Pod를 할당하지 못하게 됩니다. [3]
* Karpenter 미등록 (Untolerated Taint): 로그 중 karpenter.sh/unregistered 테인트가 있는 노드는 아직 완전히 초기화되지 않은 노드입니다. Pod들이 이 노드를 기다리지 못하고 스케줄링에 실패하고 있습니다. [4]
* 리소스 불일치: "2 node(s) didn't match Pod's node affinity/selector"는 기존에 떠 있는 노드들이 GPU가 없거나, nodeSelector에 정의된 라벨을 가지고 있지 않음을 의미합니다. [5]

### 해결방법 ### 
#### 1. Pod Affinity 조건 완화 ####
requiredDuringScheduling... (강제) 대신 preferredDuringScheduling... (선호)를 사용하면, 노드가 없는 상태에서도 첫 번째 Pod가 노드 생성을 트리거할 수 있게 된다.

```
affinity:
  podAffinity:
    preferredDuringSchedulingIgnoredDuringExecution: # 강제에서 선호로 변경
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: training.kubeflow.org/job-name
            operator: In
            values:
            - pytorch-dist-job
        topologyKey: kubernetes.io/hostname
```

#### 2. 노드 셀렉터 확인 ####
Pod에 nodeSelector가 설정되어 있다면, Karpenter의 NodePool이 해당 라벨을 가진 노드를 생성할 수 있는지 확인하세요.
예: accelerator: nvidia-gpu 등의 라벨이 노드에 부여되도록 NodePool 설정을 확인해야 합니다. [2]

#### 3. 마스터 Pod에 어피니티 제외 ####
마스터 Pod는 어피니티 없이 먼저 뜨게 하고, 워커들만 마스터를 따라가도록 설정하면 교착 상태를 피할 수 있습니다.
  
