## 트레이닝 오퍼레이터 설치 ##
kubeflow 의 여러 모듈중 트레이닝 오퍼레이터만 단독으로 설치한다. 기본적인 분산 트레이닝을 실행하기 위해서 다른 모듈은 필요하지 않다.  
```
sudo dnf install git -y
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
kubectl get crd | grep kubeflow
kubectl get pods -n kubeflow
```

[결과]
```
mpijobs.kubeflow.org                            2025-12-10T11:29:57Z
mxjobs.kubeflow.org                             2025-12-10T11:29:58Z
paddlejobs.kubeflow.org                         2025-12-10T11:29:58Z
pytorchjobs.kubeflow.org                        2025-12-10T11:29:59Z
tfjobs.kubeflow.org                             2025-12-10T11:30:00Z
xgboostjobs.kubeflow.org                        2025-12-10T11:30:01Z

NAME                                 READY   STATUS    RESTARTS   AGE
training-operator-79cc5c4557-lzqnt   1/1     Running   0          4m12s
```

## 파이썬 SDK 설치 ##
```
sudo dnf install python3-pip -y
python3 --version
pip install -U kubeflow-training
pip install -U "kubeflow-training[huggingface]"
```
kubeflow 의 경우 SDK 를 이용하여 분산 훈련 작업을 실행하는 것이 기본 설계 사상이지만, 여기서는 yaml 을 사용하여 트레이닝 작업을 실행한다.   

## 트레이닝 작업 실행하기 ##

먼저 네임스페이스를 생성한다. 
```
kubectl create ns pytorch
```

[pytorch-dist-job.yaml]
```
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-dist-job
  namespace: pytorch 
spec:
  runPolicy:
    cleanPodPolicy: Running
  
  pytorchReplicaSpecs:
    Master:                       # 마스터는 GPU 연산 작업에는 참여하지 않는다. GPU Toleration 설정 불필요
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
            command: ["/bin/bash", "-c"] 
            args: 
              - |
                git clone <GIT_REPO> /workspace/code    
                python /workspace/code/training.py
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          tolerations:            # GPU Toleration 설정 
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          - key: "gpu-workload"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
              command: ["/bin/bash", "-c"] 
            args: 
              - |
                git clone <GIT_REPO> /workspace/code    
                python /workspace/code/training.py 
            resources:
              limits:
                nvidia.com/gpu: "1"
              requests:
                nvidia.com/gpu: "1"
```

* github 코드 주소 생성

* 분산작업 스케줄링
```
kustomize 를 <GIT_REPO> 값 수정후 스케줄링
```

## 부연설명 ##

### 1. restartPolicy 에 대해서 ###



### 2. EC2 torchrun 과 Kubeflow PyTorchJob 의 차이점 ###
#### EC2 (torchrun 수동 실행): ####
- 랭크 0: 코디네이터 겸 GPU 연산 참여 (모든 노드가 동일한 역할 수행)
- 월드 사이즈: 4개의 EC2 사용 시, 월드 사이즈 4.
- 수동으로 마스터 IP 와 포트 정보를 자신을 포함한 모든 워크들에게 파라미터로 전달해야 한다.  

#### Kubeflow PyTorchJob (Operator 사용): ####
- Master Pod (랭크 0): 마스터 파드는 실제 GPU 연산(훈련)에 직접 참여하지 않지만, 분산 시스템의 원활한 작동을 위해 필수적인 조정자 및 관리자 역할을 한다. 통신 그룹 초기화 및 조정 (Orchestration), 로그 취합 및 출력 관리, 모델 체크포인트 및 저장 관리, 데이터셋 분할 조정 (IterableDataset 사용 시), 사전 준비 및 후처리 작업 (경우에 따라) 을 수행한다. 
- Worker Pods (랭크 1~4): 전용 연산 참여자. GPU 리소스를 할당받아 연산만 수행합니다.
- 월드 사이즈: 4개의 워커와 1개의 마스터 사용 시, 월드 사이즈 5.

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/
