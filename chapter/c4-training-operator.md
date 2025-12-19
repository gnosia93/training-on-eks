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

training-on-eks 의 ddp 디렉토리로 이동하여 pytorch DDP 작업을 실행한다 
```
git clone https://github.com/gnosia93/training-on-eks.git
cd /home/ec2-user/training-on-eks/kustomize/overlays/ddp

kubectl kustomize .                          # pytorhjob yaml 확인
kubectl kustomize . | kubectl apply -f -     # 실행
```

pytorch job 과 관련 파드 정보를 조회한다. 
```
kubectl get pytorchjobs
kubectl get all
```
[결과]
```
NAME                        READY   STATUS            RESTARTS   AGE
pytorch-dist-job-master-0   1/1     Running           0          3m34s
pytorch-dist-job-worker-0   0/1     PodInitializing   0          3m34s
```

마스터와 워커로드의 세부 정보를 조회한다. 
```
kubectl describe pod pytorch-dist-job-master-0
kubectl describe pod pytorch-dist-job-worker-0

kubectl logs pytorch-dist-job-master-0 
kubectl logs pytorch-dist-job-worker-0 
```

트레이닝 작업이 완료 되면 pytorchjob 삭제한다.
```
kubectl delete pytorchjob pytorch-dist-job
```

#### 카펜터 노드 로그 확인 ####
카펜터의 노드 프로비저닝 상태를 조회하고자 하는 경우 아래 명령어로 가능하다.
```
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter
```

## pytorch-dist-job 의 이해 ##
* https://github.com/gnosia93/training-on-eks/blob/main/kustomize/base/pytorch-dist-job.yaml
```
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-dist-job
spec:
  runPolicy:
    cleanPodPolicy: Running
  
  pytorchReplicaSpecs:
    Master:                       
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            karpenter.sh/nodepool: gpu
#            node.kubernetes.io/instance-type: g6.2xlarge     
          tolerations:             
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: pytorch
            image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
            command: ["/bin/bash", "-c"] 
            args: 
              - |
                git clone github.com /workspace/code    
                python /workspace/code/training.py
            resources:
              limits:
                nvidia.com/gpu: "1"
              requests:
                nvidia.com/gpu: "1"    
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            karpenter.sh/nodepool: gpu
#            node.kubernetes.io/instance-type: g6.2xlarge   
          tolerations:            
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: pytorch
            image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
            command: ["/bin/bash", "-c"] 
            args: 
              - |
                git clone github.com /workspace/code    
                python /workspace/code/training.py
            resources:
              limits:
                nvidia.com/gpu: "1"
              requests:
                nvidia.com/gpu: "1"
```

* https://github.com/gnosia93/training-on-eks/blob/main/kustomize/overlays/ddp/kustomization.yaml

 

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/
