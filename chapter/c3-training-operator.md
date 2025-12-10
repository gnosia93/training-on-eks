## 트레이닝 오퍼레이터 설치 ##
kubeflow 의 트레이닝만 오퍼레이터만 단독으로 설치한다. 분산 트레이닝을 실행하기 위해서 다른 모듈은 필요하지 않다.  
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

## 트레이닝 작업 실행하기 ##

```
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-dist-dynamic-job
  namespace: kubeflow-user-example-com # 사용자의 네임스페이스로 변경하세요
spec:
  runPolicy:
    cleanPodPolicy: Running
  
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          initContainers:
          - name: clone-repository
            image: alpine/git:latest
            # 환경 변수 GIT_REPO_URL 사용
            command: ["/bin/sh", "-c", "git clone $(GIT_REPO_URL) /workspace/code"]
            env:
            - name: GIT_REPO_URL
              # value는 외부에서 주입될 값입니다. 
              # 실제 사용 시 이 필드를 동적으로 채워야 합니다.
              value: "github.com" 
            volumeMounts:
            - name: workdir
              mountPath: /workspace
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime
            command: ["python", "/workspace/code/main.py"] 
            volumeMounts:
            - name: workdir
              mountPath: /workspace
          volumes:
          - name: workdir
            emptyDir: {}

    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          # EKS Teleration 설정
          tolerations:
          - key: "gpu"
            operator: "Equal"
            value: "true"
            effect: "NoSchedule"
            
          initContainers:
          - name: clone-repository
            image: alpine/git:latest
            # 환경 변수 GIT_REPO_URL 사용
            command: ["/bin/sh", "-c", "git clone $(GIT_REPO_URL) /workspace/code"]
            env:
            - name: GIT_REPO_URL
              # Master와 동일하게 외부에서 주입될 값
              value: "github.com" 
            volumeMounts:
            - name: workdir
              mountPath: /workspace

          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime
            command: ["python", "/workspace/code/main.py"] 
            resources:
              limits:
                nvidia.com: "1"
              requests:
                nvidia.com: "1"
            volumeMounts:
            - name: workdir
              mountPath: /workspace
          
          volumes:
          - name: workdir
            emptyDir: {}
```



## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/
