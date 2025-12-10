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

깃으로 다운로드 받은 후 해당 디렉토리로 이동한다. 
```
git clone https://github.com/gnosia93/training-on-eks.git
cd training-on-eks
```
pytorch 네임스페이스를 생성하고 pytorch-dist-job.yaml 을 kustomize 로 패치하여 pytorch ddp 작업을 실행한다. 
```
kubectl create ns pytorch
kubectl apply -k kustomize/overlays/ddp/
kubectl get pods -n pytorch
```
[결과]
```
NAME                        READY   STATUS            RESTARTS   AGE
pytorch-dist-job-master-0   1/1     Running           0          3m34s
pytorch-dist-job-worker-0   0/1     PodInitializing   0          3m34s
pytorch-dist-job-worker-1   0/1     PodInitializing   0          3m33s
```
마스터와 워커로드의 세부 정보를 조회한다. 
```
kubectl describe pod pytorch-dist-job-master-0 -n pytorch
kubectl describe pod pytorch-dist-job-worker-0 -n pytorch
kubectl describe pod pytorch-dist-job-worker-1 -n pytorch

kubectl logs pytorch-dist-job-master-0 -n pytorch
kubectl logs pytorch-dist-job-worker-0 -n pytorch
kubectl logs pytorch-dist-job-worker-1 -n pytorch
```

#### 참고 - pytorchjob 삭제하기 ####
```
kubectl delete pytorchjob pytorch-dist-job -n pytorch
```

## 부연설명 ##

#### 1. EC2 보다 초기화 시간이 긴 이유 ####

torchrun을 일반 EC2에서 실행할 때는 이미 환경이 구성된 상태에서 프로세스만 띄우면 되지만, PyTorchJob를 활용하게 되는 경우 노드 확보, 이미지 다운로드, 네트워크 설정, 데이터 준비 등 모든 과정을 포함하므로 시작하는 데 시간이 더 오래 걸리게 된다.
초기화 속도를 높이려면 GPU 이미지를 노드에 미리 캐시해두거나, 데이터셋을 PVC에 미리 준비해두는 등의 최적화가 필요합니다.


#### 2. cleanPodPolicy ####
cleanPodPolicy는 Job이 성공하거나 실패했을 때 워커 Pod들을 어떻게 처리할지 Kubernetes 오퍼레이터에게 지시하는 정책입니다. 설정 가능한 값은 다음과 같습니다.
* 디폴트 (None): 작업 완료 후 모든 Pod 유지 (로그/디버깅 용이)
* 사용자 설정 (Running): 완료된 Pod는 유지, 실행 중인 Pod만 종료
* 권장 설정 (All): 작업 완료 후 모든 Pod 삭제 (리소스 정리 자동화)

#### 3. restartPolicy ####

#### 4. pytorch-dist-job.yaml 이해하기 ####

#### 5. kustomize ####

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/
