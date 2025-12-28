![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/deepspeed-logo.svg) 마이크로소프트(Microsoft)가 개발한 [DeepSpeed](https://www.deepspeed.ai/)는 ZeRO 기술을 통해 GPU 메모리 점유율을 획기적으로 낮추고, 여러 연산을 하나로 묶는 커널 퓨전(Kernel Fusion) 및 커스텀 커널 최적화로 연산 속도를 극대화하는 딥러닝 가속 엔진이다. DeepSpeed 공식 문서에 따르면, 이 라이브러리는 하드웨어 한계를 넘어선 초거대 AI 모델의 학습과 추론을 지원하며, 특히 AWS EFA와 같은 고성능 네트워크 환경에서 통신 효율을 최대로 끌어올린다. 적은 컴퓨팅 자원으로도 대규모 모델을 가장 빠르고 효율적으로 구동할 수 있게 해주는 최적화 프레임워크 이다.

## Llama-3-8B ##
이번 챕터에서는 Llama-3-8B 모델을 허깅 페이스 trainer / deepspeed 라이브러리를 이용하여 훈련 한다. 

* [Llama-3-8B Trainer](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/llama-3-8b.py)
* [DeepSpeed Config](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/llama-3-8b-stage3.json)
* [requirements.txt](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/requirements.txt)
* [TrainJob YAML](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/trainjob.yaml)  

### 훈련 설정값 (DeepSpeed Config) ###
* gradient_checkpointing=True
역전파 시 필요한 중간 연산 결과를 저장하지 않고 다시 계산하여 메모리 사용량을 줄임(8B 이상의 모델에서는 필수)
* bf16=True
FP16은 값이 갑자기 커지면(Overflow) 학습이 망가질 수 있는데, BFloat16은 표현 범위가 넓어 안정적임.
* offload_param & offload_optimizer
Stage 3 설정 중 offload를 활성화하면, GPU 메모리가 가득 찼을 때 모델 파라미터를 CPU RAM으로 자미 이동함. 학습 속도는 느려지지만 훨씬 더 큰 모델을 학습할 수 있음.
* meta device 초기화
수십 GB의 모델을 한 GPU가 먼저 다 읽으려 하면 시작하자마자 OOM 발생함. AutoModel.from_config 사용하면 모델을 실제 메모리에 올리기 전에 구조만 파악하고, DeepSpeed가 각 GPU에 쪼개서 로드하도록 유도.

### 훈련 인스턴스 - g6e.8xlarge / EFA ###
```
$ aws ec2 describe-instance-types \
    --instance-types g6e.8xlarge \
    --query "InstanceTypes[*].{InstanceType:InstanceType, \
        EfaSupported:NetworkInfo.EfaSupported, \
        MaxNetworkInterfaces:NetworkInfo.MaximumNetworkInterfaces, \
        NetworkPerformance:NetworkInfo.NetworkPerformance}" --output table
--------------------------------------------------------------------------------
|                             DescribeInstanceTypes                            |
+--------------+---------------+------------------------+----------------------+
| EfaSupported | InstanceType  | MaxNetworkInterfaces   | NetworkPerformance   |
+--------------+---------------+------------------------+----------------------+
|  True        |  g6e.8xlarge  |  8                     |  25 Gigabit          |
+--------------+---------------+------------------------+----------------------+
```
* "kubectl get nodepool" 명령어로 gpu 노드풀이 존재하는 지 확인한다. 없으면 [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)를 참고하여 생성한다. 

* efa 디바이스 플러그인 설치
```
helm repo add eks https://aws.github.io/eks-charts
helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin --namespace kube-system

# operator:exists toleration 부여 
kubectl patch ds aws-efa-k8s-device-plugin -n kube-system --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/tolerations/-", "value": {"operator": "Exists"}}
]'
# 플러그인 설치 확인
kubectl get ds aws-efa-k8s-device-plugin -n kube-system
```
[결과]
```
NAME                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
aws-efa-k8s-device-plugin   0         0         0       0            0           <none>          27s
```
efa 플러그인 설치 시점에는 efa 인터페이스를 지원하는 노드가 없는 관계로 데몬수가 0 으로 표시된다. 

# 큐브플로우 Trainer 설치 ###
```
sudo dnf install git -y

export VERSION=v2.1.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"

kubectl get clustertrainingruntimes
```
* efa 관련 설정을 추가하기 위해 torch-distributed 런타임을 수정한다. 
```
$ kubectl edit clustertrainingruntime torch-distributed 

apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: torch-distributed
spec:
  template:
    spec:
      containers:
        - name: node
          # EFA 및 분산 학습을 위한 보안 설정 추가
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]
```

* trainjob 명령어
  * 잡 확인 - kubectl get trainjob                       
  * 잡 삭제 - kubectl delete trainjob llama-3-8b        
  * 잡 상세 - kubectl describe trainjob llama-3-8b
    
### 갱 스케줄링 ###
이 예제에서는 갱 스케줄링 기능을 활성화 하지 않는다. 즉 카펜터에서 GPU 노드를 프러비저닝 하는 즉시 파드가 스케줄링 된다.  

### 체크 포인팅 ###
체크포인트를 병렬 분산 파일 시스템인 러스터에 저장할 예정이다. 러스터가 없는 경우 [C9. 체크포인트 저장하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c9-lustre-eks.md)를 참조해서 러스터를 설치한다.   

### 훈련 시작 ###
```
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export INSTANCE_TYPE=g6e.8xlarge               # 훈련 인스턴스 타입   
export AZ=${AWS_REGION}a                 
export NODE_NUM=4                              # 4대 
export GPU_PER_NODE=1                          # g6e.8xlarge 타입은 GPU 가 1장이다.
export EFA_PER_NODE=8                          # 200Gbp 사용
export HF_TOKEN="<your huggingface token>"     # Llama-3 모델은 HF 인증이 필요.

cd ~/training-on-eks/samples/deepspeed
envsubst < trainjob.yaml | kubectl apply -f -            # envsubst 는 trainjob.yaml 파일 내부의 환경변수를 실제 값으로 치환해 준다.
```

훈련 작업에 참여하는 파드 리스트를 조회한다. 
```
kubectl get pods
```
[결과]
```
NAME                        READY   STATUS              RESTARTS   AGE
llama-3-8b-node-0-0-8bnd8   0/1     ContainerCreating   0          86s
llama-3-8b-node-0-1-zf275   0/1     ContainerCreating   0          86s
llama-3-8b-node-0-2-qnwc6   0/1     ContainerCreating   0          86s
llama-3-8b-node-0-3-r455m   0/1     ContainerCreating   0          86s
```
Pod 상태 상세정보 및 이벤트(Events)를 확인한다. 설정오류 및 기타 원인으로 인해 컨테이너가 작업을 시작하지 못하는 경우 그 원인을 쉽게 파악할 수 있다.  
```
kubectl describe pod llama-3-8b-node-0-1-zf275
```

카펜터가 프로비저닝 한 노드 리스트를 조회한다. 만약 GPU 노드가 보이지 않는다면 카펜터 로그를 확인해 봐야 한다. 
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
NAME                                                INSTANCE       ARCH       OS                             GPU       CAPACITY
ip-10-0-4-96.ap-northeast-2.compute.internal    g6e.8xlarge    amd64      Amazon Linux 2023.9.20251208   1         spot
ip-10-0-5-17.ap-northeast-2.compute.internal    g6e.8xlarge    amd64      Amazon Linux 2023.9.20251208   1         on-demand
ip-10-0-5-251.ap-northeast-2.compute.internal   g6e.8xlarge    amd64      Amazon Linux 2023.9.20251208   1         on-demand
ip-10-0-5-55.ap-northeast-2.compute.internal    g6e.8xlarge    amd64      Amazon Linux 2023.9.20251208   1         on-demand
...
```


### 훈련 모니터링 ###
* GPU 모니터링
* 노드 모니터링
* EFA 모니터링
* 러스터 모니터링.
* 훈련 로그 모니터링.
 
## 레퍼런스 ##
* [Simple DeepSpeed](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c10-deepspeed-simple.md)


