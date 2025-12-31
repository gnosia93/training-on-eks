![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/deepspeed-logo.svg) 마이크로소프트(Microsoft)가 개발한 [DeepSpeed](https://www.deepspeed.ai/)는 ZeRO 기술을 통해 GPU 메모리 점유율을 획기적으로 낮추고, 여러 연산을 하나로 묶는 커널 퓨전(Kernel Fusion) 및 커스텀 커널 최적화로 연산 속도를 극대화하는 딥러닝 가속 엔진이다. DeepSpeed 공식 문서에 따르면, 이 라이브러리는 하드웨어 한계를 넘어선 초거대 AI 모델의 학습과 추론을 지원하며, 특히 AWS EFA와 같은 고성능 네트워크 환경에서 통신 효율을 최대로 끌어올린다. 적은 컴퓨팅 자원으로도 대규모 모델을 가장 빠르고 효율적으로 구동할 수 있게 해주는 최적화 프레임워크 이다.

## Llama-3-8B ##
이번 챕터에서는 Llama-3-8B 모델을 허깅 페이스 trainer / deepspeed 라이브러리를 이용하여 훈련 한다. 

* [Llama-3-8B Trainer](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/llama-3-8b.py)
* [DeepSpeed 설정](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/llama-3-8b-stage3.json)
* [requirements.txt](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/requirements.txt)
* [TrainJob YAML](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/trainjob.yaml)  

### 훈련 인스턴스 - g6e.8xlarge / EFA ###
```
$ aws ec2 describe-instance-types \
    --instance-types g6e.12xlarge \
    --query "InstanceTypes[*].{InstanceType:InstanceType, \
        EfaSupported:NetworkInfo.EfaSupported, \
        MaxNetworkInterfaces:NetworkInfo.MaximumNetworkInterfaces, \
        MaxEfaInterfaces:NetworkInfo.EfaInfo.MaximumEfaInterfaces, \
        NetworkPerformance:NetworkInfo.NetworkPerformance}" --output table
----------------------------------------------------------------------------------------------------
|                                       DescribeInstanceTypes                                      |
+--------------+---------------+-------------------+------------------------+----------------------+
| EfaSupported | InstanceType  | MaxEfaInterfaces  | MaxNetworkInterfaces   | NetworkPerformance   |
+--------------+---------------+-------------------+------------------------+----------------------+
|  True        |  g6e.12xlarge |  1                |  10                    |  100 Gigabit         |
+--------------+---------------+-------------------+------------------------+----------------------+
```

### 카펜터 노드풀 및 디바이스 플러그인 설치 ###
* "kubectl get nodepool" 명령어로 gpu 노드풀이 존재하는 지 확인한다. 없으면 [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)를 참고하여 생성한다. 

* nvidia 디바이스 플러그인 설치
```
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia \
  --create-namespace \
  --version 0.18.0 \
  --set gfd.enabled=true
```  

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

### 큐브플로우 Trainer 설치 ###
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
      shareProcessNamespace: true 
      hostIPC: true
#      hostNetwork: true
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
이 예제에서는 편의상 체크포인트 데이터를 저장하지 않는다. 체크포인트를 저장하기 위해서는 S3, EFS, 러스터와 같은 병렬 분산 파일 시스템이 필요하다.  

### 훈련 시작 ###
envsubst 는 파리미터로 나열된 환경변수만 치환해 준다. 
```
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export INSTANCE_TYPE=g6e.12xlarge              # 훈련 인스턴스 타입   (g6e.48xlarge)
export AZ=${AWS_REGION}a                 
export NODE_NUM=4                              # 4대 
export GPU_PER_NODE=4                          # g6e.12xlarge 타입은 GPU 가 4장이다.
export EFA_PER_NODE=1                          # 100Gbp 사용
export HF_TOKEN="<your huggingface token>"     # Llama-3 모델은 HF 인증이 필요.

cd ~/training-on-eks/samples/deepspeed
kubectl get trainjob 
kubectl delete trainjob llama-3-8b
envsubst '$INSTANCE_TYPE $NODE_NUM $GPU_PER_NODE $EFA_PER_NODE $HF_TOKEN' < trainjob.yaml | kubectl apply -f - 
```

훈련 작업에 참여하는 파드 리스트를 조회한다. 
```
kubectl get pods
```
[결과]
```
NAME                        READY   STATUS              RESTARTS   AGE
llama-3-8b-node-0-0-k9rb7   1/1     Running            0               6m14s
llama-3-8b-node-0-1-7rkwd   1/1     Running            0               6m14s
llama-3-8b-node-0-2-d6564   1/1     Running            0               6m14s
llama-3-8b-node-0-3-prtxr   1/1     Running            0               6m14s
```
Pod 상태 상세정보 및 이벤트(Events)를 확인한다. 설정오류 및 기타 원인으로 인해 컨테이너가 작업을 시작하지 못하는 경우 그 원인을 쉽게 파악할 수 있다.  
```
kubectl describe pod llama-3-8b-node-0-1-zf275
```

카펜터가 프로비저닝 한 노드 리스트를 조회한다. 만약 GPU 노드가 보이지 않는다면 카펜터 로그를 확인해 봐야 한다. 
```
kubectl get nodes -o custom-columns="NAME:.metadata.name, \
   STATUS:.status.conditions[?(@.type=='Ready')].status, \
   INSTANCE:.metadata.labels['node\.kubernetes\.io/instance-type'], \
   ARCH:.status.nodeInfo.architecture, \
   GPU:.status.capacity['nvidia\.com/gpu'], \
   EFA:.status.capacity['vpc\.amazonaws\.com/efa'], \
   ZONE:.metadata.labels['topology\.kubernetes\.io/zone'], \
   CAPACITY:.metadata.labels['karpenter\.sh/capacity-type']" \
| sed 's/\.ap-northeast-2\.compute\.internal//g' | column -t
```
[결과]
```
NAME           STATUS   INSTANCE      ARCH   GPU     EFA     ZONE             CAPACITY
ip-10-0-4-157  Unknown  g6e.12xlarge  amd64  4       1       ap-northeast-2a  on-demand
ip-10-0-4-27   Unknown  g6e.12xlarge  amd64  4       1       ap-northeast-2a  on-demand
ip-10-0-5-202  True     g6e.12xlarge  amd64  4       1       ap-northeast-2b  on-demand
ip-10-0-5-226  True     g6e.12xlarge  amd64  4       1       ap-northeast-2b  spot
ip-10-0-5-238  True     c7g.2xlarge   arm64  <none>  <none>  ap-northeast-2b  <none>
ip-10-0-5-37   Unknown  g6e.12xlarge  amd64  4       1       ap-northeast-2b  on-demand
ip-10-0-5-38   True     g6e.12xlarge  amd64  4       1       ap-northeast-2b  on-demand
ip-10-0-5-41   Unknown  g6e.12xlarge  amd64  4       1       ap-northeast-2b  on-demand
ip-10-0-5-61   True     c6i.2xlarge   amd64  <none>  <none>  ap-northeast-2b  <none>
ip-10-0-7-12   True     c6i.2xlarge   amd64  <none>  <none>  ap-northeast-2d  <none>
ip-10-0-7-56   True     c7g.2xlarge   arm64  <none>  <none>  ap-northeast-2d  <none>
```


### 훈련 모니터링 ###

#### Job Pod 관찰 ###
k9s 로 TrainJob 의 파드들이 실행되고 있는지 확인한다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/k9s.png)

#### 마스터 (rank 0) 관찰 ####
```
kubectl logs -f llama-3-8b-node-0-0-k9rb7
```
[결과]
```
Cloning into '/workspace/code'...
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.
The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `hf`CLI if you want to set the git credential as well.
Token is valid (permission: read).
The token `training-on-eks` has been saved to /root/.cache/huggingface/stored_tokens
Your token has been saved to /root/.cache/huggingface/token
Login successful.
The current active token is: `training-on-eks`
=== Launching Distributed Training ===
Requirement already satisfied: torch>=2.4.0 in /usr/local/lib/python3.12/site-packages (from -r requirements.txt (line 2)) (2.8.0+cu129)
Requirement already satisfied
...

{'train_runtime': 2307.9654, 'train_samples_per_second': 5.546, 'train_steps_per_second': 0.022, 'train_loss': 2.096866226196289, 'epoch': 0.35, 'time': '02:25:32'}
100%|██████████| 50/50 [38:27<00:00, 46.16s/it]

[학습 종료 보고서]
최종 소요 시간: 0:40:41
전체 초 단위: 2441.04s

--- 학습 완료! 최종 모델 저장 중 ---
모델과 토크나이저가 저장되었습니다: /data/fsx/final_model
llama-3-8b-node-0-0:191:637 [0] NCCL INFO [Service thread] Connection closed by localRank 1
llama-3-8b-node-0-0:193:634 [0] NCCL INFO [Service thread] Connection closed by localRank 1
llama-3-8b-node-0-0:191:1071 [0] NCCL INFO [Service thread] Connection closed by localRank 1
llama-3-8b-node-0-0:193:1069 [0] NCCL INFO [Service thread] Connection closed by localRank 1
llama-3-8b-node-0-0:194:635 [3] NCCL INFO [Service thread] Connection closed by localRank 2
llama-3-8b-node-0-0:194:1072 [3] NCCL INFO [Service thread] Connection closed by localRank 2
```

 
## 레퍼런스 ##
* [Simple DeepSpeed](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c10-deepspeed-simple.md)


