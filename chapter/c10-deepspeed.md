![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/deepspeed-logo.svg) 마이크로소프트(Microsoft)가 개발한 [DeepSpeed](https://www.deepspeed.ai/)는 ZeRO 기술을 통해 GPU 메모리 점유율을 획기적으로 낮추고, 여러 연산을 하나로 묶는 커널 퓨전(Kernel Fusion) 및 커스텀 커널 최적화로 연산 속도를 극대화하는 딥러닝 가속 엔진이다. DeepSpeed 공식 문서에 따르면, 이 라이브러리는 하드웨어 한계를 넘어선 초거대 AI 모델의 학습과 추론을 지원하며, 특히 AWS EFA와 같은 고성능 네트워크 환경에서 통신 효율을 최대로 끌어올린다. 적은 컴퓨팅 자원으로도 대규모 모델을 가장 빠르고 효율적으로 구동할 수 있게 해주는 최적화 프레임워크 이다.

## Llama-3-8B ##
* [Llama-3-8B](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/llama-3-8b.py)
* [Llama-3-8B Config](https://github.com/gnosia93/training-on-eks/blob/main/samples/deepspeed/llama-3-8b.json)
  
### 훈련 설정값 ###
* gradient_checkpointing=True
역전파 시 필요한 중간 연산 결과를 저장하지 않고 다시 계산하여 메모리 사용량을 줄임(8B 이상의 모델에서는 필수)
* bf16=True
FP16은 값이 갑자기 커지면(Overflow) 학습이 망가질 수 있는데, BFloat16은 표현 범위가 넓어 안정적임.
* offload_param & offload_optimizer
Stage 3 설정 중 offload를 활성화하면, GPU 메모리가 가득 찼을 때 모델 파라미터를 CPU RAM으로 자미 이동함. 학습 속도는 느려지지만 훨씬 더 큰 모델을 학습할 수 있음.
* meta device 초기화
수십 GB의 모델을 한 GPU가 먼저 다 읽으려 하면 시작하자마자 OOM 발생함. AutoModel.from_config 사용하면 모델을 실제 메모리에 올리기 전에 구조만 파악하고, DeepSpeed가 각 GPU에 쪼개서 로드하도록 유도.

### g6e.8xlarge EFA 사양 ###
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

### 큐브플로우 Trainer 설치 ###
```
sudo dnf install git -y

export VERSION=v2.1.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"

kubectl get pods -n kubeflow-system
kubectl get clustertrainingruntimes
```

### 훈련 시작 ###
```
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export INSTANCE_TYPE=g6e.8xlarge              
export AZ=${AWS_REGION}a                 
export NODE_NUM=4                     # g6e.8xlarge 4대 
export GPU_NUM=1                      # g6e.8xlarge 타입은 GPU 가 1장이다.
export EFA_NUM=8                      # 200Gbp 사용

cd ~/training-on-eks/samples/deepspeed
kubectl apply -f trainjob.yaml

kubectl exec -it llama-3-8b -- /bin/bash
fi_info -p efa
```
 
## 레퍼런스 ##
* [Simple DeepSpeed](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c10-deepspeed-simple.md)


