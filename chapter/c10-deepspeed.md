## DeepSpeed Stage3 ##
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


