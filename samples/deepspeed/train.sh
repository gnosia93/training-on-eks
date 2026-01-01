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
envsubst '$INSTANCE_TYPE $NODE_NUM $GPU_PER_NODE $EFA_PER_NODE $HF_TOKEN' < trainjob-1to1.yaml | kubectl apply -f - 

kubectl get trainjob 
kubectl get pods 
