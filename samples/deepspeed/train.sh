# 1. HF_TOKEN 환경 변수 체크
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN 환경 변수가 설정되지 않았습니다."
    echo "Llama-3 모델 학습을 위해 Hugging Face 토큰이 필요합니다."
    echo "export HF_TOKEN='your_token_here' 명령으로 토큰을 설정해주세요."
    exit 1  # 스크립트 실행 중단 (종료)
fi

export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export INSTANCE_TYPE=g6e.12xlarge              # 훈련 인스턴스 타입   (g6e.48xlarge)
export AZ=${AWS_REGION}a                 
export NODE_NUM=4                              # 4대 
export GPU_PER_NODE=4                          # g6e.12xlarge 타입은 GPU 가 4장이다.
export EFA_PER_NODE=1                          # 100Gbp 사용
# export HF_TOKEN="<your huggingface token>"     # Llama-3 모델은 HF 인증이 필요.

cd ~/training-on-eks/samples/deepspeed
kubectl get trainjob 
kubectl delete trainjob llama-3-8b --ignore-not-found=true # 작업이 없어도 에러 없이 진행
envsubst '$INSTANCE_TYPE $NODE_NUM $GPU_PER_NODE $EFA_PER_NODE $HF_TOKEN' < trainjob-1to1.yaml | kubectl apply -f - 

kubectl get trainjob 
kubectl get pods -o wide
kubectl get nodes

kubectl get nodes -o custom-columns="NAME:.metadata.name, \
   STATUS:.status.conditions[?(@.type=='Ready')].status, \
   INSTANCE:.metadata.labels['node\.kubernetes\.io/instance-type'], \
   ARCH:.status.nodeInfo.architecture, \
   GPU:.status.capacity['nvidia\.com/gpu'], \
   EFA:.status.capacity['vpc\.amazonaws\.com/efa'], \
   ZONE:.metadata.labels['topology\.kubernetes\.io/zone'], \
   CAPACITY:.metadata.labels['karpenter\.sh/capacity-type']" \
| sed 's/\.ap-northeast-2\.compute\.internal//g' | column -t
