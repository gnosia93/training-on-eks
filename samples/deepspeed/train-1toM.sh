# 1. HF_TOKEN 환경 변수 체크
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN 환경 변수가 설정되지 않았습니다."
    echo "Llama-3 모델 학습을 위해 Hugging Face 토큰이 필요합니다."
    echo "export HF_TOKEN='your_token_here' 명령으로 토큰을 설정해주세요."
    exit 1  # 스크립트 실행 중단 (종료)
fi

cd ~/training-on-eks/samples/deepspeed
kubectl get trainjob 
kubectl delete trainjob llama-3-8b --ignore-not-found=true # 작업이 없어도 에러 없이 진행
envsubst '$HF_TOKEN' < trainjob-1toM.yaml | kubectl apply -f - 

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
