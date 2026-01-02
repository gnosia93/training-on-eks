#!/bin/bash
# 1. HF_TOKEN 환경 변수 체크
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN 환경 변수가 설정되지 않았습니다."
    echo "Llama-3 모델 학습을 위해 Hugging Face 토큰이 필요합니다."
    echo "export HF_TOKEN='your_token_here' 명령으로 토큰을 설정해주세요."
    exit 1  # 스크립트 실행 중단 (종료)
fi

# export NCCL_TOPO_FILE=""
#export NCCL_P2P_LEVEL="NVL"                          # P2P 경로를 NVLink로 강제
#export NCCL_DEBUG_SUBSYS="GRAPH,INIT,ENV"

export CUDA_VISIBLE_DEVICES=0,1,2,3  # 8대 중 4대만 사용
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1             # AWS 환경에서 종종 필요
export NCCL_FI_ISV_ENB=0             # EFA(libfabric) 사용 안 함

export MASTER_ADDR=localhost
export MASTER_PORT=29500   

echo "current working directory: $(pwd)"        
# pip install -r requirements.txt 
huggingface-cli login --token "${HF_TOKEN}"
echo "=== Launching Distributed Training ==="
torchrun \
  --log-dir=./training_logs \
  --nproc_per_node=4 \
  --rdzv_id=llama-3-8b-job \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_conf=timeout=60 \
  llama-3-8b.py 
