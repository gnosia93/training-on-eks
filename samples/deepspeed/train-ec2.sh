#!/bin/bash
export NCCL_TOPO_FILE=""
export NCCL_P2P_LEVEL="NVL"                          # P2P 경로를 NVLink로 강제
export NCCL_DEBUG="INFO"
export NCCL_DEBUG_SUBSYS="GRAPH,INIT,ENV"
export MASTER_ADDR=localhost
export MASTER_PORT=29500   

echo "current working directory: $(pwd)"        
huggingface-cli login --token "${HF_TOKEN}"
echo "=== Launching Distributed Training ==="
pip install -r requirements.txt && \
echo "pip 설치 완료, 학습을 시작합니다..." && \
echo "Master Address: ${PET_MASTER_ADDR}" && \
echo "Master Port: ${PET_MASTER_PORT}" && \



torchrun \
  --nproc_per_node=4 \
  --rdzv_id=llama-3-8b-job \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_conf=timeout=1200 \
  llama-3-8b.py 
