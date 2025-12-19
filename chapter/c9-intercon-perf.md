```
cd training-on-eks

kubectl apply -k kustomize/overlays/fsdp/

kubectl get nodes -L node.kubernetes.io/instance-type
```

## PCIe / ENA / EFA Performance ##

### Nitro V4 / NVIDIA L4 ###
* g6.48xlarge / 24GB * 8 GPU / PCIe / 1 Node - Elapsed Time 30m
* g6.16xlarge / 24GB * 1 GPU / ENA / 8 Node
* g6.16xlarge / 24GB * 1 GPU / EFA / 8 Node

### Nitro / NVIDIA A100  ###
* p4d.24xlarge.24xlarge / 40GB * 8 GPU / NLINK / 1 Node  
* p4d.24xlarge.24xlarge / 40GB * 1 GPU / ENA / 8 Node (VISIBLE_DEVICES=0)
* p4d.24xlarge.24xlarge / 40GB * 1 GPU / EFA / 8 Node (VISIBLE_DEVICES=0)

## Grace Hopper / Blackwell --> CPU/GPU NVLink ## 
* Parameter CPU Offloading 성능 (NVLink vs PCIe)
* PCIe(System Interconnect) 대역폭 병목 (Bottleneck) 회피.
* DeepSpeed and FSDP 테스트.
    
## 싱글 GPU 연산성능 측정 ##
* Bert Base 110M with G/P Types
* CUDA_VISIBLE_DEVICES = 0 (멀티 GPU인 경우)
* 연산 속도, OOM, 최대 배치사이즈, GPU Utilization.

## 레퍼런스 ##
* https://aws.amazon.com/ko/ec2/instance-types/

