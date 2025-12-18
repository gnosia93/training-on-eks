
## 성능 측정 ##

### Nitro V4 / NVIDIA L4 ###
* g6.48xlarge / 24GB * 8 GPU / PCIe / 1 Node  
* g6.16xlarge / 24GB * 1 GPU / ENA / 8 Node
* g6.16xlarge / 24GB * 1 GPU / EFA / 8 Node

### Nitro / NVIDIA L40S  ###
* g6e.48xlarge / 24GB * 8 GPU / PCIe / 1 Node  
* g6e.16xlarge / 48GB * 1 CPU / ENA / 8 Node
* g6e.16xlarge / 48GB * 1 CPU / EFA / 8 Node

### Nitro / NVIDIA A100  ###
* p4d.24xlarge.24xlarge / 40GB * 8 GPU / NLINK / 1 Node  
* p4d.24xlarge.24xlarge / 40GB * 1 GPU / ENA / 8 Node (VISIBLE_DEVICES=0)
* p4d.24xlarge.24xlarge / 40GB * 1 GPU / EFA / 8 Node (VISIBLE_DEVICES=0)


## 레퍼런스 ##
* https://aws.amazon.com/ko/ec2/instance-types/

