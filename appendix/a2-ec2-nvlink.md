## EC2 NVLINK / NCCL 테스트 ##

### 인스턴스 생성 ###
```
export AWS_REGION="ap-northeast-2"
export AZ="${AWS_REGION}b"
export KEY_NAME="aws-kp-2"
export INSTANCE_TYPE="p4d.24xlarge"

MY_IP=$(curl -s https://checkip.amazonaws.com)/32

VPC_ID=$(aws ec2 describe-vpcs --region ${AWS_REGION} \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)

SUBNET_ID=$(aws ec2 describe-subnets --region ${AWS_REGION} \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$AZ" \
    --query 'Subnets[0].SubnetId' --output text)

echo "현재 접속 IP : ${MY_IP}"
echo "AWS REGION / VPC / AZ / 서브넷 : ${AWS_REGION} / ${VPC_ID} / ${AZ} / ${SUBNET_ID}"
echo "------------------------------------------------"

# 최신 Deep Learning AMI ID 조회 (PyTorch 지원 OSS Nvidia Driver 버전)
AMI_ID=$(aws ec2 describe-images \
    --region ${AWS_REGION} \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023)*" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [[ "$AMI_ID" == "None"  || -z "$AMI_ID" ]]; then
    echo "AMI를 찾을 수 없습니다. 리전 ID를 확인해주세요."
    exit 1
fi
echo "조회된 AMI ID: ${AMI_ID}"

# 3. 보안 그룹 생성
SG_ID=$(aws ec2 create-security-group \
    --region ${AWS_REGION} \
    --group-name "P4-PyTorch-SG-$(date +%s)" \
    --description "SG for P4 with VSCode" \
    --vpc-id ${VPC_ID} --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
    --region ${AWS_REGION} --group-id ${SG_ID} \
    --protocol tcp --port 22 --cidr ${MY_IP}

aws ec2 authorize-security-group-ingress \
    --region ${AWS_REGION} --group-id ${SG_ID} \
    --protocol tcp --port 80 --cidr ${MY_IP}

# 4. User Data 작성 (VSCode CLI 설치)
cat <<EOF > userdata.sh
#!/bin/bash
sudo growpart /dev/nvme0n1 1
sudo xfs_growfs -d /

dnf update -y
dnf install -y nginx

curl -fsSL code-server.dev | sh
systemctl enable --now code-server@ec2-user

cat <<EOF > /etc/nginx/conf.d/code-server.conf
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Accept-Encoding gzip;
    }
}
EOF

# 5. 인스턴스 실행 (P4.24xlarge)
echo "인스턴스를 생성 중입니다..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region ${AWS_REGION} \
    --image-id ${AMI_ID} \
    --count 1 \
    --instance-type ${INSTANCE_TYPE} \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SG_ID} \
    --subnet-id ${SUBNET_ID} \
    --user-data file://userdata.sh \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=P4-PyTorch-DevServer},{Key=Project,Value=ML-Project-2026}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "------------------------------------------------"
echo "인스턴스 ID: $INSTANCE_ID"
echo "성공적으로 요청되었습니다. AWS 콘솔에서 상태를 확인하세요."

aws ec2 describe-instances \
    --region ${AWS_REGION} \
    --instance-ids ${INSTANCE_ID} \
    --query 'Reservations[*].Instances[*].PublicDnsName' \
    --output text
```

### 터미널 로그인 ###
```
source /opt/pytorch/bin/activate
nvidia-smi 
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}')"
```

### 토폴로지 확인 ###
```
nvidia-smi topo -m
```
[결과]
```
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NV12	NV12	NV12	NV12	NV12	NV12	NV12	0-23,48-71	0		N/A
GPU1	NV12	 X 	NV12	NV12	NV12	NV12	NV12	NV12	0-23,48-71	0		N/A
GPU2	NV12	NV12	 X 	NV12	NV12	NV12	NV12	NV12	0-23,48-71	0		N/A
GPU3	NV12	NV12	NV12	 X 	NV12	NV12	NV12	NV12	0-23,48-71	0		N/A
GPU4	NV12	NV12	NV12	NV12	 X 	NV12	NV12	NV12	24-47,72-95	1		N/A
GPU5	NV12	NV12	NV12	NV12	NV12	 X 	NV12	NV12	24-47,72-95	1		N/A
GPU6	NV12	NV12	NV12	NV12	NV12	NV12	 X 	NV12	24-47,72-95	1		N/A
GPU7	NV12	NV12	NV12	NV12	NV12	NV12	NV12	 X 	24-47,72-95	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

## 분산훈련 ##
### 사전준비 ###
```
# 파이썬 가상 워크스페이스
source /opt/pytorch/bin/activate

sudo mkdir -p /data
sudo chown ec2-user:ec2-user /data
sudo dnf update -y
sudo dnf install python3-pip -y

# AMI 크기가 15기가라서, 300GB의 공간을 할당했지만 아래 명령어로 그 크기를 늘려줘야한다.. 
df -m
sudo growpart /dev/nvme0n1 1
sudo xfs_growfs -d /
df -m

# Add CUDA libraries to the linker path
# Ensure DeepSpeed knows where CUDA is installed
# libcurand.so.10을 가리키는 libcurand.so 링크 생성
export CUDA_HOME=/opt/pytorch/cuda
export LIBRARY_PATH=$LIBRARY_PATH:${CUDA_HOME}/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib
cd /opt/pytorch/cuda/lib
ln -s libcurand.so.10 libcurand.so

git clone https://github.com/gnosia93/training-on-eks.git
cd ~/training-on-eks/samples/deepspeed
pip install -r requirements.txt
```

### 싱글 노드 훈련 ###
```
# 처음에는 프로세스 하나로 돌려준다. 이렇게 하는 이유는 2개 이상을 프로세스로 기동하는 경우 Adam 옵티마이저 관련 Lock 발생해서 Hang 이 걸린다.
# 첫 실행시 Adam 옵티마이저를 컴파일 하는 듯 하다..
export HF_TOKEN=<your token>
torchrun --nproc_per_node=1 llama-3-8b.py
```
[결과]
```
[학습 종료 보고서]
최종 소요 시간: 0:22:45
전체 초 단위: 1365.22s
```

### 멀티 노드 훈련 ###
4 노드 분산 훈련을 시작한다. 
```
# 컴파일 또는 실행이 완료되면 분산 훈련을 시작한다. 
sh train-ec2.sh
```
![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/ec2-train.png)

#### nccl log ####
```
max_steps is given, it will override any value given in num_train_epochs
Using auto half precision backend
Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4. Using DeepSpeed's value.
Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)
PyTorch: setting up devices
Adam Optimizer #0 is created with AVX512 arithmetic capability.
Config: alpha=0.000020, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
ip-172-31-3-153:35113:35113 [0] NCCL INFO Bootstrap: Using ens32:172.31.3.153<0>
ip-172-31-3-153:35113:35113 [0] NCCL INFO cudaDriverVersion 13000
ip-172-31-3-153:35113:35113 [0] NCCL INFO NCCL version 2.27.7+cuda13.0
ip-172-31-3-153:35114:35114 [1] NCCL INFO cudaDriverVersion 13000
ip-172-31-3-153:35115:35115 [2] NCCL INFO cudaDriverVersion 13000
ip-172-31-3-153:35113:35113 [0] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35114:35114 [1] NCCL INFO Bootstrap: Using ens32:172.31.3.153<0>
ip-172-31-3-153:35114:35114 [1] NCCL INFO NCCL version 2.27.7+cuda13.0
ip-172-31-3-153:35115:35115 [2] NCCL INFO Bootstrap: Using ens32:172.31.3.153<0>
ip-172-31-3-153:35115:35115 [2] NCCL INFO NCCL version 2.27.7+cuda13.0
ip-172-31-3-153:35114:35114 [1] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35115:35115 [2] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
ip-172-31-3-153:35115:35469 [2] NCCL INFO Successfully loaded external plugin libnccl-net.so
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.17.1
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Using Libfabric version 2.3
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 13000
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Configuring AWS-specific options
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Setting provider_filter to efa
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Running on p4d.24xlarge platform, topology file /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Internode latency set at 75.0 us
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI Using transport protocol SENDRECV (platform set)
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
ip-172-31-3-153:35113:35467 [0] NCCL INFO Successfully loaded external plugin libnccl-net.so
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.17.1
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Using Libfabric version 2.3
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 13000
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Configuring AWS-specific options
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Setting provider_filter to efa
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Running on p4d.24xlarge platform, topology file /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Internode latency set at 75.0 us
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI Using transport protocol SENDRECV (platform set)
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
ip-172-31-3-153:35114:35468 [1] NCCL INFO Successfully loaded external plugin libnccl-net.so
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.17.1
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Using Libfabric version 2.3
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 13000
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Configuring AWS-specific options
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Setting provider_filter to efa
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Running on p4d.24xlarge platform, topology file /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Internode latency set at 75.0 us
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI Using transport protocol SENDRECV (platform set)

[2026-01-02 10:24:27] ip-172-31-3-153:35115:35469 [2] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):208 NCCL WARN NET/OFI Failed to initialize sendrecv protocol

[2026-01-02 10:24:27] ip-172-31-3-153:35114:35468 [1] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):208 NCCL WARN NET/OFI Failed to initialize sendrecv protocol

[2026-01-02 10:24:27] ip-172-31-3-153:35115:35469 [2] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):353 NCCL WARN NET/OFI aws-ofi-nccl initialization failed

[2026-01-02 10:24:27] ip-172-31-3-153:35114:35468 [1] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):353 NCCL WARN NET/OFI aws-ofi-nccl initialization failed

[2026-01-02 10:24:27] ip-172-31-3-153:35114:35468 [1] ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t):162 NCCL WARN NET/OFI Initializing plugin failed

[2026-01-02 10:24:27] ip-172-31-3-153:35115:35469 [2] ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t):162 NCCL WARN NET/OFI Initializing plugin failed
ip-172-31-3-153:35115:35469 [2] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
ip-172-31-3-153:35114:35468 [1] NCCL INFO NCCL_IB_DISABLE set by environment to 1.

[2026-01-02 10:24:27] ip-172-31-3-153:35113:35467 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):208 NCCL WARN NET/OFI Failed to initialize sendrecv protocol

[2026-01-02 10:24:27] ip-172-31-3-153:35113:35467 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):353 NCCL WARN NET/OFI aws-ofi-nccl initialization failed

[2026-01-02 10:24:27] ip-172-31-3-153:35113:35467 [0] ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t):162 NCCL WARN NET/OFI Initializing plugin failed
ip-172-31-3-153:35113:35467 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/Socket : Using [0]ens32:172.31.3.153<0>
ip-172-31-3-153:35113:35467 [0] NCCL INFO Initialized NET plugin Socket
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/Socket : Using [0]ens32:172.31.3.153<0>
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/Socket : Using [0]ens32:172.31.3.153<0>
ip-172-31-3-153:35114:35468 [1] NCCL INFO Initialized NET plugin Socket
ip-172-31-3-153:35115:35469 [2] NCCL INFO Initialized NET plugin Socket
ip-172-31-3-153:35113:35467 [0] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35113:35467 [0] NCCL INFO Using network Socket
ip-172-31-3-153:35114:35468 [1] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35115:35469 [2] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35114:35468 [1] NCCL INFO Using network Socket
ip-172-31-3-153:35115:35469 [2] NCCL INFO Using network Socket
ip-172-31-3-153:35116:35116 [3] NCCL INFO cudaDriverVersion 13000
ip-172-31-3-153:35116:35116 [3] NCCL INFO Bootstrap: Using ens32:172.31.3.153<0>
ip-172-31-3-153:35116:35116 [3] NCCL INFO NCCL version 2.27.7+cuda13.0
ip-172-31-3-153:35116:35116 [3] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35113:35467 [0] NCCL INFO ncclCommInitRankConfig comm 0x55d7869cef50 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101c0 commId 0x52e965372ad155ed - Init START
ip-172-31-3-153:35115:35469 [2] NCCL INFO ncclCommInitRankConfig comm 0x55b49c307cc0 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 201c0 commId 0x52e965372ad155ed - Init START
ip-172-31-3-153:35114:35468 [1] NCCL INFO ncclCommInitRankConfig comm 0x55656a8e7180 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 101d0 commId 0x52e965372ad155ed - Init START
ip-172-31-3-153:35114:35468 [1] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO Successfully loaded external plugin libnccl-net.so
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.17.1
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Using Libfabric version 2.3
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 13000
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Configuring AWS-specific options
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Setting provider_filter to efa
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Running on p4d.24xlarge platform, topology file /opt/amazon/ofi-nccl/share/aws-ofi-nccl/xml/p4d-24xl-topo.xml
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Internode latency set at 75.0 us
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI Using transport protocol SENDRECV (platform set)

[2026-01-02 10:24:28] ip-172-31-3-153:35116:35475 [3] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):208 NCCL WARN NET/OFI Failed to initialize sendrecv protocol

[2026-01-02 10:24:28] ip-172-31-3-153:35116:35475 [3] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):353 NCCL WARN NET/OFI aws-ofi-nccl initialization failed

[2026-01-02 10:24:28] ip-172-31-3-153:35116:35475 [3] ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t):162 NCCL WARN NET/OFI Initializing plugin failed
ip-172-31-3-153:35116:35475 [3] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/Socket : Using [0]ens32:172.31.3.153<0>
ip-172-31-3-153:35116:35475 [3] NCCL INFO Initialized NET plugin Socket
ip-172-31-3-153:35116:35475 [3] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35116:35475 [3] NCCL INFO Using network Socket
ip-172-31-3-153:35116:35475 [3] NCCL INFO ncclCommInitRankConfig comm 0x56099d2af800 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 201d0 commId 0x52e965372ad155ed - Init START
ip-172-31-3-153:35116:35475 [3] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
ip-172-31-3-153:35115:35469 [2] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
ip-172-31-3-153:35113:35467 [0] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
ip-172-31-3-153:35113:35467 [0] NCCL INFO Bootstrap timings total 0.147981 (create 0.000035, send 0.000155, recv 0.000583, ring 0.000067, delay 0.000000)
ip-172-31-3-153:35116:35475 [3] NCCL INFO Bootstrap timings total 0.001161 (create 0.000037, send 0.000101, recv 0.000229, ring 0.000088, delay 0.000000)
ip-172-31-3-153:35115:35469 [2] NCCL INFO Bootstrap timings total 0.147769 (create 0.000046, send 0.000113, recv 0.146781, ring 0.000081, delay 0.000000)
ip-172-31-3-153:35114:35468 [1] NCCL INFO Bootstrap timings total 0.147587 (create 0.000037, send 0.000097, recv 0.000218, ring 0.146478, delay 0.000000)
ip-172-31-3-153:35115:35469 [2] NCCL INFO Setting affinity for GPU 2 to 0-23,48-71
ip-172-31-3-153:35115:35469 [2] NCCL INFO NVLS multicast support is not available on dev 2 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35116:35475 [3] NCCL INFO Setting affinity for GPU 3 to 0-23,48-71
ip-172-31-3-153:35116:35475 [3] NCCL INFO NVLS multicast support is not available on dev 3 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35114:35468 [1] NCCL INFO Setting affinity for GPU 1 to 0-23,48-71
ip-172-31-3-153:35114:35468 [1] NCCL INFO NVLS multicast support is not available on dev 1 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35113:35467 [0] NCCL INFO Setting affinity for GPU 0 to 0-23,48-71
ip-172-31-3-153:35113:35467 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35113:35467 [0] NCCL INFO comm 0x55d7869cef50 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
ip-172-31-3-153:35116:35475 [3] NCCL INFO comm 0x56099d2af800 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
ip-172-31-3-153:35114:35468 [1] NCCL INFO comm 0x55656a8e7180 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
ip-172-31-3-153:35115:35469 [2] NCCL INFO comm 0x55b49c307cc0 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 00/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 01/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 02/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 03/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 04/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 05/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 06/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 07/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 08/24 : 0 1 2 3
ip-172-31-3-153:35116:35475 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2 [2] -1/-1/-1->3->2 [3] -1/-1/-1->3->2 [4] -1/-1/-1->3->2 [5] -1/-1/-1->3->2 [6] -1/-1/-1->3->2 [7] -1/-1/-1->3->2 [8] -1/-1/-1->3->2 [9] -1/-1/-1->3->2 [10] -1/-1/-1->3->2 [11] -1/-1/-1->3->2 [12] -1/-1/-1->3->2 [13] -1/-1/-1->3->2 [14] -1/-1/-1->3->2 [15] -1/-1/-1->3->2 [16] -1/-1/-1->3->2 [17] -1/-1/-1->3->2 [18] -1/-1/-1->3->2 [19] -1/-1/-1->3->2 [20] -1/-1/-1->3->2 [21] -1/-1/-1->3->2 [22] -1/-1/-1->3->2 [23] -1/-1/-1->3->2
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 09/24 : 0 1 2 3
ip-172-31-3-153:35114:35468 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->0 [6] 2/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->0 [10] 2/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->0 [14] 2/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->0 [18] 2/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->0 [22] 2/-1/-1->1->0 [23] 2/-1/-1->1->0
ip-172-31-3-153:35115:35469 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->1 [7] 3/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->1 [11] 3/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->1 [15] 3/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->1 [19] 3/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->1 [23] 3/-1/-1->2->1
ip-172-31-3-153:35116:35475 [3] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 10/24 : 0 1 2 3
ip-172-31-3-153:35114:35468 [1] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35115:35469 [2] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 11/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 12/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 13/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 14/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 15/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 16/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 17/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 18/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 19/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 20/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 21/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 22/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Channel 23/24 : 0 1 2 3
ip-172-31-3-153:35113:35467 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] 1/-1/-1->0->-1 [5] 1/-1/-1->0->-1 [6] 1/-1/-1->0->-1 [7] 1/-1/-1->0->-1 [8] 1/-1/-1->0->-1 [9] 1/-1/-1->0->-1 [10] 1/-1/-1->0->-1 [11] 1/-1/-1->0->-1 [12] 1/-1/-1->0->-1 [13] 1/-1/-1->0->-1 [14] 1/-1/-1->0->-1 [15] 1/-1/-1->0->-1 [16] 1/-1/-1->0->-1 [17] 1/-1/-1->0->-1 [18] 1/-1/-1->0->-1 [19] 1/-1/-1->0->-1 [20] 1/-1/-1->0->-1 [21] 1/-1/-1->0->-1 [22] 1/-1/-1->0->-1 [23] 1/-1/-1->0->-1
ip-172-31-3-153:35113:35467 [0] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35115:35469 [2] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
ip-172-31-3-153:35114:35468 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
ip-172-31-3-153:35115:35480 [2] NCCL INFO [Proxy Service] Device 2 CPU core 66
ip-172-31-3-153:35115:35481 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 68
ip-172-31-3-153:35113:35467 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
ip-172-31-3-153:35113:35467 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
ip-172-31-3-153:35116:35475 [3] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
ip-172-31-3-153:35114:35482 [1] NCCL INFO [Proxy Service] Device 1 CPU core 7
ip-172-31-3-153:35114:35483 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 8
ip-172-31-3-153:35113:35484 [0] NCCL INFO [Proxy Service] Device 0 CPU core 1
ip-172-31-3-153:35116:35485 [3] NCCL INFO [Proxy Service] Device 3 CPU core 71
ip-172-31-3-153:35113:35486 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 50
ip-172-31-3-153:35116:35487 [3] NCCL INFO [Proxy Service UDS] Device 3 CPU core 58
ip-172-31-3-153:35116:35475 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35116:35475 [3] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35114:35468 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35114:35468 [1] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35113:35467 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35113:35467 [0] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35115:35469 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35115:35469 [2] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35113:35467 [0] NCCL INFO CC Off, workFifoBytes 1048576
ip-172-31-3-153:35116:35475 [3] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
ip-172-31-3-153:35113:35467 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
ip-172-31-3-153:35114:35468 [1] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
ip-172-31-3-153:35115:35469 [2] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
ip-172-31-3-153:35113:35467 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
ip-172-31-3-153:35113:35467 [0] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
ip-172-31-3-153:35116:35475 [3] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
ip-172-31-3-153:35116:35475 [3] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35113:35467 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35114:35468 [1] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
ip-172-31-3-153:35116:35475 [3] NCCL INFO ncclCommInitRankConfig comm 0x56099d2af800 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 201d0 commId 0x52e965372ad155ed - Init COMPLETE
ip-172-31-3-153:35114:35468 [1] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
ip-172-31-3-153:35113:35467 [0] NCCL INFO ncclCommInitRankConfig comm 0x55d7869cef50 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101c0 commId 0x52e965372ad155ed - Init COMPLETE
ip-172-31-3-153:35115:35469 [2] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
ip-172-31-3-153:35114:35468 [1] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35116:35475 [3] NCCL INFO Init timings - ncclCommInitRankConfig: rank 3 nranks 4 total 0.25 (kernels 0.17, alloc 0.03, bootstrap 0.00, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35115:35469 [2] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
ip-172-31-3-153:35113:35467 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 0.49 (kernels 0.21, alloc 0.08, bootstrap 0.15, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35114:35468 [1] NCCL INFO ncclCommInitRankConfig comm 0x55656a8e7180 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 101d0 commId 0x52e965372ad155ed - Init COMPLETE
ip-172-31-3-153:35115:35469 [2] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35115:35469 [2] NCCL INFO ncclCommInitRankConfig comm 0x55b49c307cc0 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 201c0 commId 0x52e965372ad155ed - Init COMPLETE
ip-172-31-3-153:35114:35468 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 4 total 0.49 (kernels 0.21, alloc 0.08, bootstrap 0.15, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35115:35469 [2] NCCL INFO Init timings - ncclCommInitRankConfig: rank 2 nranks 4 total 0.49 (kernels 0.21, alloc 0.08, bootstrap 0.15, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 01/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 02/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 02/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 03/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 01/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 04/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 03/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 04/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 02/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 05/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 04/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 06/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 03/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 05/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 05/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 07/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 04/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 06/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 06/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 08/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 07/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 07/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 05/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 09/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 08/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 08/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 06/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 10/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 09/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 09/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 07/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 11/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 10/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 10/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 08/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 12/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 11/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 11/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 09/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 13/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 12/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 12/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 14/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 10/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 13/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 13/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 15/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 11/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 14/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 14/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 16/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 12/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 15/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 15/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 17/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 16/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 13/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 16/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 18/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 17/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 14/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 17/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 19/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 18/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 15/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 18/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 20/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 16/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 19/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 19/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 21/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 17/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 20/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 20/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 22/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 21/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 21/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 18/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35488 [0] NCCL INFO Channel 23/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 22/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 22/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 19/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35489 [3] NCCL INFO Channel 23/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Channel 23/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 20/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 21/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 22/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35115:35491 [2] NCCL INFO Channel 23/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35490 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
ip-172-31-3-153:35113:35488 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
ip-172-31-3-153:35116:35489 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
ip-172-31-3-153:35115:35491 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
Stage 3 initialize beginning
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 17.04 GB, percent = 1.5%
DeepSpeedZeRoOffload initialize [begin]
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 17.17 GB, percent = 1.5%
ip-172-31-3-153:35115:35115 [2] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35113:35113 [0] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35115:35497 [2] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35115:35497 [2] NCCL INFO Using network Socket
ip-172-31-3-153:35114:35114 [1] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35116:35116 [3] NCCL INFO Comm config Blocking set to 1
ip-172-31-3-153:35115:35497 [2] NCCL INFO ncclCommInitRankConfig comm 0x55b4ac447790 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 201c0 commId 0x3da1fa247351b1f5 - Init START
ip-172-31-3-153:35113:35498 [0] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35113:35498 [0] NCCL INFO Using network Socket
ip-172-31-3-153:35114:35499 [1] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35113:35498 [0] NCCL INFO ncclCommInitRankConfig comm 0x55d78e6827a0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101c0 commId 0x3da1fa247351b1f5 - Init START
ip-172-31-3-153:35116:35500 [3] NCCL INFO Assigned NET plugin Socket to comm
ip-172-31-3-153:35114:35499 [1] NCCL INFO Using network Socket
ip-172-31-3-153:35116:35500 [3] NCCL INFO Using network Socket
ip-172-31-3-153:35114:35499 [1] NCCL INFO ncclCommInitRankConfig comm 0x556572999110 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 101d0 commId 0x3da1fa247351b1f5 - Init START
ip-172-31-3-153:35116:35500 [3] NCCL INFO ncclCommInitRankConfig comm 0x5609a4f5f350 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 201d0 commId 0x3da1fa247351b1f5 - Init START
ip-172-31-3-153:35115:35497 [2] NCCL INFO Bootstrap timings total 0.001122 (create 0.000049, send 0.000184, recv 0.000444, ring 0.000071, delay 0.000000)
ip-172-31-3-153:35113:35498 [0] NCCL INFO Bootstrap timings total 0.000937 (create 0.000034, send 0.000079, recv 0.000555, ring 0.000094, delay 0.000000)
ip-172-31-3-153:35114:35499 [1] NCCL INFO Bootstrap timings total 0.000766 (create 0.000034, send 0.000095, recv 0.000421, ring 0.000060, delay 0.000000)
ip-172-31-3-153:35116:35500 [3] NCCL INFO Bootstrap timings total 0.000765 (create 0.000033, send 0.000095, recv 0.000253, ring 0.000216, delay 0.000000)
ip-172-31-3-153:35116:35500 [3] NCCL INFO Setting affinity for GPU 3 to 0-23,48-71
ip-172-31-3-153:35116:35500 [3] NCCL INFO NVLS multicast support is not available on dev 3 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35114:35499 [1] NCCL INFO Setting affinity for GPU 1 to 0-23,48-71
ip-172-31-3-153:35115:35497 [2] NCCL INFO Setting affinity for GPU 2 to 0-23,48-71
ip-172-31-3-153:35115:35497 [2] NCCL INFO NVLS multicast support is not available on dev 2 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35114:35499 [1] NCCL INFO NVLS multicast support is not available on dev 1 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35113:35498 [0] NCCL INFO Setting affinity for GPU 0 to 0-23,48-71
ip-172-31-3-153:35113:35498 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
ip-172-31-3-153:35115:35497 [2] NCCL INFO comm 0x55b4ac447790 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
ip-172-31-3-153:35116:35500 [3] NCCL INFO comm 0x5609a4f5f350 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
ip-172-31-3-153:35113:35498 [0] NCCL INFO comm 0x55d78e6827a0 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
ip-172-31-3-153:35114:35499 [1] NCCL INFO comm 0x556572999110 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 00/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 01/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 02/24 : 0 1 2 3
ip-172-31-3-153:35115:35497 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->1 [7] 3/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->1 [11] 3/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->1 [15] 3/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->1 [19] 3/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->1 [23] 3/-1/-1->2->1
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 03/24 : 0 1 2 3
ip-172-31-3-153:35116:35500 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2 [2] -1/-1/-1->3->2 [3] -1/-1/-1->3->2 [4] -1/-1/-1->3->2 [5] -1/-1/-1->3->2 [6] -1/-1/-1->3->2 [7] -1/-1/-1->3->2 [8] -1/-1/-1->3->2 [9] -1/-1/-1->3->2 [10] -1/-1/-1->3->2 [11] -1/-1/-1->3->2 [12] -1/-1/-1->3->2 [13] -1/-1/-1->3->2 [14] -1/-1/-1->3->2 [15] -1/-1/-1->3->2 [16] -1/-1/-1->3->2 [17] -1/-1/-1->3->2 [18] -1/-1/-1->3->2 [19] -1/-1/-1->3->2 [20] -1/-1/-1->3->2 [21] -1/-1/-1->3->2 [22] -1/-1/-1->3->2 [23] -1/-1/-1->3->2
ip-172-31-3-153:35115:35497 [2] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 04/24 : 0 1 2 3
ip-172-31-3-153:35114:35499 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->0 [6] 2/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->0 [10] 2/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->0 [14] 2/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->0 [18] 2/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->0 [22] 2/-1/-1->1->0 [23] 2/-1/-1->1->0
ip-172-31-3-153:35116:35500 [3] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 05/24 : 0 1 2 3
ip-172-31-3-153:35114:35499 [1] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 06/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 07/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 08/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 09/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 10/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 11/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 12/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 13/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 14/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 15/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 16/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 17/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 18/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 19/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 20/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 21/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 22/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Channel 23/24 : 0 1 2 3
ip-172-31-3-153:35113:35498 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] 1/-1/-1->0->-1 [5] 1/-1/-1->0->-1 [6] 1/-1/-1->0->-1 [7] 1/-1/-1->0->-1 [8] 1/-1/-1->0->-1 [9] 1/-1/-1->0->-1 [10] 1/-1/-1->0->-1 [11] 1/-1/-1->0->-1 [12] 1/-1/-1->0->-1 [13] 1/-1/-1->0->-1 [14] 1/-1/-1->0->-1 [15] 1/-1/-1->0->-1 [16] 1/-1/-1->0->-1 [17] 1/-1/-1->0->-1 [18] 1/-1/-1->0->-1 [19] 1/-1/-1->0->-1 [20] 1/-1/-1->0->-1 [21] 1/-1/-1->0->-1 [22] 1/-1/-1->0->-1 [23] 1/-1/-1->0->-1
ip-172-31-3-153:35113:35498 [0] NCCL INFO P2P Chunksize set to 524288
ip-172-31-3-153:35113:35498 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
ip-172-31-3-153:35113:35501 [0] NCCL INFO [Proxy Service] Device 0 CPU core 3
ip-172-31-3-153:35113:35502 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 4
ip-172-31-3-153:35115:35503 [2] NCCL INFO [Proxy Service] Device 2 CPU core 55
ip-172-31-3-153:35115:35504 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 8
ip-172-31-3-153:35116:35505 [3] NCCL INFO [Proxy Service] Device 3 CPU core 58
ip-172-31-3-153:35116:35506 [3] NCCL INFO [Proxy Service UDS] Device 3 CPU core 11
ip-172-31-3-153:35114:35507 [1] NCCL INFO [Proxy Service] Device 1 CPU core 12
ip-172-31-3-153:35114:35508 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 13
ip-172-31-3-153:35115:35497 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35115:35497 [2] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35113:35498 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35113:35498 [0] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35116:35500 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35116:35500 [3] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35113:35498 [0] NCCL INFO CC Off, workFifoBytes 1048576
ip-172-31-3-153:35114:35499 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ip-172-31-3-153:35114:35499 [1] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ip-172-31-3-153:35113:35498 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35115:35497 [2] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35113:35498 [0] NCCL INFO ncclCommInitRankConfig comm 0x55d78e6827a0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 101c0 commId 0x3da1fa247351b1f5 - Init COMPLETE
ip-172-31-3-153:35115:35497 [2] NCCL INFO ncclCommInitRankConfig comm 0x55b4ac447790 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 201c0 commId 0x3da1fa247351b1f5 - Init COMPLETE
ip-172-31-3-153:35116:35500 [3] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35114:35499 [1] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : p4d.24xlarge, Fall back to NCCL's tuner
ip-172-31-3-153:35113:35498 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 0.05 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35116:35500 [3] NCCL INFO ncclCommInitRankConfig comm 0x5609a4f5f350 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 201d0 commId 0x3da1fa247351b1f5 - Init COMPLETE
ip-172-31-3-153:35115:35497 [2] NCCL INFO Init timings - ncclCommInitRankConfig: rank 2 nranks 4 total 0.05 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35114:35499 [1] NCCL INFO ncclCommInitRankConfig comm 0x556572999110 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 101d0 commId 0x3da1fa247351b1f5 - Init COMPLETE
ip-172-31-3-153:35116:35500 [3] NCCL INFO Init timings - ncclCommInitRankConfig: rank 3 nranks 4 total 0.05 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35114:35499 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 4 total 0.05 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.02, graphs 0.00, connections 0.02, rest 0.01)
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 02/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 03/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 04/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 05/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 04/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 01/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 01/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 05/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 06/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 02/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 06/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 02/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 07/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 03/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 07/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 03/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 08/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 04/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 08/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 04/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 09/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 05/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 09/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 10/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 05/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 10/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 06/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 11/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 06/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 11/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 07/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 12/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 07/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 12/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 08/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 08/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 13/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 13/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 09/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 14/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 09/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 14/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 10/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 15/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 15/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 10/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 11/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 16/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 16/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 11/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 12/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 17/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 17/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 12/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 18/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 13/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 18/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 19/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 13/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 19/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 14/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 20/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 14/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 20/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 15/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 21/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 21/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 15/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 22/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 16/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 22/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 16/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Channel 23/0 : 0[0] -> 1[1] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 17/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35116:35512 [3] NCCL INFO Channel 23/0 : 3[3] -> 0[0] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 17/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 18/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 18/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 19/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 19/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 20/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 20/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 21/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 21/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 22/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 22/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35114:35509 [1] NCCL INFO Channel 23/0 : 1[1] -> 2[2] via P2P/CUMEM/read
ip-172-31-3-153:35115:35510 [2] NCCL INFO Channel 23/0 : 2[2] -> 3[3] via P2P/CUMEM/read
ip-172-31-3-153:35113:35511 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
ip-172-31-3-153:35114:35509 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
ip-172-31-3-153:35116:35512 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
ip-172-31-3-153:35115:35510 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
Parameter Offload - Persistent parameters statistics: param_count = 65, numel = 266240
DeepSpeedZeRoOffload initialize [end]
MA 0.0 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 34.18 GB, percent = 3.0%
Before creating fp16 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 34.17 GB, percent = 3.0%
/opt/pytorch/lib64/python3.12/site-packages/torch/distributed/distributed_c10d.py:4876: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  warnings.warn(  # warn only once
After creating fp16 partitions: 19
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 51.82 GB, percent = 4.6%
Before creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 52.98 GB, percent = 4.7%
After creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 81.54 GB, percent = 7.3%
Before initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 83.09 GB, percent = 7.4%
After initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 118.12 GB, percent = 10.5%
After initializing ZeRO optimizer
MA 0.09 GB         Max_MA 2.05 GB         CA 15.08 GB         Max_CA 15 GB
CPU Virtual Memory:  used = 134.46 GB, percent = 12.0%
***** Running training *****
  Num examples = 36,718
  Num Epochs = 1
  Instantaneous batch size per device = 1
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 4
  Total optimization steps = 50
  Number of trainable parameters = 8,030,261,248
  0%|                                                                           
```

