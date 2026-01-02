## EC2 NVLK/NCCL 테스트 ##

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

### 훈련시작 ###
```
sudo mkdir -p /data
sudo chown ec2-user:ec2-user /data
sudo dnf update -y
sudo dnf install python3-pip -y

sudo dnf install -y cuda-toolkit
# CUDA 경로 추가
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

find /usr/local/cuda -name "libcurand.so*"
DS_BUILD_CPU_ADAM=1 pip install deepspeed --force-reinstall --no-cache-dir --no-build-isolation


git clone https://github.com/gnosia93/training-on-eks.git
cd ~/training-on-eks/samples/deepspeed
sh train-ec2.sh
```

