## 인스턴스 준비 ##

```
#!/bin/bash

read -p "Region ID를 입력하세요 (예: us-east-1): " REGION

VPC_ID=$(aws ec2 describe-vpcs \
    --region $REGION \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text)
echo "조회된 기본 VPC: $VPC_ID"

SUBNET_ID=$(aws ec2 describe-subnets \
    --region $REGION \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' \
    --output text)
echo "조회된 기본 서브넷: $SUBNET_ID"

# 4. (참고) 본인의 퍼블릭 IP 자동 조회
# 수동 입력 대신 현재 실행 중인 PC의 공인 IP를 자동으로 가져옵니다.
MY_IP=$(curl -s https://checkip.amazonaws.com)/32
echo "현재 접속 IP 자동 감지: $MY_IP"

echo "------------------------------------------------"


# 1. 환경 정보 입력 받기
read -p "Region ID를 입력하세요 (예: us-east-1): " REGION
read -p "VPC ID를 입력하세요: " VPC_ID
read -p "Subnet ID를 입력하세요: " SUBNET_ID
read -p "허용할 IP 주소 (예: 1.2.3.4/32): " MY_IP
read -p "사용할 Key Pair 이름: " KEY_NAME

echo "------------------------------------------------"
echo "$REGION 리전에서 최신 PyTorch Deep Learning AMI 조회 중..."

# 2. 최신 Deep Learning AMI ID 조회 (PyTorch 지원 OSS Nvidia Driver 버전)
# 2026년 기준 가장 안정적인 'Deep Learning OSS Nvidia Driver v202' 계열을 타겟팅합니다.
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners amazon \
    --filters "Name=name,Values=deep-learning-oss-nvidia-driver-*-ubuntu-22.04-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
    echo "AMI를 찾을 수 없습니다. 리전 ID를 확인해주세요."
    exit 1
fi

echo "조회된 AMI ID: $AMI_ID"

# 3. 보안 그룹 생성
SG_ID=$(aws ec2 create-security-group \
    --region $REGION \
    --group-name "P4-PyTorch-SG-$(date +%s)" \
    --description "SG for P4 with VSCode" \
    --vpc-id $VPC_ID --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $SG_ID \
    --protocol tcp --port 22 --cidr $MY_IP

# 4. User Data 작성 (VSCode CLI 설치)
cat <<EOF > userdata.sh
#!/bin/bash
apt-get update -y
curl -Lk 'code.visualstudio.com' --output /home/ubuntu/vscode_cli.tar.gz
tar -xf /home/ubuntu/vscode_cli.tar.gz -C /home/ubuntu/
chown ubuntu:ubuntu /home/ubuntu/code
EOF

# 5. 인스턴스 실행 (P4.24xlarge)
echo "인스턴스를 생성 중입니다..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type p4.24xlarge \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --user-data file://userdata.sh \
    --query 'Instances[0].InstanceId' --output text)

echo "------------------------------------------------"
echo "인스턴스 ID: $INSTANCE_ID"
echo "성공적으로 요청되었습니다. AWS 콘솔에서 상태를 확인하세요."

```

```
# 1. 설치된 Conda 환경 목록 확인
conda env list

# 2. PyTorch 전용 환경 활성화 (가장 최신 버전 선택)
conda activate pytorch

# 3. 파이썬에서 PyTorch 설치 및 GPU 연결 확인
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}')"
```
