provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {
  state = "available"
}

# ------------------------------------------------
# VPC 및 네트워크 구성
# ------------------------------------------------
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "training-on-eks"
  }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_subnet" "public" {
  count                   = length(data.aws_availability_zones.available.names)
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  vpc_id                  = aws_vpc.main.id
  map_public_ip_on_launch = true
  tags = { Name = "TOE-pub-subnet-${count.index + 1}" }
}

resource "aws_subnet" "private" {
  count             = length(data.aws_availability_zones.available.names)
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 4)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  vpc_id            = aws_vpc.main.id
  tags = { Name = "TOE-priv-subnet-${count.index + 1}" }
}

resource "aws_eip" "nat" {
  domain           = "vpc"
}

resource "aws_nat_gateway" "gw" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.gw.id
  }
}

resource "aws_route_table_association" "public" {
  count          = length(data.aws_availability_zones.available.names)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(data.aws_availability_zones.available.names)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}


# ------------------------------------------------
# EC2 인스턴스용 IAM Role 및 Profile 추가 <--- 이 부분이 추가되었습니다.
# ------------------------------------------------

resource "aws_iam_role" "eks_creator_role" {
  name = "TOE_EKS_EC2_Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# EKS 클러스터 생성을 위한 필수 권한 부여
# Note: 이 정책들은 클러스터 생성에 필요한 거의 모든 권한을 포함하므로 주의해야 합니다.
resource "aws_iam_role_policy_attachment" "eks_creator_policy_cluster" {
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
  role       = aws_iam_role.eks_creator_role.name
}

# EC2 인스턴스에 IAM Role을 연결하기 위한 Instance Profile
resource "aws_iam_instance_profile" "eks_creator_profile" {
  name = "EKS_Creator_Profile"
  role = aws_iam_role.eks_creator_role.name
}



# ------------------------------------------------
# Graviton / X86 EC2 인스턴스 구성
# ------------------------------------------------

data "aws_ami" "al2023_arm64" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-kernel-6.1-arm64"]
  }
}

data "aws_ami" "al2023_x86_64" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-kernel-6.1-x86_64"]
  }
}


resource "aws_security_group" "instance_sg" {
  vpc_id = aws_vpc.main.id
  name   = "eks-host-sg"

  # SSH 접속 허용
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ip_cidrs
  }

  # VS Code Server (Code Server) 접속 허용
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_ip_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "graviton_box" {
  ami                         = data.aws_ami.al2023_arm64.id
  instance_type               = var.graviton_type
  subnet_id                   = aws_subnet.public[0].id
  vpc_security_group_ids      = [aws_security_group.instance_sg.id]
  associate_public_ip_address = true
  key_name                    = var.key_name

  # IAM Instance Profile 연결 <--- EC2에 권한을 부여합니다.
  iam_instance_profile = aws_iam_instance_profile.eks_creator_profile.name

  user_data = <<_DATA
#!/bin/bash
EC2_HOME="/home/ec2-user"
CONFIG_FILE="$EC2_HOME/.config/code-server/config.yaml"

echo "Starting code-server installation as root..."

# 1. code-server 설치 (root 권한으로 RPM 패키지 설치)
curl -fsSL code-server.dev | sh

# 2. code-server 서비스 활성화 및 시작 (systemctl은 root 권한으로 실행되어야 함)
# code-server@ec2-user 서비스 인스턴스를 대상으로 합니다.
systemctl enable --now code-server@ec2-user
systemctl start --now code-server@ec2-user

echo "Waiting for config file $CONFIG_FILE to be created..."
MAX_RETRIES=10
RETRY_COUNT=0

# 3. 설정 파일 생성 대기
while [ ! -f "$CONFIG_FILE" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 2
    ((RETRY_COUNT++))
done

# 4. 파일 존재 확인 및 수정
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating bind-addr and auth in $CONFIG_FILE"
    # root 권한으로 파일 수정
    sed -i 's/127.0.0.1/0.0.0.0/g' "$CONFIG_FILE"
    sed -i 's/auth: password/auth: none/g' "$CONFIG_FILE"
    
    # 설치 스크립트가 이미 소유권을 설정했겠지만, 확실하게 ec2-user 소유로 변경
    chown ec2-user:ec2-user "$CONFIG_FILE"
else
    echo "Error: Config file not found after retries. Manual intervention needed."
fi

# 5. 변경된 설정을 적용하기 위해 서비스 재시작
systemctl restart code-server@ec2-user

echo "user data script ended successfully."

_DATA

  tags = {
    Name = "code-server-graviton"
  }
}

resource "aws_instance" "x86_box" {
  ami                         = data.aws_ami.al2023_x86_64.id
  instance_type               = var.x86_type
  subnet_id                   = aws_subnet.public[0].id
  vpc_security_group_ids      = [aws_security_group.instance_sg.id]
  associate_public_ip_address = true
  key_name                    = var.key_name

  # IAM Instance Profile 연결 <--- EC2에 권한을 부여합니다.
  iam_instance_profile = aws_iam_instance_profile.eks_creator_profile.name

  user_data = <<_DATA
#!/bin/bash
EC2_HOME="/home/ec2-user"
CONFIG_FILE="$EC2_HOME/.config/code-server/config.yaml"

echo "Starting code-server installation as root..."

# 1. code-server 설치 (root 권한으로 RPM 패키지 설치)
curl -fsSL code-server.dev | sh

# 2. code-server 서비스 활성화 및 시작 (systemctl은 root 권한으로 실행되어야 함)
# code-server@ec2-user 서비스 인스턴스를 대상으로 합니다.
systemctl enable --now code-server@ec2-user
systemctl start --now code-server@ec2-user

echo "Waiting for config file $CONFIG_FILE to be created..."
MAX_RETRIES=10
RETRY_COUNT=0

# 3. 설정 파일 생성 대기
while [ ! -f "$CONFIG_FILE" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 2
    ((RETRY_COUNT++))
done

# 4. 파일 존재 확인 및 수정
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating bind-addr and auth in $CONFIG_FILE"
    # root 권한으로 파일 수정
    sed -i 's/127.0.0.1/0.0.0.0/g' "$CONFIG_FILE"
    sed -i 's/auth: password/auth: none/g' "$CONFIG_FILE"
    
    # 설치 스크립트가 이미 소유권을 설정했겠지만, 확실하게 ec2-user 소유로 변경
    chown ec2-user:ec2-user "$CONFIG_FILE"
else
    echo "Error: Config file not found after retries. Manual intervention needed."
fi

# 5. 변경된 설정을 적용하기 위해 서비스 재시작
systemctl restart code-server@ec2-user

echo "user data script ended successfully."
_DATA

  tags = {
    Name = "code-server-x86"
  }
}


