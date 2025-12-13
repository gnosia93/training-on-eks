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
  count                   = 2
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  vpc_id                  = aws_vpc.main.id
  map_public_ip_on_launch = true
  tags = { Name = "TOE-pub-subnet-${count.index + 1}" }
}

resource "aws_subnet" "private" {
  count             = 2
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 2)
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
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
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
echo "install code-server ..."
sudo su - ec2-user -c "curl -fsSL https://code-server.dev/install.sh | sh"
sudo su - ec2-user -c "nohup code-server --bind-addr 0.0.0.0:8080 --auth none > /home/ec2-user/code-server.log 2>&1 &"

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
echo "install code-server ..."
sudo su - ec2-user -c "curl -fsSL https://code-server.dev/install.sh | sh"
sudo su - ec2-user -c "nohup code-server --bind-addr 0.0.0.0:8080 --auth none > /home/ec2-user/code-server.log 2>&1 &"

_DATA

  tags = {
    Name = "code-server-x86"
  }
}


