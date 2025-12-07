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
    Name = "eks-cluster-vpc"
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
  tags = { Name = "public-subnet-${count.index + 1}" }
}

resource "aws_subnet" "private" {
  count             = 2
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 2)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  vpc_id            = aws_vpc.main.id
  tags = { Name = "private-subnet-${count.index + 1}" }
}

resource "aws_eip" "nat" {
  vpc = true
}

resource "aws_nat_gateway" "gw" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id
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
# Graviton EC2 인스턴스 구성
# ------------------------------------------------

data "aws_ami" "al2023_arm64" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-kernel-6.1-arm64"]
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
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.instance_sg.id]
  associate_public_ip_address = true
  key_name                    = var.key_name

  # IAM Instance Profile 연결 <--- EC2에 권한을 부여합니다.
  iam_instance_profile = aws_iam_instance_profile.eks_creator_profile.name

  user_data = <<-EOF
#!/bin/bash
sudo dnf update -y
sudo dnf install -y git unzip curl

# Install AWS CLI v2
curl "awscli.amazonaws.com" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws

# Install kubectl
curl -O s3.us-west-2.amazonaws.com
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl

# Install eksctl
curl --silent --location "github.com" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install VS Code Server (Code Server) using the version variable
curl -fsSL code-server.dev | sh -s -- --version ${var.codeserver_version}

# Run Code Server in background
nohup code-server --bind-addr 0.0.0.0:8080 --auth ${var.vscode_server_password} &> /home/ec2-user/vscode.log &
EOF

  tags = {
    Name = "Graviton3-EKS-Host"
  }
}
