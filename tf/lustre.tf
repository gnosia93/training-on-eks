# 현재 계정 정보 가져오기
data "aws_caller_identity" "current" {}

# 현재 시간을 가져오기 위한 리소스
resource "time_static" "current" {}

# 현재 설정된 리전 정보를 가져오기 위한 데이터 소스
data "aws_region" "current" {}

# 시간 포맷 설정 (예: 20260107114705)
locals {
  timestamp = formatdate("YYMMDDhhmm", time_static.current.rfc3339)
}

# S3 버킷 생성 (계정 ID 포함)
resource "aws_s3_bucket" "data_bucket" {
  bucket = "training-on-eks-lustre-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.name}"
}

# FSx for Lustre 생성
resource "aws_fsx_lustre_file_system" "lustre_file_system" {
  storage_capacity            = 1200                                       # 용량 (단위: GiB, 최소 1200 또는 2400)
  subnet_ids                  = [aws_subnet.private[0].id]                 # Amazon FSx for Lustre 파일 시스템 자체는 단일 서브넷(Single Subnet)에서만 생성
  security_group_ids          = [aws_security_group.lustre_sg.id]
  deployment_type             = "SCRATCH_2"                                # SCRATCH_1, SCRATCH_2, PERSISTENT_1, PERSISTENT_2 중 선택
  import_path                 = "s3://${aws_s3_bucket.data_bucket.bucket}"
  export_path                 = "s3://${aws_s3_bucket.data_bucket.bucket}/export"
#  per_unit_storage_throughput = 200                                       # PERSISTENT 타입일 때 설정 (MB/s/TiB)
  # 자동 가져오기 설정 - S3에서 새로 생성(NEW)되거나 변경(CHANGED)된 파일을 자동으로 감지합니다.
  auto_import_policy          = "NEW_CHANGED" 
  file_system_type_version    = "2.15" 
  
  tags = {
    Name = "trainng-on-eks"
  }
}

# 3. 보안 그룹 설정 (VPC 전체 대역 허용)
resource "aws_security_group" "lustre_sg" {
  name        = "lustre-vpc-wide-sg"
  description = "Allow Lustre traffic from entire VPC"
  vpc_id      = aws_vpc.main.id

  # 988 포트: Lustre 파일 시스템 기본 포트
  ingress {
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block] # VPC 내부 모든 IP에서 접근 허용
  }

  # 1021-1023 포트: Lustre RPC/데이터 전송 포트
  ingress {
    from_port   = 1018
    to_port     = 1023
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block] # VPC 내부 모든 IP에서 접근 허용
  }

  # 모든 아웃바운드 허용
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "lustre-sg" }
}



