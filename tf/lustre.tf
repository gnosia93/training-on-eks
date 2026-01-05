# 현재 계정 정보 가져오기
data "aws_caller_identity" "current" {}

# S3 버킷 생성 (계정 ID 포함)
resource "aws_s3_bucket" "data_bucket" {
  bucket = "training-on-eks-lustre-${data.aws_caller_identity.current.account_id}"
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
    from_port   = 1021
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



# 마운트에 필요한 정보 출력
output "fsx_id" {
  value       = aws_fsx_lustre_file_system.lustre_file_system.id
  description = "FSx 파일 시스템 ID입니다. (관리 및 문제 해결용)"
}

output "fsx_dns_name" {
  value       = aws_fsx_lustre_file_system.lustre_file_system.dns_name
  description = "마운트 명령어의 주소 부분에 사용됩니다."
}

output "mount_name" {
  value       = aws_fsx_lustre_file_system.lustre_file_system.mount_name
  description = "마운트 명령어의 경로 부분(마지막)에 사용됩니다."
}




















/*
locals {
  # 생성되는 리소스의 속성에서 직접 추출
  oidc_url = replace(aws_eks_cluster.training-on-eks.identity[0].oidc[0].issuer, "https://", "")
}

# IAM 역할 생성 (EKS OIDC와 연동)
resource "aws_iam_role" "fsx_csi_role" {
  name = "AmazonEKS_FSx_Lustre_CSI_Driver_Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.oidc_url}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${local.oidc_url}:sub": "system:serviceaccount:fsx-csi-driver:fsx-csi-driver-controller-sa"
          }
        }
      }
    ]
  })
}

# 관리형 정책(AmazonFSxLustreCSIDriverPolicy) 연결
resource "aws_iam_role_policy_attachment" "fsx_csi_policy_attach" {
  role       = aws_iam_role.fsx_csi_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonFSxLustreCSIDriverPolicy"
}

# S3 접근을 위한 커스텀 정책 생성
resource "aws_iam_policy" "fsx_s3_integration_policy" {
  name        = "FSxLustreS3IntegrationPolicy"
  description = "Allows FSx for Lustre to sync with specific S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:GetBucketAcl",
          "s3:GetObject",
          "s3:GetObjectTagging",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        # 위에서 만든 버킷 경로와 일치시켜야 함
        Resource = [
          "arn:aws:s3:::training-on-eks-lustre-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::training-on-eks-lustre-${data.aws_caller_identity.current.account_id}/*"
        ]
      }
    ]
  })
}

# 기존 역할(Role)에 이 정책을 연결
resource "aws_iam_role_policy_attachment" "fsx_s3_attach" {
  role       = aws_iam_role.fsx_csi_role.name # 이전에 만든 Role 이름
  policy_arn = aws_iam_policy.fsx_s3_integration_policy.arn
}
*/
