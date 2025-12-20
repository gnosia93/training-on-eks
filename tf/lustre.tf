# 1. 현재 계정 정보 가져오기
data "aws_caller_identity" "current" {}

# 2. S3 버킷 생성 (계정 ID 포함)
resource "aws_s3_bucket" "data_bucket" {
  bucket = "training-data-${data.aws_caller_identity.current.account_id}"
}

# 1. FSx for Lustre 생성
resource "aws_fsx_lustre_file_system" "lustre_file_system" {
  storage_capacity            = 1200                                       # 용량 (단위: GiB, 최소 1200 또는 2400)
  subnet_ids                  = [aws_subnet.private[0].id]                 # 설치할 서브넷 ID
  security_group_ids          = [aws_security_group.fsx_sg.id]
  deployment_type             = "SCRATCH_2"                                # SCRATCH_1, SCRATCH_2, PERSISTENT_1, PERSISTENT_2 중 선택
  import_path                 = "s3://${aws_s3_bucket.data_bucket.bucket}"
  export_path                 = "s3://${aws_s3_bucket.data_bucket.bucket}/export"
#  per_unit_storage_throughput = 200                                       # PERSISTENT 타입일 때 설정 (MB/s/TiB)

  tags = {
    Name = "trainng-on-eks"
  }
}

# 2. 보안 그룹 설정 (Lustre 전용 포트 988 오픈)
resource "aws_security_group" "fsx_sg" {
  name        = "fsx-lustre-sg"
  description = "Allow Lustre traffic"
  vpc_id      = "vpc-12345678" # VPC ID 입력

  ingress {
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    self        = true # 동일 보안 그룹 내 통신 허용
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "trainng-on-eks"
  }
}

# 현재 AWS 계정 정보를 가져오는 데이터 소스
data "aws_caller_identity" "current" {}

# (선택) S3와 데이터 동기화를 위한 설정
resource "aws_fsx_data_repository_association" "lustre_file_system_s3" {
  file_system_id       = aws_fsx_lustre_file_system.lustre_file_system.id
  data_repository_path = "s3://training-on-eks-lustre-${data.aws_caller_identity.current.account_id}"
  file_system_path     = "/"                  # S3 버킷을 Lustre 파일 시스템에 마운트 했을때의 최상위 경로
  batch_import_meta_data_on_create = true     # 파일 시스템이 생성되는 즉시 S3에 있는 파일들의 메타데이터를 Lustre 인덱스에 등록
}


locals {
  # 생성되는 리소스의 속성에서 직접 추출
  oidc_url = replace(aws_eks_cluster.training-on-eks.identity[0].oidc[0].issuer, "https://", "")
}

# 1. IAM 역할 생성 (EKS OIDC와 연동)
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

# 2. 관리형 정책(AmazonFSxLustreCSIDriverPolicy) 연결
resource "aws_iam_role_policy_attachment" "fsx_csi_policy_attach" {
  role       = aws_iam_role.fsx_csi_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonFSxLustreCSIDriverPolicy"
}

# 1. S3 접근을 위한 커스텀 정책 생성
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
          "arn:aws:s3:::training-on-eks-lustre-s3-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::training-on-eks-lustre-s3-${data.aws_caller_identity.current.account_id}/*"
        ]
      }
    ]
  })
}

# 2. 기존 역할(Role)에 이 정책을 연결
resource "aws_iam_role_policy_attachment" "fsx_s3_attach" {
  role       = aws_iam_role.fsx_csi_role.name # 이전에 만든 Role 이름
  policy_arn = aws_iam_policy.fsx_s3_integration_policy.arn
}
