variable "aws_region" {
  description = "AWS Region to deploy resources"
  type        = string
  default     = "ap-northeast-1" # 원하는 리전으로 변경하세요 (예: "us-east-1")
}

variable "cluster_name" {
  type        = string
  default     = "training-on-eks"
}

variable "vpc_cidr_block" {
  description = "CIDR block for the main VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "graviton_type" {
  description = "EC2 Instance Type (Graviton)"
  type        = string
  default     = "c7g.2xlarge"
}

variable "x86_type" {
  description = "EC2 Instance Type (x86)"
  type        = string
  default     = "c6i.2xlarge"
}

variable "key_name" {
  description = "AWS SSH Key Pair name for EC2 access"
  type        = string
  # TODO: 이 기본값을 사용자의 실제 AWS 키페어 이름으로 변경하세요.
  default     = "aws-kp-1" 
}


# 공인 IP 확인
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

variable "custom_allowed_ips" {
  description = "허용할 IP 리스트"
  type        = list(string)
  default     = []            # 필요한 경우 "10.0.0.0/8" 등을 추가
}

locals {
  # chomp로 개행 제거 후 /32 추가
  current_ip = "${chomp(data.http.my_ip.response_body)}/32"

  # 최종 리스트: 현재 내 IP + 변수로 받은 IP 리스트 합치기
  allowed_ip_cidrs = concat([local.current_ip], var.custom_allowed_ips)
}

