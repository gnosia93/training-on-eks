variable "aws_region" {
  description = "AWS Region to deploy resources"
  type        = string
  default     = "ap-northeast-2" # 원하는 리전으로 변경하세요 (예: "us-east-1")
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
  default     = "aws-kp-2" 
}


# 공인 IP 확인
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

variable "allowed_ip_cidrs" {
  description = "List of CIDR blocks allowed to access SSH (22) and VS Code (8080)"
  type        = list(string)
  # 0.0.0.0/0 은 모든 IP를 허용합니다. 보안을 위해 본인의 IP CIDR로 변경하세요.
  default     = [
  #    "0.0.0,0/0",                                     # 모든 IP
    "${chomp(data.http.my_ip.response_body)}/32"        # CR, LF 제거.
  ]
}

