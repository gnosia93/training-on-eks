variable "aws_region" {
  description = "AWS Region to deploy resources"
  type        = string
  default     = "ap-northeast-2" # 원하는 리전으로 변경하세요 (예: "us-east-1")
}

variable "vpc_cidr_block" {
  description = "CIDR block for the main VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "instance_type" {
  description = "EC2 Instance Type (Graviton)"
  type        = string
  default     = "c7g.xlarge"
}

variable "key_name" {
  description = "AWS SSH Key Pair name for EC2 access"
  type        = string
  # TODO: 이 기본값을 사용자의 실제 AWS 키페어 이름으로 변경하세요.
  default     = "your-key-pair-name" 
}

variable "vscode_server_password" {
  description = "Password for the Code Server (VS Code Server) web UI"
  type        = string
  default     = "password" # 보안을 위해 강력한 비밀번호로 변경 권장
}

variable "allowed_ip_cidrs" {
  description = "List of CIDR blocks allowed to access SSH (22) and VS Code (8080)"
  type        = list(string)
  # 0.0.0.0/0 은 모든 IP를 허용합니다. 보안을 위해 본인의 IP CIDR로 변경하세요.
  default     = ["0.0.0.0/0"] 
  default     = [
#    "0.0.0,0/0",         # 모든 IP
    "122.36.213.114/32"  # 내아이피
  ]
}

variable "codeserver_version" {
  description = "Specific version of the code-server to install"
  type        = string
  default     = "4.21.1" # 최신 버전을 확인하여 업데이트 가능
}
