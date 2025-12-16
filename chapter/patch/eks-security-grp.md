[eks-security.tf]
```
# VPC ID 변수 (기존 VPC 사용 시)
variable "vpc_id" {
  default = "vpc-0123456789abcdef0" 
}

# 1. Cluster Control Plane 보안 그룹
resource "aws_security_group" "eks_control_plane" {
  name        = "dh-eks-control-plane-sg"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "dh-eks-control-plane-sg"
  }
}

# 2. Shared Node 보안 그룹 (Karpenter Discovery용)
resource "aws_security_group" "eks_shared_node" {
  name        = "dh-eks-shared-node-sg"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "dh-eks-shared-node-sg"
    # Karpenter가 이 태그를 보고 노드에 SG를 입힙니다.
    "karpenter.sh/discovery" = "dh-eks"
    "kubernetes.io/cluster/dh-eks" = "owned"
  }
}

# 3. 보안 규칙: 노드 간 상호 통신 (Self-Reference)
resource "aws_security_group_rule" "nodes_internal" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 65535
  protocol                 = "-1"
  security_group_id        = aws_security_group.eks_shared_node.id
  source_security_group_id = aws_security_group.eks_shared_node.id
}

# 4. 보안 규칙: Control Plane <-> Nodes 통신 (443, 10250)
resource "aws_security_group_rule" "cp_to_nodes" {
  type                     = "ingress"
  from_port                = 10250
  to_port                  = 10250
  protocol                 = "tcp"
  security_group_id        = aws_security_group.eks_shared_node.id
  source_security_group_id = aws_security_group.eks_control_plane.id
}

resource "aws_security_group_rule" "nodes_to_cp" {
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.eks_control_plane.id
  source_security_group_id = aws_security_group.eks_shared_node.id
}
```

[eksctl]
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: dh-eks
  region: ap-northeast-2
  version: "1.31" # 2025년 최신 안정 버전

vpc:
  id: "${VPC_ID}"
  securityGroup: "${CONTROL_PLANE_SG_ID}"      # 위 테라폼 1번 결과물
  sharedNodeSecurityGroup: "${SHARED_NODE_SG_ID}" # 위 테라폼 2번 결과물
  subnets:
    private:
```
