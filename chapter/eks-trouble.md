## EKS 클러스터 kubectl 접근 트러블슈팅 과정 ##

### 1단계: 네트워크 타임아웃 ###
```
dial tcp 10.0.0.34:443: i/o timeout
```
EC2에서 kubectl get nodes 실행 시 EKS API 서버에 연결 자체가 안 됨.

* 원인: EC2가 같은 VPC에 있어서 EKS endpoint가 프라이빗 IP(10.0.0.34)로 resolve되는데, EKS 클러스터 보안그룹에서 EC2 보안그룹의 443 인바운드가 허용되지 않았음.
* 해결:
```
aws ec2 authorize-security-group-ingress \
  --group-id <EKS_클러스터_보안그룹_ID> \
  --protocol tcp \
  --port 443 \
  --source-group <EC2_보안그룹_ID> \
  --region ap-northeast-2
```

### 2단계: 인증 에러 ###
```
the server has asked for the client to provide credentials
```
네트워크는 뚫렸지만, EC2의 IAM Role이 EKS 클러스터에 접근 권한이 없음.

* 해결 시도: EKS Access Entry API로 EC2 IAM Role 추가하려 했으나 →

### 3단계: 인증 모드 에러 ###
```
The cluster's authentication mode must be set to one of [API, API_AND_CONFIG_MAP]
```
클러스터 인증 모드가 CONFIG_MAP으로만 설정되어 있어서 Access Entry API 사용 불가. aws-auth ConfigMap으로 추가하려 해도 kubectl 자체가 안 되는 순환 문제.

* 해결:
```
# 1. 인증 모드 변경
aws eks update-cluster-config \
  --name eks-agentic-ai \
  --access-config authenticationMode=API_AND_CONFIG_MAP \
  --region ap-northeast-2

# 2. 클러스터 ACTIVE 확인 후 access entry 추가
aws eks create-access-entry \
  --cluster-name eks-agentic-ai \
  --principal-arn arn:aws:iam::499514681453:role/EAI_EC2_Role-ap-northeast-2 \
  --region ap-northeast-2

# 3. 클러스터 관리자 권한 부여
aws eks associate-access-policy \
  --cluster-name eks-agentic-ai \
  --principal-arn arn:aws:iam::499514681453:role/EAI_EC2_Role-ap-northeast-2 \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
  --access-scope type=cluster \
  --region ap-northeast-2
```
### 핵심 포인트 ###
* EKS public endpoint가 열려있어도, 같은 VPC의 EC2는 프라이빗 IP로 resolve되므로 클러스터 보안그룹 인바운드 규칙이 필요
* sts:assumed-role ARN이 아닌 iam:role ARN을 사용해야 함
* CONFIG_MAP 모드에서는 Access Entry API 사용 불가 → API_AND_CONFIG_MAP으로 변경 필요
* Terraform에서 근본적으로 해결하려면 EKS 모듈에 authentication_mode = "API_AND_CONFIG_MAP" 설정과 EC2 Role에 대한 access entry를 추가
