AWS 에서 Lustre 파일 시스템을 AI 시스템에 사용하는 가장 빠른 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 와 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

### 1단계: FSx for Lustre 파일 시스템 생성 ##
EKS 클러스터와 동일한 VPC 내에 Amazon FSx for Lustre 파일 시스템을 생성합니다. S3 버킷과 연결하여 데이터를 자동으로 가져오거나 내보낼 수 있다.

[lustre.tf]
```
# 1. FSx for Lustre 생성
resource "aws_fsx_lustre_file_system" "example" {
  storage_capacity            = 1200 # 용량 (단위: GiB, 최소 1200 또는 2400)
  subnet_ids                  = ["subnet-12345678"] # 설치할 서브넷 ID
  security_group_ids          = [aws_security_group.fsx_sg.id]
  deployment_type             = "SCRATCH_2" # SCRATCH_1, SCRATCH_2, PERSISTENT_1, PERSISTENT_2 중 선택
  import_path                 = "s3://my-data-bucket-name" # 연동할 S3 버킷 (선택 사항)
  export_path                 = "s3://my-data-bucket-name/export" # 결과 내보낼 S3 경로
  per_unit_storage_throughput = 200 # PERSISTENT 타입일 때 설정 (MB/s/TiB)

  tags = {
    Name = "MyLustreFileSystem"
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
}

# (선택) S3와 데이터 동기화를 위한 설정
resource "aws_fsx_data_repository_association" "example" {
  file_system_id       = aws_fsx_lustre_file_system.example.id
  data_repository_path = "s3://my-data-bucket-name"
  file_system_path     = "/"
  batch_import_meta_data_on_create = true
}
```
#### 주요 설정 항목 설명 ####
* deployment_type:
  * SCRATCH: 임시 데이터 처리용 (데이터 복제 없음, 비용 저렴).
  * PERSISTENT: 장기 보관용 (고가용성 및 데이터 복제 지원).
* storage_capacity: SSD 기준 최소 1,200 GiB부터 시작하며, 1,200 또는 2,400 단위로 증설해야 합니다.
* 네트워크(Port 988): Lustre 클라이언트(EC2 등)와 FSx 사이에는 TCP 988 포트가 반드시 열려 있어야 통신이 가능합니다. AWS 공식 문서에서 보안 그룹 요구 사항을 더 확인할 수 있습니다.
* S3 연동 (import_path / export_path): 기존 S3 버킷의 데이터를 Lustre로 불러오거나 결과를 다시 S3로 저장할 때 사용합니다.


### 2단계: IAM 역할 생성 및 연결 ###
FSx CSI 드라이버 컨트롤러가 AWS API를 호출할 수 있도록 적절한 권한을 가진 IAM 역할이 필요한데, AWS 관리형 정책인 
AmazonFSxLustreCSIDriverPolicy를 사용하면 된다

### 3단계: Amazon FSx CSI 드라이버 설치 ### 
Helm을 사용하여 EKS 클러스터에 FSx for Lustre CSI 드라이버를 배포한다. 
```
# EKS 클러스터에 CSI 드라이버 배포를 위한 네임스페이스 생성
kubectl create namespace fsx-csi-driver

# Helm 차트 추가 및 업데이트
helm repo add aws-fsx-csi-driver kubernetes-sigs.github.io
helm repo update

# CSI 드라이버 설치 (서비스 계정에 IAM 역할을 연결해야 함)
helm install fsx-csi-driver --namespace fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver --set image.repository=<이미지_레포지토리_URL>,controller.serviceAccount.name=fsx-csi-driver-controller-sa,controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=<IAM_역할_ARN>
```

### 4단계: StorageClass 및 Persistent Volume Claim (PVC) 배포 ###
동적 프로비저닝을 위해 StorageClass를 정의하고, 워크로드에서 사용할 PersistentVolumeClaim을 생성합니다.  
```
# storageclass.yaml 예시 (동적 프로비저닝)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-sc
provisioner: fsx.csi.aws.com
parameters:
  subnetId: "subnet-xxxxxxxxxxxxxxxxx" # EKS 클러스터 서브넷 ID
  securityGroupIds: "sg-xxxxxxxxxxxxxxxxx" # FSx 접근 허용 보안 그룹 ID
  # ... 기타 FSx 생성 옵션 ...
```

### 5단계: 애플리케이션 파드에서 사용 ###
```
# pod-with-fsx.yaml 예시
apiVersion: v1
kind: Pod
metadata:
  name: app-using-fsx
spec:
  containers:
  - name: app
    image: centos:latest
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: fsx-volume
      mountPath: /data/fsx
  volumes:
    - name: fsx-volume
      persistentVolumeClaim:
        claimName: fsx-pvc # 위에서 생성한 PVC 이름
```

## 버전별 특징 및 성능 ##


## 사용시 주의점 및 고려사항 ##

## 레퍼런스 ##
* https://aws.amazon.com/ko/blogs/tech/lustre/

