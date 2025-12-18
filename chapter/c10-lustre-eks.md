AWS 에서 Lustre 파일 시스템을 AI 시스템에 사용하는 가장 빠른 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 와 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

#### 1단계: FSx for Lustre 파일 시스템 생성 ###
EKS 클러스터와 동일한 VPC 내에 Amazon FSx for Lustre 파일 시스템을 생성합니다. S3 버킷과 연결하여 데이터를 자동으로 가져오거나 내보낼 수 있다.

<<< 러스터 설치하는 방범 추가 >>>


#### 2단계: IAM 역할 생성 및 연결 ####
FSx CSI 드라이버 컨트롤러가 AWS API를 호출할 수 있도록 적절한 권한을 가진 IAM 역할이 필요한데, AWS 관리형 정책인 
AmazonFSxLustreCSIDriverPolicy를 사용하면 된다

#### 3단계: Amazon FSx CSI 드라이버 설치 #### 
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

#### 4단계: StorageClass 및 Persistent Volume Claim (PVC) 배포 ###
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

#### 5단계: 애플리케이션 파드에서 사용 ####
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


## 사용시 주의점 또는 고려사항 ##


