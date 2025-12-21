## 러스터(Lustre) ##
러스터(Lustre) 파일 시스템은 높은 처리량, 낮은 지연 시간, 뛰어난 확장성을 제공하는 병렬 분산 파일 시스템으로, 대규모 데이터셋을 처리해야 하는 AI 시스템에 필수적이다.
AWS 에서 Lustre 파일 시스템을 사용하는 가장 빠른 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 와 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

### 1. 구성하기 ###
#### 1-1. Amazon FSx for Lustre 설치 #### 
```
export CLUSTER_NAME="training-on-eks"
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)

# 첫 번째 프라이빗 서브넷 ID 가져오기
export SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:Name,Values=*priv-subnet-1*" --query "Subnets[0].SubnetId" --output text)
export BUCKET_NAME="training-on-eks-lustre-${ACCOUNT_ID}"

echo "Using VPC: ${VPC_ID}, Subnet: ${SUBNET_ID}, Bucket: ${BUCKET_NAME}"

# S3 버킷 생성
aws s3 mb s3://${BUCKET_NAME} --region ${AWS_REGION}

# 시큐리티 그룹 생성 및 포트 오픈
FSX_SG_ID=$(aws ec2 create-security-group --group-name fsx-lustre-sg \
    --description "Allow Lustre traffic" --vpc-id ${VPC_ID} --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id ${FSX_SG_ID} --protocol tcp --port 988 --self
# EKS 노드 그룹 보안 그룹에서 988 허용 (필요 시 추가)

# FSx for Lustre 생성 (SCRATCH_2, 1200 GiB)
FSX_ID=$(aws fsx create-file-system \
    --file-system-type LUSTRE \
    --storage-capacity 1200 \
    --subnet-ids ${SUBNET_ID} \
    --security-group-ids ${FSX_SG_ID} \
    --lustre-configuration "DeploymentType=SCRATCH_2,ImportPath=s3://${BUCKET_NAME},ExportPath=s3://${BUCKET_NAME}/export,AutoImportPolicy=NEW_CHANGED" \
    --query "FileSystem.FileSystemId" --output text)

echo "FSx File System Creating: $FSX_ID"

# IAM 역할(IRSA) 생성 (eksctl 활용)
# 이 명령은 IAM Role 생성, 정책 연결, 서비스 어카운트 주석 처리를 한 번에 수행합니다.
eksctl create iamserviceaccount \
    --name fsx-csi-driver-controller-sa \
    --namespace fsx-csi-driver \
    --cluster ${CLUSTER_NAME} \
    --role-name "FSx_Lustre_CSI_Driver_Role" \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonFSxLustreCSIDriverPolicy \
    --approve \
    --override-existing-serviceaccounts

# 6. 커스텀 S3 접근 정책 생성 및 연결
cat <<EOF > s3-policy.json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["s3:GetBucketLocation","s3:ListBucket","s3:GetBucketAcl","s3:GetObject","s3:GetObjectTagging","s3:PutObject","s3:DeleteObject"],
        "Resource": ["arn:aws:s3:::${BUCKET_NAME}","arn:aws:s3:::${BUCKET_NAME}/*"]
    }]
}
EOF

S3_POLICY_ARN=$(aws iam create-policy --policy-name FSxLustreS3IntegrationPolicy --policy-document file://s3-policy.json --query Policy.Arn --output text)
aws iam attach-role-policy --role-name "FSx_Lustre_CSI_Driver_Role" --policy-arn $S3_POLICY_ARN

echo "Setup Complete!"
echo "FSX ID: ${FSX_ID}"
```
AVAILABLE 상태가 될 때까지 기다린다.
```
aws fsx describe-file-systems --file-system-ids ${FSX_ID} --query "FileSystems[0].Status"
```

#### 1-2. Amazon FSx CSI 드라이버 설치 #### 
Helm을 사용하여 EKS 클러스터에 FSx for Lustre CSI 드라이버를 배포한다. 
AWS FSx CSI 드라이버의 이미지는 각 리전별 AWS 전용 ECR 레포지토리에서 가져와야 한다.(image.repository)
```
kubectl create namespace fsx-csi-driver
helm repo add aws-fsx-csi-driver kubernetes-sigs.github.io
helm repo update

helm install fsx-csi-driver --namespace fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver \
--set image.repository=602401143452.dkr.ecr.${AWS_REGION}.amazonaws.com/eks/aws-fsx-csi-driver, \
controller.serviceAccount.name=fsx-csi-driver-controller-sa, \
controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"==arn:aws:iam::${ACCOUNT_ID}:role/AmazonEKS_FSx_Lustre_CSI_Driver_Role
```

#### 1-3. StorageClass 및 Persistent Volume Claim (PVC) 배포 ####
동적 프로비저닝을 위해 StorageClass를 정의한다.  
```
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-sc
provisioner: fsx.csi.aws.com
reclaimPolicy: Retain # PVC 삭제 시 FSx는 유지 (안전)
volumeBindingMode: Immediate
```

PersistentVolume (PV) 생성한다.
```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: fsx-pv
spec:
  capacity:
    storage: 1200Gi # FSx 생성 용량과 일치시킴
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fsx-sc
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: fs-xxxxxxxxxxxxxxxxx # 테라폼 결과물인 FSx ID
    volumeAttributes:
      dnsname: fs-xxxxxxxxxxxxxxxxx.fsx.ap-northeast-2.amazonaws.com # DNS 주소
      mountname: "xxxxxxxx" # 테라폼 output에서 확인 가능한 Mount Name
```

PersistentVolumeClaim을 생성한다. Pod는 이 PVC를 통해 볼륨을 요청합니다.
```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fsx-pvc
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: fsx-sc
  resources:
    requests:
      storage: 1200Gi
  volumeName: fsx-pv # 위에서 정의한 PV와 수동 연결
```

### 2. 테스트 하기 ###
#### 2-1. 애플리케이션 파드에서 사용 ####
```
cat <<EOF > pod-fsx.yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-fsx
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
EOF

kubectl apply -f pod-fsx.yaml
```

#### 2-2. S3 연동 테스트 ####
```
echo "Hello FSx Lustre" > test-file.txt
aws s3 cp test-file.txt s3://사용자-버킷-이름/

# Pod 내부 접속
kubectl exec -it pod-fsx -- /bin/bash

cd /
ls -l
```


## 레퍼런스 ##
* https://aws.amazon.com/ko/blogs/tech/lustre/

