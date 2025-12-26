랑데뷰가 복구되어 파드들이 다시 모여도, 파이썬 스크립트(mnist.py)는 처음부터 다시 실행됩니다. 학습이 중단된 지점부터 이어서 하려면 반드시 다음 로직이 코드에 포함되어 있어야 합니다.
* 동작: 학습 코드 내부에서 최신 체크포인트(예: model.pt)가 있는지 확인하고, 있으면 로드하여 이어서 학습해야 합니다.
* 저장소: 파드가 죽어도 데이터가 유지되도록 PVC(Persistent Volume Claim) 또는 S3(Boto3)와 같은 외부 스토리지에 체크포인트를 저장해야 합니다.

## 체크포인트 ##
모든 노드가 최신 체크포인트 파일에 접근할 수 있어야 합니다. 이를 위해 현업에서는 크게 두 가지 방법을 사용합니다.
#### 1. 공유 스토리지 사용 (가장 권장됨) ####
모든 노드가 NFS, AWS FSx, Google Cloud Filestore와 같은 공유 네트워크 스토리지를 동일한 경로에 마운트하는 방식입니다.
* 장점: 특정 노드가 완전히 사라져도 데이터가 안전하며, 모든 노드가 같은 경로(/mnt/nfs/checkpoint.pt)를 바라보기만 하면 됩니다.
* 방식: 0번 마스터 노드가 체크포인트를 저장하면 나머지 노드들이 재시작 시 해당 파일을 읽어옵니다.

#### 2. 로컬 스토리지 + 복제 ####
각 노드의 로컬 디스크(SSD)에 저장하는 방식입니다.
* 단점: 노드 자체가 물리적으로 고장 나면 해당 노드에 있던 체크포인트는 유실됩니다.
* 방식: 이를 해결하려면 학습 중 주기적으로 체크포인트를 S3 같은 클라우드 스토리지로 업로드하거나, 모든 노드가 각자 자기 디스크에 동일한 복사본을 저장하도록 설계해야 합니다.

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
NODE_SG_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} \
    --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

FSX_SG_ID=$(aws ec2 create-security-group --group-name fsx-lustre-sg \
    --description "Allow Lustre traffic" --vpc-id ${VPC_ID} --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id ${FSX_SG_ID} \
    --protocol tcp --port 988  --source-group ${NODE_SG_ID}
aws ec2 authorize-security-group-ingress --group-id ${FSX_SG_ID} \
    --protocol tcp --port 1018-1023 --source-group ${NODE_SG_ID}


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
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
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
aws fsx describe-file-systems --file-system-ids ${FSX_ID} --query "FileSystems[0].Lifecycle"

aws fsx describe-file-systems \
    --file-system-ids ${FSX_ID} \
    --query "FileSystems[0].{FileSystemId:FileSystemId, DNSName:DNSName, MountName:LustreConfiguration.MountName}" \
    --output table
```
[결과]
```
------------------------------------------------------------------------------------------------
|                                      DescribeFileSystems                                     |
+--------------------------------------------------------+------------------------+------------+
|                         DNSName                        |     FileSystemId       | MountName  |
+--------------------------------------------------------+------------------------+------------+
|  fs-04cb64a224a31f75d.fsx.ap-northeast-2.amazonaws.com |  fs-04cb64a224a31f75d  |  ddyezbev  |
+--------------------------------------------------------+------------------------+------------+
```

#### 1-2. Amazon FSx CSI 드라이버 설치 #### 
Helm을 사용하여 EKS 클러스터에 FSx for Lustre CSI 드라이버를 배포한다. 
AWS FSx CSI 드라이버의 이미지는 각 리전별 AWS 전용 ECR 레포지토리에서 가져와야 한다.(image.repository)
```
kubectl create namespace fsx-csi-driver
helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver
helm repo update

helm install fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver \
    --namespace fsx-csi-driver \
    --set image.repository=602401143452.dkr.ecr.${AWS_REGION}.amazonaws.com/eks/aws-fsx-csi-driver \
    --set controller.serviceAccount.create=false \
    --set controller.serviceAccount.name=fsx-csi-driver-controller-sa \
    --set controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}
```

#### 1-3. StorageClass 및 Persistent Volume Claim (PVC) 배포 ####
```
FSxID=$(aws fsx describe-file-systems --file-system-ids ${FSX_ID} --query "FileSystems[0].{FileSystemId:FileSystemId}" --output text)
DNSNAME=$(aws fsx describe-file-systems --file-system-ids ${FSX_ID} --query "FileSystems[0].{DNSName:DNSName}" --output text)
MOUNTNAME=$(aws fsx describe-file-systems --file-system-ids ${FSX_ID} --query "FileSystems[0].{MountName:LustreConfiguration.MountName}" --output text)
```

동적 프로비저닝을 위해 SC, PV, PVC를 생성한다.  
```
cat << EOF > fsx-pvc.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-sc
provisioner: fsx.csi.aws.com
reclaimPolicy: Retain # PVC 삭제 시 FSx는 유지 (안전)
volumeBindingMode: Immediate
---
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
    volumeHandle: ${FSxID}
    volumeAttributes:
      dnsname: ${DNSNAME}
      mountname: ${MOUNTNAME}
---
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
EOF
```
```
kubectl apply -f fsx-pvc.yaml
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
aws s3 cp test-file.txt s3://${BUCKET_NAME}/

# Pod 내부 접속
kubectl exec -it pod-fsx -- /bin/bash

cd /
ls -l
```

```
nc -zv fs-04cb64a224a31f75d.fsx.ap-northeast-2.amazonaws.com 988
```




#### 오류메시지 ####
```
Warning  FailedMount  25s (x2 over 2m6s)  kubelet            MountVolume.SetUp failed for volume "fsx-pv" : rpc error: code = Internal desc = Could not mount "fs-04cb64a224a31f75d.fsx.ap-northeast-2.amazonaws.com@tcp:/ddyezbev" at "/var/lib/kubelet/pods/54547989-658e-4990-90f5-e98ef9634e87/volumes/kubernetes.io~csi/fsx-pv/mount": mount failed: exit status 22
Mounting command: mount
Mounting arguments: -t lustre fs-04cb64a224a31f75d.fsx.ap-northeast-2.amazonaws.com@tcp:/ddyezbev /var/lib/kubelet/pods/54547989-658e-4990-90f5-e98ef9634e87/volumes/kubernetes.io~csi/fsx-pv/mount
Output: mount.lustre: mount fs-04cb64a224a31f75d.fsx.ap-northeast-2.amazonaws.com@tcp:/ddyezbev at /var/lib/kubelet/pods/54547989-658e-4990-90f5-e98ef9634e87/volumes/kubernetes.io~csi/fsx-pv/mount failed: Invalid argument
This may have multiple causes.
Is 'ddyezbev' the correct filesystem name?
Are the mount options correct?
Check the syslog for more info.
```


## 리소스 삭제 ##
```
export CLUSTER_NAME="training-on-eks"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET_NAME="training-on-eks-lustre-${ACCOUNT_ID}"
export ROLE_NAME="FSx_Lustre_CSI_Driver_Role"

echo "=== EKS FSx for Lustre 리소스 삭제 시작 ==="

# 1. eksctl iamserviceaccount 삭제 (IAM Role과 연관성 제거)
echo "1. eksctl iamserviceaccount 삭제 중..."
eksctl delete iamserviceaccount \
    --name fsx-csi-driver-controller-sa \
    --namespace fsx-csi-driver \
    --cluster ${CLUSTER_NAME} \
    --wait

# 2. 커스텀 S3 정책 삭제
echo "2. IAM 정책(FSxLustreS3IntegrationPolicy) 삭제 중..."
POLICY_ARN=$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='FSxLustreS3IntegrationPolicy'].Arn" --output text)
if [ "$POLICY_ARN" != "None" ] && [ -n "$POLICY_ARN" ]; then
    # 역할이 아직 남아있을 경우를 대비해 연결 해제 시도
    aws iam detach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}" 2>/dev/null
    aws iam delete-policy --policy-arn "${POLICY_ARN}"
    echo "정책 삭제 완료: ${POLICY_ARN}"
fi

# 3. FSx_Lustre_CSI_Driver_Role이 남아있을 경우 직접 삭제
echo "3. IAM 역할(${ROLE_NAME}) 확인 및 삭제 중..."
if aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
    # 연결된 모든 정책 해제
    for p_arn in $(aws iam list-attached-role-policies --role-name "${ROLE_NAME}" --query 'AttachedPolicies[].PolicyArn' --output text); do
        aws iam detach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${p_arn}"
    done
    aws iam delete-role --role-name "${ROLE_NAME}"
    echo "역할 삭제 완료."
fi

# 4. FSx for Lustre 파일 시스템 삭제
echo "4. FSx for Lustre 파일 시스템 삭제 중..."
FSX_IDS=$(aws fsx describe-file-systems --query "FileSystems[?FileSystemType=='LUSTRE'].FileSystemId" --output text)
for fs_id in $FSX_IDS; do
    aws fsx delete-file-system --file-system-id "${fs_id}"
    echo "삭제 요청됨: ${fs_id} (완전 삭제까지 시간이 소요될 수 있습니다)"
done

# 5. 보안 그룹 삭제 (FSx가 완전히 삭제된 후에만 가능하므로 지연 발생 가능)
echo "5. 보안 그룹(fsx-lustre-sg) 삭제 중..."
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=fsx-lustre-sg" --query "SecurityGroups[0].GroupId" --output text)
if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
    # 의존성 문제로 바로 삭제 안 될 수 있으므로 시도만 함
    aws ec2 delete-security-group --group-id "${SG_ID}" 2>/dev/null || echo "알림: FSx 삭제가 완료될 때까지 보안 그룹은 잠시 후 다시 삭제가 필요할 수 있습니다."
fi

# 6. S3 버킷 삭제 (내부 객체 모두 삭제 후 버킷 제거)
echo "6. S3 버킷(${BUCKET_NAME}) 비우기 및 삭제 중..."
if aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    aws s3 rm "s3://${BUCKET_NAME}" --recursive
    aws s3 rb "s3://${BUCKET_NAME}" --force
    echo "버킷 삭제 완료."
fi

echo "=== 모든 리소스 삭제 요청 완료 ==="
```


## 레퍼런스 ##

* https://github.com/kubernetes-sigs/aws-fsx-csi-driver/blob/master/docs/install.md
* https://aws.amazon.com/ko/blogs/tech/lustre/

