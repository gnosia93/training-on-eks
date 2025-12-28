분산 훈련 환경에서 Lustre를 스토리지로 사용할 때, 체크포인트 저장의 핵심은 대역폭(Throughput) 최적화와 데이터 일관성이다.
수천 개의 GPU 노드가 동시에 쓰기 작업을 수행할 때 성능 병목을 줄이기 위해서는 병렬 및 비동기 저장 (Parallel & Asynchronous Checkpointing) 기법을 활용하여 
체크포인트를 저장하는 동안 GPU 연산이 중단되지 않도록 하는 것이다.  
또한 PyTorch Distributed Checkpoint (DCP)를 활용하여 각 Rank(프로세스)가 자신의 가중치만 별도의 파일로 저장하게 함으로써 FSx와 같은 병렬 파일 시스템 아키텍처를 최대한 활용하여 쓰기 속도를 극대화하는 것이다. 

## 러스터(Lustre) ##
러스터(Lustre) 파일 시스템은 높은 처리량, 낮은 지연 시간, 뛰어난 확장성을 제공하는 병렬 분산 파일 시스템으로, 대규모 데이터셋을 처리해야 하는 AI 시스템에 필수적이다.
AWS 에서 Lustre 파일 시스템을 사용하는 가장 빠른 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 와 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

```
export CLUSTER_NAME="training-on-eks"
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
```

### 1. lustre 파일시스템 생성 ###
```
# 첫 번째 프라이빗 서브넷 ID 가져오기
export PRIV_SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
        "Name=tag:Name,Values=*priv-subnet-1*" \
        --query "Subnets[0].SubnetId" --output text)
export BUCKET_NAME="training-on-eks-lustre-${ACCOUNT_ID}"

# S3 버킷 생성
aws s3 mb s3://${BUCKET_NAME} --region ${AWS_REGION}           

# FSX 시큐리티 그룹 생성
NODE_SG_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} \
    --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)
FSX_SG_ID=$(aws ec2 create-security-group --group-name fsx-lustre-sg \
    --description "Allow Lustre traffic" --vpc-id ${VPC_ID} --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id ${FSX_SG_ID} \
    --protocol tcp --port 988  --source-group ${NODE_SG_ID}
aws ec2 authorize-security-group-ingress --group-id ${FSX_SG_ID} \
    --protocol tcp --port 1018-1023 --source-group ${NODE_SG_ID}
aws ec2 authorize-security-group-ingress --group-id ${FSX_SG_ID} --protocol -1 --port -1 --source-group ${FSX_SG_ID}
aws ec2 authorize-security-group-egress --group-id ${FSX_SG_ID} --protocol -1 --port -1 --source-group ${FSX_SG_ID}

# 테스트용 FSx for Lustre 생성 (SCRATCH_2, 1200 GiB, ENI)
aws fsx create-file-system \
    --file-system-type LUSTRE \
    --storage-capacity 1200 \
    --subnet-ids ${PRIV_SUBNET_ID} \
    --security-group-ids ${FSX_SG_ID} \
    --lustre-configuration "DeploymentType=SCRATCH_2,\
        ImportPath=s3://${BUCKET_NAME},\
        ExportPath=s3://${BUCKET_NAME}/export,\
        AutoImportPolicy=NEW_CHANGED_DELETED}" \
    --query "FileSystem.FileSystemId" --output text


# 운영용 FSx for Lustre 생성 (PERSISTENT_2, 38400 GiB, EFA)
#FSx_ID=$(aws fsx create-file-system \
#    --file-system-type LUSTRE \
#    --storage-capacity 38400 \
#    --subnet-ids ${PRIV_SUBNET_ID} \
#    --security-group-ids ${FSX_SG_ID} \
#    --lustre-configuration "DeploymentType=PERSISTENT_2,\
#        ImportPath=s3://${BUCKET_NAME},\
#        ExportPath=s3://${BUCKET_NAME}/export,\
#        AutoImportPolicy=NEW_CHANGED,\
#        EfaEnabled=true, \
#        PerUnitStorageThroughput=125, \
#        MetadataConfiguration={Mode=AUTOMATIC}" \
#    --query "FileSystem.FileSystemId" --output text)

echo "FSx File System Creating: ${FSx_ID}"
```

### 2. IAM 역할(IRSA) 생성 ###
이 명령은 IAM Role 생성, 정책 연결, 서비스 어카운트 생성 및 annontation 처리를 한 번에 수행한다
```
eksctl create iamserviceaccount \
    --name fsx-csi-sa \
    --namespace fsx-csi \
    --cluster ${CLUSTER_NAME} \
    --role-name "FSxLustreRole" \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
    --approve \
    --override-existing-serviceaccounts

# FSxLustreRole 에 S3 접근권한 부여
cat <<EOF > s3-policy.json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "s3:GetBucketLocation",
            "s3:ListBucket",
            "s3:GetBucketAcl",
            "s3:GetObject",
            "s3:GetObjectTagging",
            "s3:PutObject",
            "s3:DeleteObject"
        ],
        "Resource": ["arn:aws:s3:::${BUCKET_NAME}","arn:aws:s3:::${BUCKET_NAME}/*"]
    }]
}
EOF

S3_POLICY_ARN=$(aws iam create-policy --policy-name FSxLustreS3Policy --policy-document file://s3-policy.json --query Policy.Arn --output text)
aws iam attach-role-policy --role-name "FSxLustreRole" --policy-arn $S3_POLICY_ARN
```

### 3. lustre 파일시스템 조회 ###
```
aws fsx describe-file-systems \
    --file-system-ids ${FSx_ID} \
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
|  fs-04e6429ada0540632.fsx.ap-northeast-2.amazonaws.com |  fs-04e6429ada0540632  |  klje3bev  |
+--------------------------------------------------------+------------------------+------------+
```

### 4. Amazon FSx CSI 드라이버 설치 ### 
Helm을 사용하여 EKS 클러스터에 FSx for Lustre CSI 드라이버를 배포한다. 
```
kubectl create namespace fsx-csi-driver
helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver
helm repo update

helm install fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver \
    --namespace fsx-csi \
    --set image.repository=602401143452.dkr.ecr.${AWS_REGION}.amazonaws.com/eks/aws-fsx-csi-driver \
    --set controller.serviceAccount.create=false \
    --set controller.serviceAccount.name=fsx-csi-sa \
    --set controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::${ACCOUNT_ID}:role/FSxLustreRole
```

### 5. PV/PVC 배포 ###
```
#FSxID=$(aws fsx describe-file-systems --file-system-ids ${FSX_ID} --query "FileSystems[0].{FileSystemId:FileSystemId}" --output text)
FSx_DNS=$(aws fsx describe-file-systems --file-system-ids ${FSx_ID} --query "FileSystems[0].{DNSName:DNSName}" --output text)
FSx_MOUNT=$(aws fsx describe-file-systems --file-system-ids ${FSx_ID} --query "FileSystems[0].{MountName:LustreConfiguration.MountName}" --output text)
echo ${FSx_ID} ${FSx_DNS} ${FSx_MOUNT}

cat << EOF > fsx-pvc.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-sc
provisioner: fsx.csi.aws.com
reclaimPolicy: Retain                     # PVC 삭제 시 FSx는 유지 (안전)
volumeBindingMode: Immediate
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: fsx-pv
spec:
  capacity:
    storage: 38400Gi # FSx 생성 용량과 일치시킴
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fsx-sc
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: ${FSx_ID}
    volumeAttributes:
      dnsname: ${FSx_DNS}
      mountname: ${FSx_MOUNT}
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
      storage: 38400Gi
  volumeName: fsx-pv 
EOF

kubectl apply -f fsx-pvc.yaml
```

### 6. 생성 상태 확인 ###
```
echo "FSx 파일 시스템 [$FSx_ID] 상태를 확인합니다..."
while true; do
    STATUS=$(aws fsx describe-file-systems \
        --file-system-ids "$FSx_ID" \
        --query "FileSystems[0].Lifecycle" \
        --output text 2>/dev/null)
    # 결과가 없는 경우 종료
    if [ -z "$STATUS" ] || [ "$STATUS" == "None" ]; then
        echo "오류: 해당 ID를 찾을 수 없습니다."
        break
    fi

    echo "[$(date +%H:%M:%S)] 현재 상태: $STATUS"
    if [ "$STATUS" == "AVAILABLE" ]; then
        echo "축하합니다! 파일 시스템이 준비되었습니다."
        break
    elif [ "$STATUS" == "FAILED" ]; then
        echo "오류: 파일 시스템 생성에 실패했습니다."
        break
    fi

    sleep 5
done
```

### 7. 러스터 성능 조회 ####
```
aws fsx describe-file-systems --query "FileSystems[?FileSystemType=='LUSTRE']" --output json | jq -r '
  ["ID", "Status", "Storage_GiB", "Unit_MB/s", "Total_MB/s", "MountName", "Type"],
  (.[] | [
    .FileSystemId, 
    .Lifecycle, 
    .StorageCapacity, 
    .LustreConfiguration.PerUnitStorageThroughput, 
    ((.StorageCapacity / 1024) * .LustreConfiguration.PerUnitStorageThroughput), 
    .LustreConfiguration.MountName, 
    .LustreConfiguration.DeploymentType
  ]) | @tsv' | column -t -s $'\t'
```
[결과]
```
ID                    Status     Storage_GiB  Unit_MBps  Total_MBps  MountName  Type
fs-07ac8c09c8b9e1981  AVAILABLE  38400        125        4687.5      nelu5bev   PERSISTENT_2
```

## Pod 마운트 테스트 ##
```
cat <<EOF > pod-fsx.yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-fsx
spec:
  containers:
  - name: app
    image: public.ecr.aws/amazonlinux/amazonlinux:2023
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: fsx
      mountPath: /data/fsx
  volumes:
    - name: fsx
      persistentVolumeClaim:
        claimName: fsx-pvc # 위에서 생성한 PVC 이름
EOF

kubectl apply -f pod-fsx.yaml
```

S3 에 파일을 업로드 하고 Pod 에서 조회되는지 확인한다.  
```
echo "Hello FSx Lustre" > test-file.txt
aws s3 cp test-file.txt s3://${BUCKET_NAME}/
kubectl exec -it pod-fsx -- bash -c "cd /data/fsx && ls -l"
```

## 리소스 삭제 ##
```
export CLUSTER_NAME="training-on-eks"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET_NAME="training-on-eks-lustre-${ACCOUNT_ID}"
export ROLE_NAME="FSxLustreRole"

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
POLICY_ARN=$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='FSxLustreS3Policy'].Arn" --output text)
if [ "$POLICY_ARN" != "None" ] && [ -n "$POLICY_ARN" ]; then
    # 역할이 아직 남아있을 경우를 대비해 연결 해제 시도
    aws iam detach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}" 2>/dev/null
    aws iam delete-policy --policy-arn "${POLICY_ARN}"
    echo "정책 삭제 완료: ${POLICY_ARN}"
fi

# 3. FSx_Lustre_CSI_Driver_Role이 남아있을 경우 직접 삭제
echo "3. IAM 역할(${ROLE_NAME}) 삭제 중..."
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

# 5. S3 버킷 삭제 (내부 객체 모두 삭제 후 버킷 제거)
echo "5. S3 버킷(${BUCKET_NAME}) 비우기 및 삭제 중..."
if aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    aws s3 rm "s3://${BUCKET_NAME}" --recursive
    aws s3 rb "s3://${BUCKET_NAME}" --force
    echo "버킷 삭제 완료."
fi

# 6. 시큐리티 그룹 삭제
echo "6. 보안 그룹(fsx-lustre-sg) 삭제 중..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=fsx-lustre-sg" \
    --query "SecurityGroups[0].GroupId" \
    --output text)

if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
    echo "보안 그룹 발견: $SG_ID. 규칙 삭제 및 그룹 삭제를 시작합니다..."

    # 인바운드 규칙 삭제
    RULES=$(aws ec2 describe-security-groups --group-ids "$SG_ID" --query "SecurityGroups[0].IpPermissions" --output json)
    if [ "$INGRESS_RULES" != "[]" ] && [ "$INGRESS_RULES" != "null" ]; then
        # 인바운드 규칙 삭제 실행
        aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --ip-permissions "$RULES"
        echo "성공: 모든 인바운드 규칙이 제거되었습니다."
    fi

    # 아웃바운드 규칙 삭제
    EGRESS_RULES=$(aws ec2 describe-security-groups --group-ids "$SG_ID" --query "SecurityGroups.IpPermissionsEgress" --output json)
    if [ "$EGRESS_RULES" != "[]" ] && [ "$EGRESS_RULES" != "null" ]; then
        # 아웃바운드 규칙 삭제 실행 (revoke-security-group-egress)
        aws ec2 revoke-security-group-egress --group-id "$SG_ID" --ip-permissions "$EGRESS_RULES"
        echo "성공: 모든 아웃바운드 규칙이 제거되었습니다."
    fi

    # 보안 그룹 최종 삭제
    aws ec2 delete-security-group --group-id $SG_ID
    echo "성공적으로 삭제되었습니다."
else
    echo "삭제할 보안 그룹(fsx-lustre-sg)이 존재하지 않습니다."
fi

echo "=== 모든 리소스 삭제 요청 완료 ==="
```

## 레퍼런스 ##

* https://github.com/kubernetes-sigs/aws-fsx-csi-driver/blob/master/docs/install.md
* https://aws.amazon.com/ko/blogs/tech/lustre/

