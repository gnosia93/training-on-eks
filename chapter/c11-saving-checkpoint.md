대규모 분산 훈련에서 체크포인트 저장시 중요한 핵심 포인트는 네트워크 대역폭(Throughput) 및 I/O 성능을 최적화하고 데이터 일관성을 보장하는 것이다. 수천 개의 GPU 노드가 동시에 쓰기 작업을 수행할 때 성능 병목을 줄이기 위해서는 병렬 및 비동기 저장 (Parallel & Asynchronous Checkpointing) 기법 활용이 중요하며, 이를 통해 체크포인트를 저장하는 동안 GPU 연산이 중단되지 않도록 하는 것이다.  
또한 PyTorch Distributed Checkpoint (DCP)를 활용하여 각 Rank(프로세스)가 자신의 가중치만 별도의 파일로 저장하게 함으로써 FSx와 같은 병렬 파일 시스템 아키텍처를 최대한 활용하여 쓰기 속도를 극대화할 필요가 있다. 

## 러스터(Lustre) ##
러스터(Lustre) 파일 시스템은 높은 처리량, 낮은 지연 시간, 뛰어난 확장성을 제공하는 병렬 분산 파일 시스템으로, 대규모 데이터셋을 처리해야 하는 AI 시스템에 필수적이다.
AWS 에서 Lustre 파일 시스템을 사용하는 가장 빠른 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 와 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

```
export CLUSTER_NAME="training-on-eks"
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
export FSX_ROLE="FSxLustreRole"
export FSX_S3Policy="FSxLustreS3Policy"
export S3_BUCKET="training-on-eks-lustre-${ACCOUNT_ID}"
```

### 1. lustre 파일시스템 조회 ###
테라폼 에서 이미 러스터 클러스터를 생성하였다. 

* 러스터 파일 시스템 조회
```
aws fsx describe-file-systems \
    --query "FileSystems[?Tags[?Key=='Name' && Value=='trainng-on-eks']].\
             {ID:FileSystemId, MountName:LustreConfiguration.MountName, DNS:DNSName, Status:Lifecycle}" \
    --output table
```
[결과]
```
-------------------------------------------------------------------------------------------------------------
|                                            DescribeFileSystems                                            |
+--------------------------------------------------------+-----------------------+------------+-------------+
|                           DNS                          |          ID           | MountName  |   Status    |
+--------------------------------------------------------+-----------------------+------------+-------------+
|  fs-0261bb15621d24e21.fsx.ap-northeast-1.amazonaws.com |  fs-0261bb15621d24e21 |  rycutbev  |  AVAILABLE  |
+--------------------------------------------------------+-----------------------+------------+-------------+
```

* 러스터 파일 시스템 성능조회
```
aws fsx describe-file-systems --query "FileSystems[?FileSystemType=='LUSTRE']" --output json | jq -r '
  ["ID", "Status", "Storage_GiB", "Unit_MB/s", "Total_MB/s", "MountName", "Type"],
  (.[] | 
    (.LustreConfiguration.PerUnitStorageThroughput // 0 | tonumber) as $unit |
    [
      .FileSystemId, 
      .Lifecycle, 
      .StorageCapacity, 
      (if $unit == 0 then "Default" else $unit end), 
      (if $unit == 0 then "Variable" else (($unit * .StorageCapacity / 1024) | floor) end), 
      .LustreConfiguration.MountName, 
      .LustreConfiguration.DeploymentType
    ]
  ) | @tsv' | column -t -s $'\t'
```
[결과]
```
ID                    Status     Storage_GiB  Unit_MB/s  Total_MB/s  MountName  Type
fs-0261bb15621d24e21  AVAILABLE  1200         Default    Variable    rycutbev   SCRATCH_2
```

### 2. IAM 역할(IRSA) 생성 ###
fsx 용 서비스 어카운트를 생성한다.  
```
kubectl create namespace fsx-csi-driver

eksctl create iamserviceaccount \
    --name fsx-csi-sa \
    --namespace fsx-csi-driver \
    --cluster ${CLUSTER_NAME} \
    --role-name "${FSX_ROLE}" \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
    --approve \
    --override-existing-serviceaccounts
```

FSxLustreRole 에 S3 접근 권한을 부여한다. 
```
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

S3_POLICY_ARN=$(aws iam create-policy --policy-name ${FSX_S3Policy} --policy-document file://s3-policy.json --query Policy.Arn --output text)
aws iam attach-role-policy --role-name ${FSX_ROLE} --policy-arn $S3_POLICY_ARN
```

### 3. Amazon FSx CSI 드라이버 설치 ### 
Helm을 사용하여 EKS 클러스터에 FSx for Lustre CSI 드라이버를 배포한다. 
```
helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver
helm repo update

helm install fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver \
    --namespace fsx-csi-driver \
    --set image.repository=602401143452.dkr.ecr.${AWS_REGION}.amazonaws.com/eks/aws-fsx-csi-driver \
    --set controller.serviceAccount.create=false \
    --set controller.serviceAccount.name=fsx-csi-sa \
    --set controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::${ACCOUNT_ID}:role/FSxLustreRole
```

fsx 관련 컨트롤러와 Pod 를 조회한다. fsx-csi-controller 는 파일 시스템의 생성, 삭제, 볼륨 연결 등을 담당하는 컨트롤러이다. 
fsx-csi-node 는 실제 워커 노드마다 하나씩 실행되는 DaemonSet 으로, EC2 노드 위에서 Lustre 파일 시스템을 실제로 마운트(Mount)하는 역할을 수행한다.
```
kubectl get pods -n fsx-csi-driver
```
[결과]
```
NAME                                 READY   STATUS    RESTARTS   AGE
fsx-csi-controller-9fb564f88-44n89   4/4     Running   0          5m35s
fsx-csi-controller-9fb564f88-tvk5q   4/4     Running   0          5m35s
fsx-csi-node-6r59f                   3/3     Running   0          5m35s
fsx-csi-node-s79mz                   3/3     Running   0          5m35s
fsx-csi-node-st5fc                   3/3     Running   0          5m35s
fsx-csi-node-wj7lj                   3/3     Running   0          5m35s
```



## EKS 연결하기 ##
### 1. PV/PVC 배포 ###
```
export FSx_ID=$(aws fsx describe-file-systems \
    --query "FileSystems[?Tags[?Key=='Name' && Value=='trainng-on-eks']].FileSystemId" --output text)
export FSx_DNS=$(aws fsx describe-file-systems \
    --query "FileSystems[?Tags[?Key=='Name' && Value=='trainng-on-eks']].DNSName" --output text)
export FSx_MOUNTNAME=$(aws fsx describe-file-systems \
    --query "FileSystems[?Tags[?Key=='Name' && Value=='trainng-on-eks']].LustreConfiguration.MountName" --output text)

echo ${FSx_ID} ${FSx_DNS} ${FSx_MOUNTNAME}
```

```
cat << EOF > fsx-pvc.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-sc
provisioner: fsx.csi.aws.com
reclaimPolicy: Retain                     # PVC 삭제되어도 유지 (Retain 으로 설정)
volumeBindingMode: Immediate
---
apiVersion: v1
kind: PersistentVolume                    # 이미 생성한 러스터 클러스터를 연결하기 위한 정적 프로비저닝. vs. 동적 프로비저닝의 경우 PV 대신 SC 에 러스터 정보 선언
metadata:
  name: fsx-pv
spec:
  capacity:
    storage: 1200Gi                       # FSx 생성 용량과 일치시킴 - 운영용은 38400
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain   # PVC 삭제되어도 유지 (Retain 으로 설정)
  storageClassName: fsx-sc
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: ${FSx_ID}
    volumeAttributes:
      dnsname: ${FSx_DNS}
      mountname: ${FSx_MOUNTNAME}
  mountOptions:
    - flock        # 파일 잠금 기능 활성화 (학습 시 필요)
    - lazystatfs   # 성능 최적화
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
      storage: 1200Gi                      # FSx 생성 용량과 일치시킴 - 운영용은 38400  
  volumeName: fsx-pv 
EOF

kubectl apply -f fsx-pvc.yaml
```

### 2. Pod 마운트 테스트 ###
```
cat <<EOF | kubectl apply -f - 
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
```

S3 에 파일을 업로드 하고 Pod 에서 조회되는지 확인한다.  
```
echo "Hello FSx Lustre" > test-file.txt
aws s3 cp test-file.txt s3://${S3_BUCKET}/
kubectl exec -it pod-fsx -- bash -c "cd /data/fsx && ls -l"
```

* fsx-node 로그 조회
```
kubectl logs -f -l app=fsx-csi-node -n fsx-csi-driver -c fsx-plugin
```


## 리소스 삭제 ##
```
export CLUSTER_NAME="training-on-eks"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET_NAME="training-on-eks-lustre-${ACCOUNT_ID}"

# 1. eksctl iamserviceaccount 삭제 (IAM Role과 연관성 제거)
echo "1. eksctl iamserviceaccount 삭제 중..."
eksctl delete iamserviceaccount \
    --name fsx-csi-sa \
    --namespace fsx-csi \
    --cluster ${CLUSTER_NAME} \
    --wait

# 2. 커스텀 S3 정책 삭제
echo "2. IAM 정책(FSxLustreS3IntegrationPolicy) 삭제 중..."
POLICY_ARN=$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${FSX_S3Policy}'].Arn" --output text)
if [ "$POLICY_ARN" != "None" ] && [ -n "$POLICY_ARN" ]; then
    # 역할이 아직 남아있을 경우를 대비해 연결 해제 시도
    aws iam detach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}" 2>/dev/null
    aws iam delete-policy --policy-arn "${POLICY_ARN}"
    echo "정책 삭제 완료: ${POLICY_ARN}"
else
    echo "${FSX_S3Policy} 정책이 이미 삭제되었습니다."
fi

# 3. FSx_Lustre_CSI_Driver_Role이 남아있을 경우 직접 삭제
echo "3. IAM 역할(${FSX_ROLE}) 삭제 중..."
if aws iam get-role --role-name "${FSX_ROLE}" 2>/dev/null; then
    # 연결된 모든 정책 해제
    for p_arn in $(aws iam list-attached-role-policies --role-name "${FSX_ROLE}" --query 'AttachedPolicies[].PolicyArn' --output text); do
        aws iam detach-role-policy --role-name "${FSX_ROLE}" --policy-arn "${p_arn}"
    done
    # 인라인 정책이 남아있으면 delete-role 명령이 실패합니다.
    INLINE_POLICIES=$(aws iam list-role-policies --role-name "${FSX_ROLE}" --query 'PolicyNames[]' --output text)
    for p_name in $INLINE_POLICIES; do
        aws iam delete-role-policy --role-name "${FSX_ROLE}" --policy-name "${p_name}"
        echo "인라인 정책 삭제 완료: ${p_name}"
    done 

    aws iam delete-role --role-name "${FSX_ROLE}"
    echo "역할 삭제 완료."
else
    echo "${FSX_ROLE} 이 이미 삭제되었습니다."
fi
```

## 레퍼런스 ##

* https://github.com/kubernetes-sigs/aws-fsx-csi-driver/blob/master/docs/install.md
* https://aws.amazon.com/ko/blogs/tech/lustre/

