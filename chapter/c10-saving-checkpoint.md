대규모 분산 훈련에서 체크포인트 저장시 핵심 포인트는 데이터 저장을 위한 네트워크 대역폭(Throughput) 및 I/O 성능을 최대화 하는 것이다. 수천 개의 GPU 노드가 동시에 쓰기 작업을 수행할 때 성능 병목을 줄이기 위해서는 병렬 및 비동기 저장 (Parallel & Asynchronous Checkpointing) 기법 활용이 중요하며, 이를 통해 체크포인트를 저장하는 동안 GPU 연산이 중단되지 않도록 하는 것이다.  
또한 PyTorch Distributed Checkpoint(DCP)를 활용하는 경우 각 Rank(프로세스)가 자신의 가중치만 저장하게 되고, 그 대상 경로를 FSx 러스터로 설정함으로써 쓰기 속도를 최대화 할 수 있다. 


## 2. 고성능 병렬 공유 스토리지(Lustre) 기반 체크포인팅 설계

**Lustre(예: AWS FSx for Lustre)**는 대규모 분산 컴퓨팅(HPC, AI 학습)을 위해 설계된 분산 병렬 파일시스템입니다. 모든 학습 노드가 단일 네임스페이스(Single Namespace)를 공유하면서도, 데이터를 여러 스토리지 서버(OST)에 분산 저장해 높은 처리량(Throughput)을 제공합니다.

---

### 2-1. Lustre 기반 체크포인팅의 핵심 장점

#### 1. 아키텍처 및 복구 로직의 단순성 (Zero Staging)
* **별도 업로드/다운로드 파이프라인 불필요:** 로컬 NVMe에 먼저 쓰고 백그라운드로 S3에 올리는 워커 프로세스나, 장애 시 전체 파일을 다시 내려받는 스테이징(`restore.tmp/`) 과정이 필요 없습니다.
* **즉시 재개(Instant Resume):** 장애 복구 시 모든 노드가 이미 마운트된 공유 경로(예: `/fsx/ckpt/step_0001000/`)를 즉시 바라보므로, 데이터 복사 대기 시간 없이 곧바로 메모리로 로드하여 학습을 재개할 수 있습니다.

#### 2. 분산 체크포인트(DCP) 표준 전제와의 완벽한 호환
* PyTorch DCP(`torch.distributed.checkpoint`)나 Megatron-LM은 기본적으로 **모든 Rank가 동일한 디렉토리를 공유하는 POSIX 파일시스템**을 전제로 설계되었습니다.
* Rank 0이 기록하는 `.metadata`, `common.pt`와 각 Rank가 쓰는 `.distcp` 파일이 한 공간에 모이므로, 병렬 구성(TP/PP/DP)을 변경해 재시작할 때도 추가 파일 취합 과정 없이 프레임워크 기본 기능만으로 자동 리샤딩(Resharding)됩니다.

#### 3. 병렬 I/O 처리량 확장성
* 일반 NFS/EFS와 달리 클라이언트(학습 노드)가 메타데이터 서버(MDS)에서 위치 정보만 받은 뒤, 실제 데이터는 여러 **OST(Object Storage Target)**와 직접 병렬 통신하여 읽고 씁니다.
* 스토리지 용량을 늘릴수록 OST 개수와 네트워크 대역폭이 선형적으로 확장되어 수십 GB/s급의 I/O 성능을 확보할 수 있습니다.

---

### 2-2. 실무 운영 방법 및 최적화 전략

Lustre를 안정적으로 운영하려면 파일 분산 배치(Striping)와 영구 백업(S3 연동) 설정이 필수적입니다.

#### 1. 파일 스트라이핑(Striping) 최적화 (`lfs setstripe`)
* 체크포인트 파일(`.distcp`)은 수 GB~수십 GB에 달하는 대용량 파일이므로, 단일 OST에 병목이 몰리지 않도록 여러 OST에 쪼개어 저장해야 합니다.
* 체크포인트 디렉토리에 미리 스트라이프 설정을 적용해 하위 파일들이 자동으로 분산되도록 구성합니다.
  ```bash
  # 체크포인트 저장 디렉토리에 스트라이핑 설정 적용
  # -c -1: 가용한 모든 OST에 분산 저장 / -S 16M: 스트라이프 단위 크기 16MB
  lfs setstripe -c -1 -S 16M /fsx/ckpt/
  ```

#### 2. S3 데이터 리포지토리 연동 (DRA: Data Repository Association)
* Lustre는 고성능 스크래치(Scratch) 스토리지 성격이 강하므로, 장애 대비 및 장기 보관을 위해 S3와 양방향 동기화를 설정합니다.
* **자동 내보내기(Auto-Export):** Lustre에 새 체크포인트가 기록되면 백그라운드에서 S3 버킷으로 자동 아카이빙되도록 설정합니다.
* **지연 로딩(Lazy Loading):** S3에 백업된 과거 체크포인트로 롤백해야 할 경우, Lustre가 메타데이터만 유지하다가 학습 프로세스가 파일을 읽는 순간 S3에서 필요한 블록만 고속으로 당겨옵니다.

#### 3. 용량 관리 및 보존 정책 (Retention Policy)
* Lustre는 NVMe나 S3에 비해 단위 용량당 비용이 높으므로, 전체 학습 이력을 모두 쌓아두지 않습니다.
* **슬라이딩 윈도우 삭제:** 최신 **2~3개 스텝**만 Lustre에 유지하고, S3 백업(`Export`)이 완료된 이전 체크포인트는 Lustre에서 파일 본문만 비우거나(`lfs hsm_release`) 오래된 디렉토리를 주기적으로 삭제하여 가용 용량을 70~80% 이하로 관리합니다.

#### 4. 메타데이터 병목 완화
* 수백 개 Rank가 동시에 수천 개의 작은 파일을 생성하면 MDS(Metadata Server)에 부하가 집중됩니다.
* 체크포인트 저장 시 프레임워크 옵션에서 Rank당 파일 수를 최소화하고, 디렉토리 구조를 단순하게(예: `/fsx/ckpt/iter_{N}/`) 유지합니다.

---

### 2-3. 요약: 2-Tier(NVMe + S3) vs Lustre 선택 기준

| 비교 항목 | 1. 2-Tier (로컬 NVMe + S3) | 2. 병렬 공유 스토리지 (Lustre) |
| :--- | :--- | :--- |
| **스토리지 비용** | **매우 저렴** (인스턴스 기본 NVMe + S3) | **높음** (고성능 프로비저닝 용량 비용 발생) |
| **저장 속도** | **매우 빠름** (로컬 디스크 대역폭 온전히 사용) | **빠름** (OST 수와 네트워크 스펙에 비례) |
| **복구 속도** | 보통 (S3 전체 다운로드 시간 4~7분 소요) | **매우 빠름** (공유 경로에서 즉시 로드) |
| **구현·운영 복잡도** | 높음 (백그라운드 업로더, 폴더 격리, GC 구현 필요) | **낮음** (POSIX 표준 경로 그대로 사용) |





## Lustre에 직접 쓸 때 병목이 생기는 이유 ##

* 동시 쓰기 경합(Burst Write Contention): 학습이 체크포인트 스텝에 도달하는 순간, 8대 노드(64개 GPU)가 동시에 수십 GB~수 TB의 데이터를 네트워크를 통해 Lustre의 OST(Object Storage Target)와 MDS(Metadata Server)로 한꺼번에 쏟아붓습니다.
* 메타데이터 락(Metadata Lock) 부하: 분산 체크포인트는 수십 개의 파일(.distcp, .metadata 등)을 동시에 생성하고 기록합니다. 수많은 Rank가 동시에 파일을 열거나 크기를 갱신할 때 Lustre의 메타데이터 서버(MDS)에 락 경합이 걸려 쓰기 속도가 급격히 떨어집니다.
* 비용 대비 대역폭 한계: 64개 Rank가 로컬 NVMe에 쓰는 합산 대역폭(수십 GB/s)만큼의 쓰기 성능을 Lustre에서 뽑아내려면, 스토리지 용량과 IOPS 스펙을 매우 높게 프로비저닝해야 하므로 인프라 비용이 기하급수적으로 늘어납니다.


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

테라폼으로 이미 생성된 러스터 파일 시스템을 조회한다.  
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

아래의 스크립트는 러스터 파일 시스템 상세 정보를 출력해 준다.
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
워크샵에서 생성된 러스터 파일 시스템은 SCRATCH_2 로 EFA 를 지원하지 않는다. 실제 분산훈련 환경에서는 배포 유형 (Deployment Type)을 반드시 PERSISTENT_2를 선택해야 한다.
다음은 러스터에서 EFA 인터페이스와 연관된 설정들 이다.
   
* deployment_type: 반드시 PERSISTENT_2
* storage_capacity: 최소 1.2 TiB(1200 GiB) 이상이어야 하며, 1.2의 배수 단위로 설정
* throughput_capacity: 최소 125 MB/s/TiB 이상 권장.

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

### 3. Amazon FSx CSI (Container Storage Interface) 드라이버 설치 ### 
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
fsx-csi-node 는 실제 워커 노드마다 하나씩 실행되는 데몬셋으로, EC2 노드 위에서 Lustre 파일 시스템을 실제로 마운트(Mount)하는 역할을 수행한다.
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

만약 러스터 관련 오류가 발생하는 경우 아래의 명령어로 fsx-csi-node 로그를 조회하면 그 원인을 쉽게 파악할 수 있다.
```
kubectl logs -f -l app=fsx-csi-node -n fsx-csi-driver -c fsx-plugin
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

## 자동 백업 & 삭제 보호 ##

분산 훈련시 lustre 파일 시스템에는 체크포인트 데이터와 훈련 데이터를 저장하고 있으므로 자동 백업 가능과 삭제 보호 기능을 활성화 해야한다.  

### 1. 자동 백업 ###

#### FSx 자체 백업 기능 (Persistent 파일 시스템 대상) #### 
S3 데이터 리포지토리와 연결되지 않은 영구(Persistent) 파일 시스템의 경우, Amazon FSx는 다음 두 가지 타입의 백업을 지원한다. 
* 자동 일일 백업 (Automatic daily backups): 사용자가 설정한 백업 기간(1~90일, 기본 30일) 동안 매일 자동으로 백업이 수행된다.
* 사용자 시작 백업 (User-initiated backups): 필요시 수동으로 특정 시점 백업을 생성할 수 있다. 
이 두 가지 모두 블록 기반의 증분 백업으로 S3에 저장되어 비용 효율적이며 내구성이 뛰어나다. 
```
aws fsx describe-file-systems --file-system-ids <파일-시스템-ID> --query "FileSystems[0].LustreConfiguration"

aws fsx update-file-system \
  --file-system-id <파일-시스템-ID> \
  --lustre-configuration "AutomaticBackupRetentionDays=<유지-일수>,DailyAutomaticBackupStartTime=<HH:MM>"
```

#### S3 데이터 리포지토리와의 자동 동기화 (Automatic Import/Export) #### 
파일 시스템이 S3 버킷에 연결된 경우, "자동 내보내기(Automatic export)" 기능을 설정하여 파일 시스템의 변경 사항(파일 추가, 변경, 삭제)이 S3 버킷에 자동으로 업데이트되도록 할 수 있다. 이는 엄밀히 말해 전통적인 "백업"이라기보다는 지속적인 데이터 동기화 메커니즘이다 

```
# 파일 시스템의 /ns1/ 경로를 S3 버킷 s3://my-bucket/data/와 연결하는 예시
aws fsx create-data-repository-association \
  --file-system-id <파일-시스템-ID> \
  --file-system-path /ns1/ \
  --data-repository-path s3://<버킷-이름>/<폴더>/ \
  --s3 '{"AutoImportPolicy": {"Events": ["NEW", "CHANGED"]}, "AutoExportPolicy": {"Events": ["NEW", "CHANGED"]}}'
```

### 2. 삭제 보호 ###

RDS 나 EC2 와는 달리 lustre 의 경우 삭제 보호 기능이 제공되지 않는다. 
삭제 보호 효과를 내는 유일한 대안은 특정 사용자나 스크립트가 실수로 삭제하지 못하도록 fsx:DeleteFileSystem 권한을 차단하는 것이다. 이렇게 하면 콘솔에서 삭제 버튼을 누르거나 CLI로 명령을 내려도 거부된다.

[policy.json]
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PreventFSxDeletion",
            "Effect": "Deny",
            "Action": "fsx:DeleteFileSystem",
            "Resource": "arn:aws:fsx:region:account-id:file-system/fs-xxxxxxxxxxxxxxxxx"
        }
    ]
}
```
```
aws iam create-policy \
    --policy-name FSxDeleteProtectionPolicy \
    --policy-document file://policy.json

aws iam attach-user-policy \
    --user-name <사용자-이름> \
    --policy-arn arn:aws:iam::<계정-ID>:policy/FSxDeleteProtectionPolicy

aws iam attach-role-policy \
    --role-name <역할-이름> \
    --policy-arn arn:aws:iam::<계정-ID>:policy/FSxDeleteProtectionPolicy
```


## 레퍼런스 ##

* https://github.com/kubernetes-sigs/aws-fsx-csi-driver/blob/master/docs/install.md
* https://aws.amazon.com/ko/blogs/tech/lustre/

