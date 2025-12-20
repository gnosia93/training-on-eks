## 러스터(Lustre) ##
러스터(Lustre) 파일 시스템은 높은 처리량, 낮은 지연 시간, 뛰어난 확장성을 제공하는 병렬 분산 파일 시스템으로, 대규모 데이터셋을 처리해야 하는 AI 시스템에 필수적이다.
AWS 에서 Lustre 파일 시스템을 사용하는 가장 빠른 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 와 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

### 1단계 Amazon FSx for Lustre 설치 확인 ### 
```
aws cli... 
```

### 2단계: Amazon FSx CSI 드라이버 설치 ### 
Helm을 사용하여 EKS 클러스터에 FSx for Lustre CSI 드라이버를 배포한다. 
AWS FSx CSI 드라이버의 이미지는 각 리전별 AWS 전용 ECR 레포지토리에서 가져와야 한다.(image.repository)
```
kubectl create namespace fsx-csi-driver
helm repo add aws-fsx-csi-driver kubernetes-sigs.github.io
helm repo update

helm install fsx-csi-driver --namespace fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver \
--set image.repository=602401143452.dkr.ecr.ap-northeast-2.amazonaws.com, \
controller.serviceAccount.name=fsx-csi-driver-controller-sa, \
controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=<IAM_역할_ARN>
```

### 3단계: StorageClass 및 Persistent Volume Claim (PVC) 배포 ###
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

### 4단계: 애플리케이션 파드에서 사용 ###
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

## 레퍼런스 ##
* https://aws.amazon.com/ko/blogs/tech/lustre/

