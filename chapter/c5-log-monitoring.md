<<>>
<< 제대로 설치가 되지 않는다.. 디버깅 필요. >>
<<>>

#### S3 버킷생성 ####
Before deploying Loki, you need to create two S3 buckets; one to store logs (chunks), the second to store alert rules. You can create the bucket using the AWS Management Console or the AWS CLI. The bucket name must be globally unique.














---
## grafana loki ## 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki.webp)

Grafana Loki는 Grafana Labs에서 개발한 오픈소스 로그 집계 시스템으로, 대규모 시스템의 로그를 효율적이고 저렴하게 저장하고 검색하기 위해 설계되었습니다. 흔히 "로그판 프로메테우스(Prometheus)"라고도 불리며, 현대적인 클라우드 환경(Kubernetes 등)에 최적화되어 있다.
분산 학습 환경에서 Grafana Loki를 사용할 때의 핵심 장점은 다음과 같다. 
* 실시간 통합 모니터링: 여러 노드에 흩어진 파드 로그를 하나의 타임라인으로 병합하여 실시간(Live)으로 확인하고, 키워드 필터링을 통해 대량의 로그 속에서도 Loss나 에러를 즉각 추적.
* 강력한 디버깅 지원: 에러 발생 시점 앞뒤의 로그를 보여주는 'Context 조회' 기능을 통해 분산 학습 중 발생하는 복잡한 데드락이나 통신 오류의 원인을 쉽고 빠르게 파악.
* 높은 효율성과 가성비: S3와 같은 객체 스토리지 연동으로 대용량 학습 로그를 저렴하게 보관하며, 인덱싱 최적화를 통해 전문 검색 엔진 대비 리소스를 적게 사용하면서도 현대적인 분석 환경을 제공.

### 1. 설치하기 ###

#### S3 버킷 생성 ####
Loki가 로그를 저장할 S3 버킷을 생성하고, EKS 노드가 이 버킷에 쓰기 권한을 가질 수 있도록 태그를 추가하거나 IAM 정책을 연결합니다.

```
# S3 버킷 생성
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
UUID=$(cat /proc/sys/kernel/random/uuid)
REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
BUCKET_NAME="training-on-eks-${ACCOUNT_ID}-`date +%Y-%m-%d`"

aws s3api create-bucket \
    --bucket ${BUCKET_NAME} \
    --create-bucket-configuration LocationConstraint=${REGION}

aws s3api put-public-access-block \
    --bucket ${BUCKET_NAME} \
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Policy 파일 생성
cat <<EOF > loki-s3-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::dh-eks-loki-storage",
                "arn:aws:s3:::dh-eks-loki-storage/*"
            ]
        }
    ]
}
EOF

# IAM 정책 생성
aws iam create-policy \
    --policy-name EKSLokiS3AccessPolicy \
    --description "Allow Loki to access S3 for log storage" \
    --policy-document file://loki-s3-policy.json
```

#### Helm 으로 Loki 설치 ####
```
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

cat <<EOF > loki-values.yaml
loki:
  auth_enabled: false
  commonConfig:
    replication_factor: 1
  
  storage:
    type: 's3'
    s3:
      region: ${REGION}
      # 기존 bucket 또는 bucketnames 대신 아래 형식을 사용해야 합니다.
    bucketNames:
      chunks: ${BUCKET_NAME}
      ruler: ${BUCKET_NAME}
      admin: ${BUCKET_NAME}
    s3ForcePathStyle: true # AWS S3 표준 연결을 위해 권장

  schemaConfig:
    configs:
      - from: "2025-12-19"
        index:
          period: 24h
          prefix: index_
        object_store: s3
        schema: v13
        store: tsdb         # 2025년 표준 인덱스 형식

# 학습 로그의 원활한 처리를 위해 단일 바이너리 모드로 설치
deploymentMode: SingleBinary 

# (옵션) 로그 보관 주기 설정 (예: 30일)
# limits_config:
#   retention_period: 720h
EOF

helm install loki grafana/loki -f loki-values.yaml -n loki --create-namespace
```

#### Promtail 설치 (로그 수집기) ####
각 노드에서 분산 학습 파드의 Raw Text 로그를 긁어 Loki로 쏘아주는 역할입니다.
```
helm install promtail grafana/promtail \
  --set config.lokiAddress=http://loki:3100/loki/api/v1/push \
  -n loki
```
* 오류 발생시
```
kubectl patch storageclass gp2 -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```


## 레퍼런스 ##

* [Loki Architecture: A Log Aggregation Journey with Grafana](https://sujayks007.medium.com/loki-architecture-a-log-aggregation-journey-with-grafana-bde6d9df6a04)
* https://grafana.com/docs/loki/latest/setup/install/helm/deployment-guides/aws/



