## grafana loki ## 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki.webp)

Grafana Loki는 Grafana Labs에서 개발한 오픈소스 로그 집계 시스템으로, 대규모 시스템의 로그를 효율적이고 저렴하게 저장하고 검색하기 위해 설계되었습니다. 흔히 "로그판 프로메테우스(Prometheus)"라고도 불리며, 현대적인 클라우드 환경(Kubernetes 등)에 최적화되어 있다.
분산 학습 환경에서 Grafana Loki를 사용할 때의 핵심 장점은 다음과 같다. 
* 실시간 통합 모니터링: 여러 노드에 흩어진 파드 로그를 하나의 타임라인으로 병합하여 실시간(Live)으로 확인하고, 키워드 필터링을 통해 대량의 로그 속에서도 Loss나 에러를 즉각 추적.
* 강력한 디버깅 지원: 에러 발생 시점 앞뒤의 로그를 보여주는 'Context 조회' 기능을 통해 분산 학습 중 발생하는 복잡한 데드락이나 통신 오류의 원인을 쉽고 빠르게 파악.
* 높은 효율성과 가성비: S3와 같은 객체 스토리지 연동으로 대용량 학습 로그를 저렴하게 보관하며, 인덱싱 최적화를 통해 전문 검색 엔진 대비 리소스를 적게 사용하면서도 현대적인 분석 환경을 제공.

### 1. 설치하기 ###

#### 단계 1: S3 버킷 및 IAM 설정 (Terraform 예시) ####
Loki가 로그를 저장할 S3 버킷을 생성하고, EKS 노드가 이 버킷에 쓰기 권한을 가질 수 있도록 태그를 추가하거나 IAM 정책을 연결합니다.

```
# 1. S3 버킷 생성
resource "aws_s3_bucket" "loki_storage" {
  bucket = "dh-eks-loki-storage" # 고유한 이름으로 수정
}

# 2. IAM 정책 (이 정책을 노드 그룹의 IAM 역할에 연결)
resource "aws_iam_policy" "loki_s3_policy" {
  name        = "EKS-Loki-S3-Access"
  description = "Allow Loki to access S3 for log storage"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action   = ["s3:ListBucket", "s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
        Effect   = "Allow"
        Resource = [
          "${aws_s3_bucket.loki_storage.arn}",
          "${aws_s3_bucket.loki_storage.arn}/*"
        ]
      }
    ]
  })
}
```

#### 단계 2: Helm 저장소 추가 ####
```
helm repo add grafana grafana.github.io
helm repo update
```

#### 단계 3: Loki 설치 (S3 전용 설정) ####
분산 학습 로그는 양이 많으므로 S3 저장소 설정이 필수입니다. loki-values.yaml 파일을 만듭니다.
```
loki:
  auth_enabled: false
  commonConfig:
    replication_factor: 1
  
  storage:
    type: 's3'
    s3:
      region: ap-northeast-2
      bucketnames: dh-eks-loki-storage # 위에서 만든 S3 이름
  
  schemaConfig:
    configs:
      - from: "2024-01-01"
        index:
          period: 24h
          prefix: index_
        object_store: s3
        schema: v13
        store: tsdb # 2025년 표준 인덱스 형식

# 학습 로그의 원활한 처리를 위해 단일 바이너리 모드로 설치
deploymentMode: SingleBinary 

# (옵션) 로그 보관 주기 설정 (예: 30일)
# limits_config:
#   retention_period: 720h
```

```
helm install loki grafana/loki -f loki-values.yaml -n monitoring --create-namespace
```

#### 단계 4: Promtail 설치 (로그 수집기) ####
각 노드에서 분산 학습 파드의 Raw Text 로그를 긁어 Loki로 쏘아주는 역할입니다.
```
helm install promtail grafana/promtail \
  --set config.lokiAddress=http://loki:3100/loki/api/v1/push \
  -n monitoring
```

#### 단계 5: Grafana 연동 ####
* Connections -> Data Sources -> Add Loki
URL에 http://loki.monitoring.svc.cluster.local:3100 입력 후 Save & Test

* 분산 학습 로그 실시간 검색 방법
설치가 완료되면 Grafana 왼쪽 메뉴의 Explore에서 다음처럼 검색합니다.
   * 파드별 로그 보기:
   {pod="training-rank-0"} 선택 후 Run Query를 누르면 해당 파드가 내뱉는 모든 Raw Text가 보입니다.
   * 텍스트 안에서 검색:
   {pod=~"training-rank-.*"} |= "Loss"   
   rank-0, 1, 2... 모든 파드의 로그 중 "Loss"라는 단어가 들어간 줄만 실시간으로 필터링합니다.
   * 실시간 스트리밍:
   상단의 Live 버튼을 누르면 파드가 로그를 내뱉는 족족 화면에 흐르듯 나타납니다.


## 레퍼런스 ##
* [Loki Architecture: A Log Aggregation Journey with Grafana](https://sujayks007.medium.com/loki-architecture-a-log-aggregation-journey-with-grafana-bde6d9df6a04)







