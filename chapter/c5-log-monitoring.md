## grafana loki ## 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki.webp)

분산 학습 환경에서 Grafana Loki를 사용할 때의 핵심 장점음 다음과 같다. 
* 실시간 통합 모니터링: 여러 노드에 흩어진 파드 로그를 하나의 타임라인으로 병합하여 실시간(Live)으로 확인하고, 키워드 필터링을 통해 대량의 로그 속에서도 Loss나 에러를 즉각 추적.
* 강력한 디버깅 지원: 에러 발생 시점 앞뒤의 로그를 보여주는 'Context 조회' 기능을 통해 분산 학습 중 발생하는 복잡한 데드락이나 통신 오류의 원인을 쉽고 빠르게 파악.
* 높은 효율성과 가성비: S3와 같은 객체 스토리지 연동으로 대용량 학습 로그를 저렴하게 보관하며, 인덱싱 최적화를 통해 전문 검색 엔진 대비 리소스를 적게 사용하면서도 현대적인 분석 환경을 제공.


### 1. 장점 ###
#### 1-1. 실시간 Raw 텍스트 조회 및 검색 (핵심 기능) ####
* 실시간 스트리밍: 분산 학습 파드가 내뱉는 표준 출력(stdout/stderr)을 Grafana 화면에서 Live 모드로 실시간 모니터링할 수 있습니다.
* 텍스트 검색: 특정 파드 내 로그에서 Error, Loss, Epoch 같은 특정 키워드를 구글 검색하듯 필터링할 수 있습니다.
* 쿼리 예시: {app="dist-training"} |= "Loss" (Loss라는 단어가 포함된 로그만 출력)
* 파드별 개별 조회: 각 파드(rank-0, rank-1 등)마다 레이블이 자동으로 붙기 때문에, 특정 파드의 로그만 따로 떼어서 보거나 여러 파드의 로그를 시계열로 합쳐서 볼 수 있습니다.

#### 1-2. 분산 학습 시 유용한 기능 ####
* 로그 병합 (Log Aggregation): 여러 노드에 흩어져 있는 파드들의 로그를 시간순으로 정렬하여 하나의 타임라인으로 보여줍니다. 분산 학습 중 특정 Rank에서 발생한 문제가 전체 학습에 어떤 영향을 주었는지 파악하기 쉽습니다.
* Context 조회: 검색 결과에서 특정 에러 로그를 클릭하면, 그 에러 발생 직전과 직후의 로그(Context)를 바로 보여줍니다. 분산 학습 중 데드락이나 통신 오류 원인을 찾을 때 매우 강력합니다.

#### 1-3. 설정 시 주의사항 (학습 로그 최적화) ####
분산 학습 로그는 양이 많고 한 줄이 길 수 있습니다. 이를 위해 다음 설정을 권장합니다.
* 수집기(Promtail/Fluent Bit) 설정:
  * 파드의 이름을 레이블로 지정하여 pod_name으로 필터링할 수 있게 합니다.
  * 멀티라인 로그(예: 파이썬 Traceback 에러)가 한 줄로 깨지지 않도록 Multi-line stage 설정을 추가합니다.

* Loki 용량 관리:
  * 학습 로그는 한 번에 수십 GB가 쌓일 수 있으므로, 앞서 말씀드린 S3 연동은 필수입니다.

#### 1-4. Grafana에서의 실제 사용 모습  ####
- Grafana의 Explore 메뉴로 이동합니다.
- Data Source를 Loki로 선택합니다.
- Label browser에서 pod 또는 app 이름을 선택하여 학습 파드를 지정합니다.
- 상단의 Live 버튼을 누르면 학습 중인 로그가 실시간으로 올라오는 것을 볼 수 있습니다.
- 검색창에 필터링할 단어를 입력하여 실시간으로 Raw 텍스트를 분석합니다.

결론적으로, 분산 학습 파드들이 내뱉는 raw 텍스트를 중앙 집중식으로 저장하고, 웹 UI에서 실시간으로 검색하거나 특정 시점의 로그를 복기하는 데 Loki는 가장 현대적이고 가성비 좋은 선택입니다. Grafana 공식 가이드 - LogQL 검색법을 참고하시면 텍스트 검색 쿼리 작성에 도움이 됩니다.


### 2. 전체 데이터 흐름 (3단계) ###
#### 2-1. 수집 (Collector): ####
* 도구: Fluentd, Fluent Bit, 또는 Promtail (보통 가벼운 Fluent Bit나 Loki 전용인 Promtail을 가장 많이 씁니다).
* 역할: 각 노드(EC2)에서 실행 중인 컨테이너들의 로그 파일(/var/log/pods/...)을 실시간으로 읽어서 Loki 서버로 전송(Push)합니다.

#### 2-2. 저장 및 인덱싱 (Loki Server): ####
* 도구: Loki
* 역할: 수집기에서 받은 로그를 전달받아 두 가지로 나누어 처리합니다.
  * Chunk (본문): 실제 로그 내용입니다. 이걸 압축해서 S3에 파일 형태로 저장합니다.
  * Index (이정표): 로그를 빨리 찾기 위한 주소록입니다. 2025년 현재는 이 인덱스 정보까지도 S3에 같이 저장하는 방식(TSDB 구조)이 표준입니다.

#### 2-3. 조회 (Grafana): ####
* 도구: Grafana
* 역할: 사용자가 대시보드에서 로그를 검색하면, Grafana가 Loki에게 요청을 보냅니다. Loki는 S3에서 필요한 로그 조각(Chunk)을 가져와서 사용자에게 보여줍니다.

### 3. 설치가이드 ###

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

#### 단계 5: Grafana 설치 및 로그 확인 ####
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

#### 2025년 운영 팁 ####
* 분산 학습용 레이블: 학습 스크립트 작성 시 파드 이름에 rank-ID를 포함시키면 Loki에서 개별 노드의 상태를 추적하기 매우 쉬워집니다.
* S3 비용: S3에 로그가 무한정 쌓이지 않도록 S3 Lifecycle Policy를 설정하여 오래된 로그는 자동 삭제되도록 하세요.

#### Loki에서 왜 추적이 쉬워지나요? ####
Loki는 로그를 저장할 때 파드 이름(pod)을 자동으로 인덱스(색인)로 사용합니다. 이렇게 이름에 Rank가 들어 있으면 Grafana 검색창에서 다음과 같은 작업을 할 수 있습니다.
* 특정 Rank만 보기: "2번 워커가 자꾸 죽는데, 2번 로그만 보여줘"
쿼리: {pod="training-job-worker-2"}
* 전체 워커 로그 합쳐서 시간순으로 보기: "모든 워커의 로그를 한 화면에 섞어서 시간순으로 정렬해줘"
쿼리: {pod=~"training-job-worker-.*"}
* 특정 에러 비교: "0번(마스터)은 멀쩡한데 1번 워커에만 Connection Timeout이 뜨는지 확인해줘"

## 레퍼런스 ##
* [Loki Architecture: A Log Aggregation Journey with Grafana](https://sujayks007.medium.com/loki-architecture-a-log-aggregation-journey-with-grafana-bde6d9df6a04)







