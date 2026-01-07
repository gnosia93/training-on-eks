<< 그라파나 대시보드에서 데이터가 보이지 않는다. 디버깅 필요. >>

## grafana loki ## 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki.webp)

Grafana Loki는 Grafana Labs에서 개발한 오픈소스 로그 집계 시스템으로, 대규모 시스템의 로그를 효율적이고 저렴하게 저장하고 검색하기 위해 설계되었다. 흔히 "로그판 프로메테우스(Prometheus)"라고도 불리며, 현대적인 클라우드 환경(Kubernetes 등)에 최적화되어 있다. 분산 학습 환경에서 Grafana Loki를 사용할 때의 핵심 장점은 다음과 같다. 
* 실시간 통합 모니터링: 여러 노드에 흩어진 파드 로그를 하나의 타임라인으로 병합하여 실시간(Live)으로 확인하고, 키워드 필터링을 통해 대량의 로그 속에서도 Loss나 에러를 즉각 추적.
* 강력한 디버깅 지원: 에러 발생 시점 앞뒤의 로그를 보여주는 'Context 조회' 기능을 통해 분산 학습 중 발생하는 복잡한 데드락이나 통신 오류의 원인을 쉽고 빠르게 파악.
* 높은 효율성과 가성비: S3와 같은 객체 스토리지 연동으로 대용량 학습 로그를 저렴하게 보관하며, 인덱싱 최적화를 통해 전문 검색 엔진 대비 리소스를 적게 사용하면서도 현대적인 분석 환경을 제공.
  
### [Log Backend(Loki) 설치](https://grafana.com/docs/loki/latest/setup/install/helm/deployment-guides/aws/) ###
```
export CLUSTER_NAME="training-on-eks"
export AWS_REGION=$(aws ec2 describe-availability-zones --query "AvailabilityZones[0].RegionName" --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
export OIDC=$(aws eks describe-cluster --name training-on-eks --query "cluster.identity.oidc.issuer" --output text | cut -d '/' -f 5)
```

#### 1. S3 버킷생성 ####
Loki를 배포하기 전에 두 개의 S3 버킷을 생성해야 한다. 첫 번째는 로그 데이터(Chunks)를 저장하기 위한 것이고, 두 번째는 알람 규칙(Alert Rules)을 저장하기 위한 것이다.
Loki는 로그 정보를 인덱스와 실제 데이터(Chunks)로 나누어 저장하는데, 로컬 디스크(EBS) 대신 S3를 주 저장소로 사용함으로써 비용을 절감하고 공간을 무제한으로 사용할 수 있게 된다. 

기존 버킷이 있는 경우 삭제한다.
```
aws s3 rb s3://loki-aws-dev-chunks-${ACCOUNT_ID} --force 2>/dev/null || true
aws s3 rb s3://loki-aws-dev-ruler-${ACCOUNT_ID} --force 2>/dev/null || true
```
버킷을 생성한다.
```
CHUNK_BUCKET=$(aws s3api create-bucket --bucket loki-aws-dev-chunks-${ACCOUNT_ID} --region ${AWS_REGION} \
  --create-bucket-configuration LocationConstraint=${AWS_REGION} --query "Location" --output text)
RULER_BUCKET=$(aws s3api create-bucket --bucket loki-aws-dev-ruler-${ACCOUNT_ID} --region ${AWS_REGION} \
  --create-bucket-configuration LocationConstraint=${AWS_REGION} --query "Location" --output text)

export CHUNK_BUCKET=$(echo ${CHUNK_BUCKET} | cut -d'/' -f3 | cut -d'.' -f1)
export RULER_BUCKET=$(echo ${RULER_BUCKET} | cut -d'/' -f3 | cut -d'.' -f1)
```

#### 2. IAM 역할 및 정책 ####
```
aws iam detach-role-policy --role-name LokiServiceAccountRole --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/LokiS3AccessPolicy 2>/dev/null
aws iam delete-role --role-name LokiServiceAccountRole 2>/dev/null
aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/LokiS3AccessPolicy 2>/dev/null
```


```
cat <<EOF > loki-s3-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "LokiStorage",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::${CHUNK_BUCKET}",
                "arn:aws:s3:::${CHUNK_BUCKET}/*",
                "arn:aws:s3:::${RULER_BUCKET}",
                "arn:aws:s3:::${RULER_BUCKET}/*"
            ]
        }
    ]
}
EOF

aws iam create-policy --policy-name LokiS3AccessPolicy --policy-document file://loki-s3-policy.json
```

```
cat <<EOF > trust-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC}:sub": "system:serviceaccount:loki:loki",
                    "oidc.eks.${AWS_REGION}.amazonaws.com/id/${OIDC}:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
EOF

aws iam create-role --role-name LokiServiceAccountRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name LokiServiceAccountRole --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/LokiS3AccessPolicy
```

#### 3. loki 네임스페이스 생성 ####

```
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
kubectl create namespace loki
```

#### 4. Loki 인증 설정 #### 
```
sudo dnf install httpd-tools -y

htpasswd -c .htpasswd loki               # 패스워드는 loki 로 설정한다.
kubectl create secret generic loki-basic-auth --from-file=.htpasswd -n loki

#kubectl create secret generic canary-basic-auth \
#  --from-literal=username=loki \
#  --from-literal=password=loki \
#  -n loki
```

#### 5. Loki 헬름 차트 설정 ####
```
export MY_OFFICE_IP="122.36.213.114/32"

cat <<EOF > loki-values.yaml
loki:
   schemaConfig:
     configs:
       - from: "2024-04-01"
         store: tsdb
         object_store: s3
         schema: v13
         index:
           prefix: loki_index_
           period: 24h
   storage_config:
     aws:
       region: ${AWS_REGION}  
       bucketnames: ${CHUNK_BUCKET}
       s3forcepathstyle: false
   ingester:
       chunk_encoding: snappy
   pattern_ingester:
       enabled: true
   limits_config:
     allow_structured_metadata: true
     volume_enabled: true
     retention_period: 672h             # 28 days retention
   compactor:
     retention_enabled: true 
     delete_request_store: s3
   ruler:
    enable_api: true
    storage:
      type: s3
      s3:
        region: ${AWS_REGION}
        bucketnames: ${RULER_BUCKET}
        s3forcepathstyle: false
      alertmanager_url: http://prom:9093       # The URL of the Alertmanager to send alerts (Prometheus, Mimir, etc.)

   querier:
      max_concurrent: 4

   storage:
      type: s3
      bucketNames:
        chunks: "${CHUNK_BUCKET}" 
        ruler: "${RULER_BUCKET}" 
      s3:
        region: ${AWS_REGION}

serviceAccount:
 create: true
 annotations:
   "eks.amazonaws.com/role-arn": "arn:aws:iam::${ACCOUNT_ID}:role/LokiServiceAccountRole" 

deploymentMode: Distributed

ingester:
 replicas: 3
 zoneAwareReplication:
  enabled: false

querier:
 replicas: 3
 maxUnavailable: 2

queryFrontend:
 replicas: 2
 maxUnavailable: 1

queryScheduler:
 replicas: 2

distributor:
 replicas: 3
 maxUnavailable: 2
compactor:
 replicas: 1

indexGateway:
 replicas: 2
 maxUnavailable: 1

ruler:
 replicas: 1
 maxUnavailable: 1

# This exposes the Loki gateway so it can be written to and queried externaly
gateway:
 service:
   type: LoadBalancer
   loadBalancerSourceRanges:
      - ${MY_OFFICE_IP}                       # 내 아이피 로만 접근 가능
 basicAuth:                                   # gateway 접근시 4. Loki 인증 설정에서 작성한 id/password 로 기본 인증(Basic Auth) 인증해야 한다. 
     enabled: true
     existingSecret: loki-basic-auth

test:
  enabled: false

lokiCanary:
  enabled: false

# Enable minio for storage
minio:
 enabled: false

backend:
 replicas: 0
read:
 replicas: 0
write:
 replicas: 0

singleBinary:
 replicas: 0
EOF
```

#### 6. Loki 배포하기 ####
```
helm install loki grafana/loki -n loki \
    --values loki-values.yaml 
```
[결과]
```
NAME: loki
LAST DEPLOYED: Sat Dec 27 15:06:47 2025
NAMESPACE: loki
STATUS: deployed
REVISION: 1
DESCRIPTION: Install complete
TEST SUITE: None
NOTES:
***********************************************************************
 Welcome to Grafana Loki
 Chart version: 6.49.0
 Chart Name: loki
 Loki version: 3.6.3
***********************************************************************

** Please be patient while the chart is being deployed **

Tip:

  Watch the deployment status using the command: kubectl get pods -w --namespace loki

If pods are taking too long to schedule make sure pod affinity can be fulfilled in the current cluster.

***********************************************************************
Installed components:
***********************************************************************
* gateway
* compactor
* index gateway
* query scheduler
* ruler
* distributor
* ingester
* querier
* query frontend


***********************************************************************
Sending logs to Loki
***********************************************************************

Loki has been configured with a gateway (nginx) to support reads and writes from a single component.

You can send logs from inside the cluster using the cluster DNS:

http://loki-gateway.loki.svc.cluster.local/loki/api/v1/push

You can test to send data from outside the cluster by port-forwarding the gateway to your local machine:

  kubectl port-forward --namespace loki svc/loki-gateway 3100:80 &

And then using http://127.0.0.1:3100/loki/api/v1/push URL as shown below:


curl -H "Content-Type: application/json" -XPOST -s "http://127.0.0.1:3100/loki/api/v1/push"  \
--data-raw "{\"streams\": [{\"stream\": {\"job\": \"test\"}, \"values\": [[\"$(date +%s)000000000\", \"fizzbuzz\"]]}]}" \
-H X-Scope-OrgId:foo


Then verify that Loki did receive the data using the following command:


curl "http://127.0.0.1:3100/loki/api/v1/query_range" --data-urlencode 'query={job="test"}' -H X-Scope-OrgId:foo | jq .data.result


***********************************************************************
Connecting Grafana to Loki
***********************************************************************

If Grafana operates within the cluster, you'll set up a new Loki datasource by utilizing the following URL:

http://loki-gateway.loki.svc.cluster.local/

***********************************************************************
Multi-tenancy
***********************************************************************

Loki is configured with auth enabled (multi-tenancy) and expects tenant headers (`X-Scope-OrgID`) to be set for all API calls.

You must configure Grafana's Loki datasource using the `HTTP Headers` section with the `X-Scope-OrgID` to target a specific tenant.
For each tenant, you can create a different datasource.

The agent of your choice must also be configured to propagate this header.
For example, when using Promtail you can use the `tenant` stage. https://grafana.com/docs/loki/latest/send-data/promtail/stages/tenant/

When not provided with the `X-Scope-OrgID` while auth is enabled, Loki will reject reads and writes with a 404 status code `no org id`.

You can also use a reverse proxy, to automatically add the `X-Scope-OrgID` header as suggested by https://grafana.com/docs/loki/latest/operations/authentication/

For more information, read our documentation about multi-tenancy: https://grafana.com/docs/loki/latest/operations/multi-tenancy/

> When using curl you can pass `X-Scope-OrgId` header using `-H X-Scope-OrgId:foo` option, where foo can be replaced with the tenant of your choice.
```

```
kubectl get pods -n loki
```
[결과]
```
NAME                                    READY   STATUS    RESTARTS   AGE
loki-chunks-cache-0                     2/2     Running   0          54s
loki-compactor-0                        1/1     Running   0          54s
loki-distributor-f68976f8f-g42jp        1/1     Running   0          54s
loki-distributor-f68976f8f-h5dkb        1/1     Running   0          53s
loki-distributor-f68976f8f-xmt6h        1/1     Running   0          53s
loki-gateway-6f6d8c796f-t9qzn           1/1     Running   0          54s
loki-index-gateway-0                    1/1     Running   0          54s
loki-index-gateway-1                    1/1     Running   0          25s
loki-ingester-0                         1/1     Running   0          54s
loki-ingester-1                         1/1     Running   0          53s
loki-ingester-2                         1/1     Running   0          53s
loki-querier-5867f585f5-2htk4           1/1     Running   0          53s
loki-querier-5867f585f5-kzb4q           1/1     Running   0          54s
loki-querier-5867f585f5-z6npv           1/1     Running   0          53s
loki-query-frontend-646bd6f4df-jjwh7    1/1     Running   0          54s
loki-query-frontend-646bd6f4df-ldq45    1/1     Running   0          53s
loki-query-scheduler-7b75c5fdc9-f95gk   1/1     Running   0          54s
loki-query-scheduler-7b75c5fdc9-qnqnk   1/1     Running   0          53s
loki-results-cache-0                    2/2     Running   0          54s
loki-ruler-0                            1/1     Running   0          54s 
```


### [Log Sender (Grafana Alloy) 설치](https://grafana.com/docs/alloy/latest/collect/logs-in-kubernetes/) ###
Alloy는 fluentBit 와 같은 로그 수집기로 "데이터를 어디서 읽어서 어디로 보낼지"에 대한 파이프라인을 설정 해야 한다. 노드의 로그에 접근하기 위해 DaemonSet 모드로 실행해야 하며, Loki를 타켓으로 설정하고 쿠버네티스 Pod를 탐색하도록 구성해야 한다. 
```
cat <<'EOF' > alloy-values.yaml
alloy:
  configMap:
    create: true
    # 여기에 제시하신 .alloy 설정 내용을 넣는다.
    content: |
      discovery.kubernetes "pod" {
        role = "pod"
        selectors {
          role = "pod"
          field = "spec.nodeName=" + coalesce(sys.env("HOSTNAME"), constants.hostname)
        }
      }

      discovery.relabel "pod_logs" {
        targets = discovery.kubernetes.pod.targets

        // Label creation - "namespace" field from "__meta_kubernetes_namespace"
        rule {
          source_labels = ["__meta_kubernetes_namespace"]
          action = "replace"
          target_label = "namespace"
        }

        // Label creation - "pod" field from "__meta_kubernetes_pod_name"
        rule {
          source_labels = ["__meta_kubernetes_pod_name"]
          action = "replace"
          target_label = "pod"
        }

        // Label creation - "container" field from "__meta_kubernetes_pod_container_name"
        rule {
          source_labels = ["__meta_kubernetes_pod_container_name"]
          action = "replace"
          target_label = "container"
        }

        // Label creation -  "app" field from "__meta_kubernetes_pod_label_app_kubernetes_io_name"
        rule {
          source_labels = ["__meta_kubernetes_pod_label_app_kubernetes_io_name"]
          action = "replace"
          target_label = "app"
        }

        // Label creation -  "job" field from "__meta_kubernetes_namespace" and "__meta_kubernetes_pod_container_name"
        // Concatenate values __meta_kubernetes_namespace/__meta_kubernetes_pod_container_name
        rule {
          source_labels = ["__meta_kubernetes_namespace", "__meta_kubernetes_pod_container_name"]
          action = "replace"
          target_label = "job"
          separator = "/"
          replacement = "$1/$2"
        }

        // Label creation - "__path__" field from "__meta_kubernetes_pod_uid" and "__meta_kubernetes_pod_container_name"
        // Concatenate values __meta_kubernetes_pod_uid/__meta_kubernetes_pod_container_name.log
        rule {
          source_labels = ["__meta_kubernetes_pod_uid", "__meta_kubernetes_pod_container_name"]
          action = "replace"
          target_label = "__path__"
          separator = "/"
          replacement = "/var/log/pods/*$1/*.log"
        }

        // Label creation -  "container_runtime" field from "__meta_kubernetes_pod_container_id"
        rule {
          source_labels = ["__meta_kubernetes_pod_container_id"]
          action = "replace"
          target_label = "container_runtime"
          regex = "^(\\S+):\\/\\/.+$"
          replacement = "$1"
        }
      }

      //loki.source.file "pod_logs" {
      //  targets    = discovery.relabel.pod_logs.output
      //  forward_to = [loki.process.pod_logs.receiver]
      //}

      //loki.process "pod_logs" {
      //  stage.static_labels {
      //     values = {
      //        cluster = "training-on-eks",
      //      }
      //  }
      //  forward_to = [loki.write.grafana_loki.receiver]
      //}

      // 필수: 로그를 실제로 보낼 Loki 주소 설정
      // url 은 프로토콜(http)과 API 경로(/loki/api/v1/push) 추가 필요. 
      //loki.write "grafana_loki" {
      //  endpoint {
      //    url = "loki-gateway.loki.svc.cluster.local"
      //     url = "http://loki-gateway.loki.svc.cluster.local//loki/api/v1/push"
      //  }
      //}

      // 2. 중요: 와일드카드 경로를 실제 파일 목록으로 확장
      local.file_match "pod_logs" {
        path_targets = discovery.relabel.pod_logs.output
      }

      // 3. 확장된 파일 타겟(targets)을 전달받아 읽기
      loki.source.file "pod_logs" {
        targets    = local.file_match.pod_logs.targets // 수정됨: file_match의 output을 사용
        forward_to = [loki.process.pod_logs.receiver]
      }

      // 4. 로그 프로세싱 (정적 레이블 추가)
      loki.process "pod_logs" {
        stage.static_labels {
            values = {
              cluster = "training-on-eks",
            }
        }
        forward_to = [loki.write.grafana_loki.receiver]
      }

      // 5. Loki로 전송
      loki.write "grafana_loki" {
        endpoint {
           // URL 수정: 중복 슬래시 제거
           url = "http://loki-gateway.loki.svc.cluster.local/loki/api/v1/push"

           // 401 에러 해결을 위한 인증 설정
           basic_auth {
             username = "loki"  // 설정하신 사용자 이름
             password = "loki"  // 설정하신 비밀번호
           }

           // 멀티테넌시를 사용하는 경우 헤더 추가
           headers = {
             "X-Scope-OrgId" = "foo",
           }
        }
      } 

  # 필수: 노드의 로그 파일(/var/log/pods)에 접근하기 위한 설정
  controller:
    type: daemonset

  securityContext:
    privileged: true # 권한 문제 해결을 위해 필수일 수 있습니다.

  # 핵심: 호스트의 로그 경로를 컨테이너 내부로 연결
  extraVolumeMounts:
    - name: varlog
      mountPath: /var/log
      readOnly: true
    - name: varlibdockercontainers
      mountPath: /var/lib/docker/containers
      readOnly: true

  extraVolumes:
    - name: varlog
      hostPath:
        path: /var/log
    - name: varlibdockercontainers
      hostPath:
        path: /var/lib/docker/containers

# RBAC 설정 (파드 목록을 조회하기 위해 필요)
rbac:
  create: true
EOF
```
alloy 설정은 discovery → source → process → write로 이어지는 파이프라인 흐름(receiver 및 output 참조)이다.  
alloy 를 설치하고 pod 를 조회한다.
```
helm install alloy grafana/alloy --namespace alloy --create-namespace -f alloy-values.yaml
kubectl get pods -n alloy
```
[결과]
```
NAME          READY   STATUS    RESTARTS   AGE
alloy-d6wwh   2/2     Running   0          13s
alloy-l6djd   1/2     Running   0          13s
alloy-rldlj   2/2     Running   0          13s
alloy-wp2hk   2/2     Running   0          13s
```

pod 로그가 제대로 수집되어 있는지 확인한다. 
```
kubectl logs -n alloy -l app.kubernetes.io/name=alloy | grep -iE "error|failed|401|403|Unauthorized"d"
```

[결과]
```
ts=2026-01-07T06:08:39.678943257Z level=info msg="failed to register collector with remote server" service=remotecfg id=bc50fae7-dc7a-4463-a49c-e14ef6999a55 name="" err="noop client"
ts=2026-01-07T06:08:39.627810764Z level=info msg="failed to register collector with remote server" service=remotecfg id=edb8a72a-22a6-491b-8cea-539cea380329 name="" err="noop client"
ts=2026-01-07T06:08:39.678890389Z level=info msg="failed to register collector with remote server" service=remotecfg id=6b730f9f-e41f-4271-b5e5-852d9547d8d8 name="" err="noop client"
ts=2026-01-07T06:08:39.69334033Z level=info msg="finished node evaluation" controller_path=/ controller_id="" trace_id=911e2b3cb0296dbd1e88dd0fc0092eff node_id=cluster duration=4.674µs
ts=2026-01-07T06:08:39.693635952Z level=info msg="failed to register collector with remote server" service=remotecfg id=5d7dbf41-4167-4d96-bd45-579ac78bbfe5 name="" err="noop client"
```
로그 출력 결과를 보니 Loki 전송과 관련된 에러(401, failed to send batch 등)는 전혀 보이지 않는다.
출력된 failed to register collector... 메시지는 Grafana Cloud의 원격 관리 기능을 쓰지 않을 때 나타나는 정보성 로그이다.

Alloy 파드 중 하나에 들어가서 실제 로그 파일 리스트가 출력되는지 확인한다.
```
kubectl get pods -n alloy
```
[결과]
```
NAME          READY   STATUS    RESTARTS   AGE
alloy-d6wwh   2/2     Running   0          5m21s
alloy-l6djd   2/2     Running   0          5m21s
alloy-rldlj   2/2     Running   0          5m21s
alloy-wp2hk   2/2     Running   0          5m21s
```
```
kubectl exec -it alloy-d6wwh -n alloy -- ls -R /var/log/pods
```

### [Grafana Dashboard 설정]() ###

#### 1. 데이터 소스 설정 단계 ####
* Grafana 접속: 웹 브라우저에서 Grafana UI에 로그인합니다.
* Connections 메뉴 이동: 왼쪽 사이드바에서 Connections > Data sources를 클릭합니다.
* 데이터 소스 추가: Add data source 버튼을 누르고 검색창에 Loki를 입력하여 선택합니다.
* 상세 정보 입력 (중요):
* Name: Loki (원하는 이름)
* HTTP URL: Alloy 설정에 사용했던 주소의 앞부분만 입력합니다.
* 입력할 값: http://loki-gateway.loki.svc.cluster.local (주의: /loki/api/v1/push는 빼고 도메인만 적습니다.)
* 저장 및 테스트: 하단의 Save & test 버튼을 누릅니다.
* Data source successfully connected.라는 초록색 메시지가 뜨면 성공입니다.

#### 2. 로그 데이터 조회 방법 (Explore) ####
* 설정이 끝났다면 실제로 로그가 나오는지 확인합니다.
* 왼쪽 메뉴에서 Explore (나침반 모양 아이콘)를 클릭합니다.
* 상단 드롭다운에서 방금 추가한 Loki 데이터 소스를 선택합니다.
* Label browser 버튼을 누르거나 쿼리창에 다음을 입력합니다.
* logql
{cluster="training-on-eks"}.

*Run query를 클릭하여 아래쪽에 실시간 로그가 올라오는지 확인합니다.


## 레퍼런스 ##

* [Loki Architecture: A Log Aggregation Journey with Grafana](https://sujayks007.medium.com/loki-architecture-a-log-aggregation-journey-with-grafana-bde6d9df6a04)



