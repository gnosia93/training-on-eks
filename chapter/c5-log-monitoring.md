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

#### 1. loki-ng 노드그룹 추가 ####
```
cat <<EOF > ng-loki.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: training-on-eks   # 기존 클러스터 이름
  region: ${AWS_REGION}   # 실제 사용 중인 리전

managedNodeGroups:
  - name: ng-loki
    instanceType: m7i.2xlarge
    desiredCapacity: 1
    minSize: 1
    maxSize: 1
    amiFamily: AmazonLinux2023
    privateNetworking: true           # 이 노드 그룹이 PRIVATE 서브넷만 사용하도록 지정합니다. 
    iam:
      withAddonPolicies:
        ebs: true         # EBS CSI 드라이버가 작동하기 위한 IAM 권한 부여
EOF

eksctl create nodegroup -f ng-loki.yaml
```
"ebs: true" 설정은 loki-ng 노드 그룹의 노드들에게 EBS 볼륨을 생성,삭제,연결(Attach),해제(Detach)할 수 있는 권한을 부여한다는 의미이다.  



#### 2. S3 버킷생성 ####
Loki를 배포하기 전에 두 개의 S3 버킷을 생성해야 한다. 첫 번째는 로그 데이터(Chunks)를 저장하기 위한 것이고, 두 번째는 알람 규칙(Alert Rules)을 저장하기 위한 것이다.
Loki는 로그 정보를 인덱스와 실제 데이터(Chunks)로 나누어 저장하는데, 로컬 디스크(EBS) 대신 S3를 주 저장소로 사용함으로써 비용을 절감하고 공간을 무제한으로 사용할 수 있게 된다. 
```
CHUNK_BUCKET=$(aws s3api create-bucket --bucket loki-aws-dev-chunks-${ACCOUNT_ID} --region ${AWS_REGION} \
  --create-bucket-configuration LocationConstraint=${AWS_REGION} --query "Location" --output text)
RULER_BUCKET=$(aws s3api create-bucket --bucket loki-aws-dev-ruler-${ACCOUNT_ID} --region ${AWS_REGION} \
  --create-bucket-configuration LocationConstraint=${AWS_REGION} --query "Location" --output text)

CHUNK_BUCKET=$(echo ${CHUNK_BUCKET} | cut -d'/' -f3 | cut -d'.' -f1)
RULER_BUCKET=$(echo ${RULER_BUCKET} | cut -d'/' -f3 | cut -d'.' -f1)
```

#### 3. IAM 역할 및 정책 ####
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
```

```
aws iam create-role --role-name LokiServiceAccountRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name LokiServiceAccountRole --policy-arn arn:aws:iam::<Account ID>:policy/LokiS3AccessPolicy
```

#### 4. Deploying the Helm chart ####

```
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
kubectl create namespace loki
```

#### 5. Loki Basic Authentication ####
Loki by default does not come with any authentication. Since we will be deploying Loki to AWS and exposing the gateway to the internet, we recommend adding at least basic authentication. In this guide we will give Loki a username and password
```
htpasswd -c .htpasswd <username>
kubectl create secret generic loki-basic-auth --from-file=.htpasswd -n loki

kubectl create secret generic canary-basic-auth \
  --from-literal=username=<USERNAME> \
  --from-literal=password=<PASSWORD> \
  -n loki
```

#### 6. Loki Helm chart configuration ####
```
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
       region: <S3 BUCKET REGION> # for example, eu-west-2  
       bucketnames: <CHUNK BUCKET NAME> # Your actual S3 bucket name, for example, loki-aws-dev-chunks
       s3forcepathstyle: false
   ingester:
       chunk_encoding: snappy
   pattern_ingester:
       enabled: true
   limits_config:
     allow_structured_metadata: true
     volume_enabled: true
     retention_period: 672h # 28 days retention
   compactor:
     retention_enabled: true 
     delete_request_store: s3
   ruler:
    enable_api: true
    storage:
      type: s3
      s3:
        region: <S3 BUCKET REGION> # for example, eu-west-2
        bucketnames: <RULER BUCKET NAME> # Your actual S3 bucket name, for example, loki-aws-dev-ruler
        s3forcepathstyle: false
      alertmanager_url: http://prom:9093 # The URL of the Alertmanager to send alerts (Prometheus, Mimir, etc.)

   querier:
      max_concurrent: 4

   storage:
      type: s3
      bucketNames:
        chunks: "<CHUNK BUCKET NAME>" # Your actual S3 bucket name (loki-aws-dev-chunks)
        ruler: "<RULER BUCKET NAME>" # Your actual S3 bucket name (loki-aws-dev-ruler)
        # admin: "<Insert s3 bucket name>" # Your actual S3 bucket name (loki-aws-dev-admin) - GEL customers only
      s3:
        region: <S3 BUCKET REGION> # eu-west-2
        #insecure: false
      # s3forcepathstyle: false

serviceAccount:
 create: true
 annotations:
   "eks.amazonaws.com/role-arn": "arn:aws:iam::<Account ID>:role/LokiServiceAccountRole" # The service role you created

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
 basicAuth: 
     enabled: true
     existingSecret: loki-basic-auth

# Since we are using basic auth, we need to pass the username and password to the canary
lokiCanary:
  extraArgs:
    - -pass=$(LOKI_PASS)
    - -user=$(LOKI_USER)
  extraEnv:
    - name: LOKI_PASS
      valueFrom:
        secretKeyRef:
          name: canary-basic-auth
          key: password
    - name: LOKI_USER
      valueFrom:
        secretKeyRef:
          name: canary-basic-auth
          key: username

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
```

* 로키 까나리 disable
```
lokiCanary:
  enabled: false
```

#### 7.Deploy Loki ####
Now that you have created the values.yaml file, you can deploy Loki using the Helm chart.
```
helm install --values values.yaml loki grafana/loki -n loki --create-namespace \
    --set loki.nodeSelector."alpha\.eksctl\.io/nodegroup-name"=loki-workers

kubectl get pods -n loki
```
It is important to create a namespace called loki as our trust policy is set to allow the IAM role to be used by the loki service account in the loki namespace. This is configurable but make sure to update your service account


### [Log Sender (Grafana Alloy) 설치](https://grafana.com/docs/alloy/latest/collect/logs-in-kubernetes/) ###
Alloy는 기본적으로 "어디서 읽고 어디로 보낼지"에 대한 Pipeline 설정이 필요하다. 노드 파일 시스템의 로그에 접근하기 위해 DaemonSet 모드로 실행해야 하며,  
Loki를 목적지를 설정하고 쿠버네티스 Pod를 탐색하도록 구성해야 한다. 

[alloy-values.yaml]
```
alloy:
  configMap:
    create: true
    # 여기에 제시하신 .alloy 설정 내용을 넣습니다.
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

      loki.source.file "pod_logs" {
        targets    = discovery.relabel.pod_logs.output
        forward_to = [loki.process.pod_logs.receiver]
      }

      loki.process "pod_logs" {
        stage.static_labels {
            values = {
              cluster = "training-on-eks",
            }
        }
        forward_to = [loki.write.grafana_loki.receiver]
      }

      // 필수: 로그를 실제로 보낼 Loki 주소 설정
      // url 은 프로토콜(http)과 API 경로(/loki/api/v1/push) 추가 필요. 
      loki.write "grafana_loki" {
        endpoint {
          url = "loki-gateway.loki.svc.cluster.local"
        }
      }

# 필수: 노드의 로그 파일(/var/log/pods)에 접근하기 위한 설정
controller:
  type: daemonset
  volumes:
    extra:
      - name: varlog
        hostPath:
          path: /var/log

extraVolumeMounts:
  - name: varlog
    mountPath: /var/log
    readOnly: true
```
Alloy 문법: discovery → source → process → write로 이어지는 파이프라인 흐름(receiver 및 output 참조)이 정확합니다

```
helm install alloy grafana/alloy --namespace alloy --create-namespace \
  -f alloy-values.yaml
```

Pod 의 로그가 제대로 수집되어 있는지 확인한다. 
```
# Alloy 팟 이름 확인
kubectl get pods -n alloy

# Alloy 로그 실시간 확인
kubectl logs -f -n alloy <alloy-pod-이름>
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



