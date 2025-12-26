## grafana loki ## 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki.webp)

Grafana Loki는 Grafana Labs에서 개발한 오픈소스 로그 집계 시스템으로, 대규모 시스템의 로그를 효율적이고 저렴하게 저장하고 검색하기 위해 설계되었다. 흔히 "로그판 프로메테우스(Prometheus)"라고도 불리며, 현대적인 클라우드 환경(Kubernetes 등)에 최적화되어 있다. 분산 학습 환경에서 Grafana Loki를 사용할 때의 핵심 장점은 다음과 같다. 
* 실시간 통합 모니터링: 여러 노드에 흩어진 파드 로그를 하나의 타임라인으로 병합하여 실시간(Live)으로 확인하고, 키워드 필터링을 통해 대량의 로그 속에서도 Loss나 에러를 즉각 추적.
* 강력한 디버깅 지원: 에러 발생 시점 앞뒤의 로그를 보여주는 'Context 조회' 기능을 통해 분산 학습 중 발생하는 복잡한 데드락이나 통신 오류의 원인을 쉽고 빠르게 파악.
* 높은 효율성과 가성비: S3와 같은 객체 스토리지 연동으로 대용량 학습 로그를 저렴하게 보관하며, 인덱싱 최적화를 통해 전문 검색 엔진 대비 리소스를 적게 사용하면서도 현대적인 분석 환경을 제공.
  
### [Log Collection Backend(Loki) 설치](https://grafana.com/docs/loki/latest/setup/install/helm/deployment-guides/aws/) ###
#### 1. cluster.yaml 변경 ####
```
addons:
  - name: aws-ebs-csi-driver             <----- csi 애드온 설치

managedNodeGroups:
  - name: loki-workers
    instanceType: m7i.2xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 3
    amiFamily: AmazonLinux2023
    iam:                                  <---- iam 생성.
      withAddonPolicies:
        ebs: true
```

#### 2. S3 버킷생성 ####
Before deploying Loki, you need to create two S3 buckets; one to store logs (chunks), the second to store alert rules. You can create the bucket using the AWS Management Console or the AWS CLI. The bucket name must be globally unique.
```
aws s3api create-bucket --bucket  <YOUR CHUNK BUCKET NAME e.g. `loki-aws-dev-chunks`> --region <S3 region your account is on, e.g. `eu-west-2`> --create-bucket-configuration LocationConstraint=<S3 region your account is on, e.g. `eu-west-2`> \

aws s3api create-bucket --bucket  <YOUR RULER BUCKET NAME e.g. `loki-aws-dev-ruler`> --region <S3 REGION your account is on, e.g. `eu-west-2`> --create-bucket-configuration LocationConstraint=<S3 REGION your account is on, e.g. `eu-west-2`>
```

#### 3. Defining IAM roles and policies ####
The recommended method for connecting Loki to AWS S3 is to use an IAM role. 
[loki-s3-policy.json]
```
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
                "arn:aws:s3:::< CHUNK BUCKET NAME >",
                "arn:aws:s3:::< CHUNK BUCKET NAME >/*",
                "arn:aws:s3:::< RULER BUCKET NAME >",
                "arn:aws:s3:::< RULER BUCKET NAME >/*"
            ]
        }
    ]
}
```

```
aws iam create-policy --policy-name LokiS3AccessPolicy --policy-document file://loki-s3-policy.json
```

[trust-policy.json]
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::< ACCOUNT ID >:oidc-provider/oidc.eks.<INSERT REGION>.amazonaws.com/id/< OIDC ID >"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "oidc.eks.<INSERT REGION>.amazonaws.com/id/< OIDC ID >:sub": "system:serviceaccount:loki:loki",
                    "oidc.eks.<INSERT REGION>.amazonaws.com/id/< OIDC ID >:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
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
helm install --values values.yaml loki grafana/loki -n loki --create-namespace
kubectl get pods -n loki
```
It is important to create a namespace called loki as our trust policy is set to allow the IAM role to be used by the loki service account in the loki namespace. This is configurable but make sure to update your service account


### [Log Sender (Grafana Alloy) 설치](https://grafana.com/docs/alloy/latest/) ###


### [Grafana Dashboard 설정]() ###




## 레퍼런스 ##

* [Loki Architecture: A Log Aggregation Journey with Grafana](https://sujayks007.medium.com/loki-architecture-a-log-aggregation-journey-with-grafana-bde6d9df6a04)



