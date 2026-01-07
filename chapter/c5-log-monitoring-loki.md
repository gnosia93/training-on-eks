

## [Log Backend(Loki) 설치](https://grafana.com/docs/loki/latest/setup/install/helm/deployment-guides/aws/) ##
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





