
## [Log Sender (Grafana Alloy) 설치](https://grafana.com/docs/alloy/latest/collect/logs-in-kubernetes/) ##
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
  mounts:
    varlog: true               # /var/log 마운트 활성화
    dockercontainers: true     # /var/lib/docker/containers 마운트 활성화
    # 만약 위 옵션이 직접적인 경로를 안 열어준다면 아래 extra를 사용
    extra: []

  # 핵심: 호스트의 로그 경로를 컨테이너 내부로 연결
  extraVolumeMounts:
    - name: varlog
      mountPath: /var/log
      readOnly: true
    - name: varlibdockercontainers
      mountPath: /var/lib/docker/containers
      readOnly: true

  volumes:
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
kubectl logs -n alloy -l app.kubernetes.io/name=alloy | grep -iE "error|failed|401|403|Unauthorized"
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
alloy 파드에서 호스트의 /var/log/pods 경로를 읽어오는지 확인한다.  
```
kubectl exec -it alloy-d6wwh -n alloy -- ls -R /var/log/pods
```
[결과]
```
/var/log/pods:
alloy_alloy-88qdc_e27968d6-961d-43c6-932c-123faaa30a8d
default_nginx-55bbbf955c-rpn2n_4bf70d88-384d-45c5-b4c5-a52b97f8ceec
karpenter_karpenter-86595b65d4-mfskc_0006adcf-1655-4373-be91-e3483b6ba2c9
kube-system_aws-node-vxqn6_d101e04f-3d06-4c2d-8352-ac852aee7d0d
kube-system_coredns-7ccc7b7d9b-gcs9x_c9c47561-5c5d-46d0-a004-119ac5f22522
kube-system_ebs-csi-node-zx46k_10765e4a-b32c-4e4d-9971-0eedf1f0bac8
kube-system_eks-pod-identity-agent-gs4jx_07e057d2-a5e1-4afa-ad3c-485c277110d4
kube-system_kube-proxy-wl2fk_da2be218-b756-49c5-bc41-8720c7b7325c
loki_loki-distributor-d55f8bf48-fth7m_3e2f5540-2ded-41f0-975a-1915101503d7
loki_loki-index-gateway-0_06a0ff1b-dc67-4b47-82a2-f7896f631e15
loki_loki-ingester-1_f50ec90c-2a00-4218-b83d-548a40acf973
loki_loki-querier-54c66b55c8-kjrbb_277a2951-3205-471e-a34d-c23b5756df11
loki_loki-query-frontend-64f4887b9d-dpfqm_ac5ae267-ab25-4304-bc36-194f4543f0b0
loki_loki-query-scheduler-78d8746f46-pt5sv_2b7cbe39-9537-4f3a-b9a0-49dd4bfc6e76
monitoring_prometheus-prometheus-node-exporter-thrkj_6a46cef4-8a7c-4191-b9ca-202b739427a9
```

### Loki 전송 여부 확인 ###
grep 결과가 아무것도 나오지 않는다는 것은 Alloy가 Loki로 로그를 보내는 과정에서 아무런 에러나 경고가 발생하지 않았다 것이다
```
kubectl logs -n alloy -l app.kubernetes.io/name=alloy | grep -iE "error|failed|warn"
```

### Loki 저장 로그 확인 ###
디버깅용 컨테이너를 하나 띄워서 아래 결과를 확인한다. 
```
# Loki에서 최근 5분간의 로그 샘플 조회 (인증 정보 포함)
curl -u "loki:loki" -G -s "loki-gateway.loki.svc.cluster.local/loki/api/v1/query_range" \
  --data-urlencode 'query={cluster="training-on-eks"}' \
  --data-urlencode 'limit=10' \
  -H "X-Scope-OrgId: foo"
```
[결과]
```
{"status":"success","data":{"resultType":"streams","result":[{"stream":{"app":"loki","cluster":"training-on-eks","container":"nginx","container_runtime":"containerd","detected_level":"unknown","filename":"/var/log/pods/loki_loki-gateway-6f6d8c796f-2x8fq_b3920be4-b23e-47a8-8a79-b60256c796c3/nginx/0.log","job":"loki/nginx/","namespace":"loki","pod":"loki-gateway-6f6d8c796f-2x8fq","service_name":"loki"},"values":[["1767768281534797089","2026-01-07T06:44:41.328244533Z stderr F 10.0.11.76 - loki [07/Jan/2026:06:44:41 +0000]  204 \"POST /loki/api/v1/push HTTP/1.1\" 0 \"-\" \"Alloy/v1.12.1 (linux; helm)\" \"-\""],["1767768280280872142","2026-01-07T06:44:40.128261401Z stderr F 10.0.11.76 - loki [07/Jan/2026:06:44:40 +0000]  204 \"POST /loki/api/v1/push HTTP/1.1\" 0 \"-\" \"Alloy/v1.12.1 (linux; helm)\" \"-\""],["1767768279028018791","2026-01-07T06:44:38.827390941Z stderr F 10.0.11.76 - loki [07/Jan/2026:06:44:38 +0000]  204 \"POST /loki/api/v1/push HTTP/1.1\" 0 \"-\" \"Alloy/v1.12.1 (linux; helm)\" \"-\""],["1767768277773925126","2026-01-07T06:44:37.529893945Z stderr F 10.0.11.76 - loki [07/Jan/2026:06:44:37 +0000]  204 \"POST /loki/api/v1/push HTTP/1.1\" 0 \"-\" \"Alloy/v1.12.1 (linux; helm)\" \"-\""]]},
```



## (참고) helm 차트에서 지원되는 value 값 보기 ##
```
$ helm show values grafana/alloy | grep -iE "volume|mount"

  mounts:
    # -- Mount /var/log from the host into the container for log collection.
    # -- Mount /var/lib/docker/containers from the host into the container for log
    # -- Extra volume mounts to add into the Grafana Alloy container. Does not
  # Whether the Alloy pod should automatically mount the service account token.
  automountServiceAccountToken: true
  volumes:
    # -- Extra volumes to add to the Grafana Alloy pod.
  # -- volumeClaimTemplates to add when controller.type is 'statefulset'.
  volumeClaimTemplates: []
```
