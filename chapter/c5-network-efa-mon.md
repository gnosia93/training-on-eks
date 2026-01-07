## EFA Node Exporter ##

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/aws-gallary-efa-node-exporter.png)

Node Exporter 는 헬름으로 프로메테우스 스택을 설치하면 자동으로 설치가 된다. 기본적으로 CPU, 메모리 등을 수집하지만 EFA 지표는 수집하지 않는다. EFA 의 지표를 수집하기 위해서는 별도의 efa 전용 node exporter 를 추가로 설치해 줘야한다. 
```
kubectl get pods -n monitoring -l "app.kubernetes.io/name=prometheus-node-exporter"
```
[결과]
```
NAME                                        READY   STATUS    RESTARTS   AGE
prometheus-prometheus-node-exporter-5hgt4   1/1     Running   0          2d14h
prometheus-prometheus-node-exporter-wf4w6   1/1     Running   0          2d14h
```

### 1. efa 전용 exporter 설치 ###
```
cat <<EOF | kubectl apply -f - 
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: efa-prometheus-exporter
  namespace: monitoring
  labels:
    app: efa-prometheus-exporter
spec:
  selector:
    matchLabels:
      app: efa-prometheus-exporter
  template:
    metadata:
      labels:
        app: efa-prometheus-exporter
    spec:
      hostNetwork: true
      tolerations:
      - operator: "Exists"               # 모든 테인트를 무력화
      containers:
      - name: exporter
        image: public.ecr.aws/hpc-cloud/efa-node-exporter:latest
        args:
          - "--web.listen-address=:9810" # 9810 포트 리슨 
        securityContext:
          privileged: true
        ports:
        - containerPort: 9810
          name: metrics
        volumeMounts:
        - name: sys
          mountPath: /sys
          readOnly: true
      volumes:
      - name: sys
        hostPath:
          path: /sys
---
apiVersion: v1
kind: Service
metadata:
  name: efa-prometheus-exporter
  namespace: monitoring
  labels:
    app: efa-prometheus-exporter
spec:
  ports:
  - port: 9810
    targetPort: 9810
    name: metrics
  selector:
    app: efa-prometheus-exporter
EOF
```

```
kubectl get pods -n kube-system -l app=efa-prometheus-exporter
```

### 2. ServiceMonitor 설정 ###
```
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: efa-exporter-monitor
  namespace: monitoring
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: efa-prometheus-exporter
  endpoints:
  - port: metrics
    interval: 15s
EOF
```

### 3. Grafana 대시보드 연결 ###

* https://grafana.com/grafana/dashboards/20579-efa-metrics-dev/

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-efa.png)

## Amazon CloudWatch Container Insights ##

Amazon CloudWatch Container Insights는 EFA 디바이스 플러그인과 통합되어 EFA 관련 지표를 자동으로 수집한다. 
수집되는 주요 지표는 retrans_bytes, retrans_pkts, retrans_timeout_events 등과 같은 특정 재전송 관련 지표를 제외한 대부분의 EFA 드라이버 지표를 지원한다.

#### 설정 방법 #### 
```
aws iam attach-role-policy --role-name <노드-역할-이름> --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
aws eks create-addon --cluster-name <클러스터-이름> --addon-name amazon-cloudwatch-observability
kubectl get pods -n amazon-cloudwatch
```
성공적으로 활성화되면 CloudWatch 콘솔의 인사이트(Insights) > Container Insights 메뉴에서 클러스터, 노드, Pod 단위의 리소스 사용량(CPU, 메모리 등)을 시각화된 대시보드로 확인 한다.

## 레퍼런스 ##
* https://gallery.ecr.aws/hpc-cloud/efa-node-exporter
* https://grafana.com/grafana/dashboards/20579-efa-metrics-dev/
