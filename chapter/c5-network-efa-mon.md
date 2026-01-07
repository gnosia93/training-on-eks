### 1. EFA Node Exporter 설정 ###
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

#### (참고) 프로메테우스 스택 기본 설정 ####
헬름은 차트가 제공하는 기본 설정값을 조회하는 기능을 제공해 준다. 여기서는 프로메테우스 스택이 제공하는 프로메테우스, 그라파나, alertManager 등의 모든 모듈들의 기본 설정값을 values.yaml 파일에 저정하고 있다.    
```
helm show values prometheus/kube-prometheus-stack > values.yaml
```

#### (참고) 프로메테우스 변경 내용 조회 ####
프로메테우스 모듈에 대해서 사용자가 수정한 설정값만을 보여준다. 
```
helm get values prometheus -n monitoring > my-prometheus-values.yaml
```

#### efa 모니니터링 설정 ( 이 부분은 필요 없을 수도 ) ####
이 설정은 node-exporter가 호스트의 EFA 장치를 스캔하도록 허용한다.       
```
cat <<EOF > efa-tuning.yaml
prometheus-node-exporter:
  extraArgs:
    - --collector.ethtool
    - --collector.ethtool.device-include=.*
    - --collector.infiniband
EOF
```
```
helm upgrade prometheus prometheus/kube-prometheus-stack -n monitoring \
    -f efa-tuning.yaml \
    --reuse-values             # 기존 설정에 추가
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




## 레퍼런스 ##
* https://gallery.ecr.aws/hpc-cloud/efa-node-exporter
* https://grafana.com/grafana/dashboards/20579-efa-metrics-dev/
