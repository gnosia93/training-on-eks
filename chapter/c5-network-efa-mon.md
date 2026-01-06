
### 1. Node Exporter 설정 (EFA 지표 수집) ###

Node Exporter 는 헬름으로 프로메테우스 스택을 설치하면 자동으로 설치가 된다. 기본적으로 CPU, 메모리 등을 수집하지만 EFA 지표를 가져오기 위해서는 ethtool 콜렉터가 활성화되어야 한다.
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

#### efa 모니니터링 설정 ####
efa 네트워크 인터페이스를 모니터링 하기위해서 collector.ethtool를 추가해야 한다. --reuse-values 옵션을 이용하여 기존 설정에 추가하도록 한다. 이 옵션을 사용하지 않으면 기존에 설정했던 내용은 default 값으로 변경된다.         
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
node_exporter의 ethtool 콜렉터는 Pod의 /sys/class/infiniband 경로에서 EFA 메트릭을 수집하여 node_net_ethtool 형태의 메트릭으로 변환한다.

### 2. 수집여부 확인하기 ###
```
EFA_NODE=$(kubectl get pod efa-test-pod -o jsonpath='{.spec.nodeName}')
POD_NAME=$(kubectl get pods -n monitoring \
    -l "app.kubernetes.io/name=prometheus-node-exporter" \
    --field-selector spec.nodeName=${EFA_NODE} \
    -o jsonpath='{.items[0].metadata.name}')

echo ${POD_NAME} " / " ${EFA_NODE}

kubectl exec -it ${POD_NAME} -n monitoring -- wget -qO- localhost:9100/metrics | grep node_infiniband
kubectl exec -it ${POD_NAME} -n monitoring -- wget -qO- localhost:9100/metrics | grep "node_infiniband" | grep -E "bytes|packets"
```


### 3. 수집되는 주요 EFA 메트릭 ###

정상적으로 설정되면 Prometheus에서 다음과 같은 쿼리로 EFA 지표를 확인할 수 있다.
* 전송된 바이트 수: node_net_ethtool{device="rdma0", stat="rdma_read_bytes"}
* 수신된 바이트 수: node_net_ethtool{device="rdma0", stat="rdma_write_bytes"}
* EFA 에러 카운트: node_net_ethtool{device="rdma0", stat="lif_error_errors"}


### 3. Grafana 대시보드 연결 ###

* Grafana 대시보드에서 우측의 [New 버튼] -> [Import 서브메뉴] 을 선택한 다음 대시보드 Node Exporter Full(ID: 1860)를 임포트 한다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/node-exporter-full.png)

* Grafana 대시보드에서 우측의 [New 버튼] -> [Import 서브메뉴] 을 선택한 다음 대시보드 Node/Network(ID: 22273)를 임포트 한다. 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/network-mon.png)  


## 레퍼런스 ##
* https://gallery.ecr.aws/hpc-cloud/efa-node-exporter
* https://grafana.com/grafana/dashboards/20579-efa-metrics-dev/
