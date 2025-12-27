
### 1. Node Exporter 설정 (EFA 지표 수집) ###
기본적으로 node_exporter는 CPU, 메모리 등을 수집하지만, EFA 지표를 가져오기 위해서는 ethtool 콜렉터가 활성화되어야 한다.


#### 프로메테우스 스택 기본 설정 ####
스택 전체에 대한 기본 설정을 values.yaml 로 만든다. 여기에는 프로메테우스, 그라파나, alertManager 등의 모듈들의 기본 설정값이 들어 있다.  
```
helm show values prometheus/kube-prometheus-stack > values.yaml
```

#### 프로메테우스 변경 내용 조회 ####
프로메테우스에 대해서 사용자가 수정한 설정값을 보여준다. 
```
helm get values prometheus -n monitoring > my-prometheus-values.yaml
```

#### efa 모니니터링 설정 ####
efa 를 모니터링 하기위해서 collector.ethtool 를 추가한다. --reuse-values 옵션을 이용하여 기존 설정에 Update 한다. 이 옵션을 사용하지 않으면 기존에 설정했던 내용은 default 값으로 변경된다.         
```
cat <<EOF > efa-tuning.yaml
prometheus-node-exporter:
  extraArgs:
    - --collector.ethtool
    - --collector.ethtool.device-include=^rdma.*
EOF

helm upgrade prometheus prometheus/kube-prometheus-stack -n monitoring
    -f efa-tuning.yaml \
    --reuse-values             # 기존 설정에 추가
```

```
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack -f values.yaml
```
EFA는 리눅스 커널에서 네트워크 인터페이스로 인식됩니다. node_exporter의 ethtool 콜렉터는 노드의 /sys/class/net/ 경로에 있는 통계 정보를 읽어 node_net_ethtool 형태의 메트릭으로 변환합니다.

### 2. 수집되는 주요 EFA 메트릭 ###
정상적으로 설정되면 Prometheus에서 다음과 같은 쿼리로 EFA 지표를 확인할 수 있습니다.
* 전송된 바이트 수: node_net_ethtool{device="rdma0", stat="rdma_read_bytes"}
* 수신된 바이트 수: node_net_ethtool{device="rdma0", stat="rdma_write_bytes"}
* EFA 에러 카운트: node_net_ethtool{device="rdma0", stat="lif_error_errors"}


### 3. Grafana 대시보드 연결 ###
AWS에서 공식적으로 제공하거나 커뮤니티에서 널리 쓰이는 대시보드 템플릿을 활용하면 시각화가 쉽습니다.

#### 권장 대시보드 템플릿 ####
* AWS 공식 EFA Grafana Dashboard: AWS 샘플 저장소에서 제공하는 JSON 파일을 다운로드하여 임포트할 수 있습니다.
* Grafana ID 이용: Grafana "Import" 메뉴에서 ID 14531 또는 12457 등을 시도해 볼 수 있으나, EFA 전용인 AWS 공식 가이드의 JSON을 복사하여 사용하는 것이 가장 정확합니다.

#### 임포트 방법 ####
* Grafana 접속 -> 왼쪽 메뉴의 Dashboards -> New -> Import 클릭.
* 위에서 다운로드한 JSON 파일을 업로드하거나 JSON 텍스트를 붙여넣기 합니다.
* 데이터 소스로 현재 연동된 Prometheus를 선택합니다
