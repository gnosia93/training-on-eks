
### 1. Node Exporter 설정 (EFA 지표 수집) ###
기본적으로 node_exporter는 CPU, 메모리 등을 수집하지만, EFA 지표를 가져오기 위해서는 ethtool 콜렉터가 활성화되어야 합니다.

#### 헬름(Helm)으로 설치 시 설정 ####
이미 Prometheus Stack을 사용 중이라면 values.yaml 파일에 아래 내용을 추가하여 업그레이드합니다.
```
prometheus-node-exporter:
  extraArgs:
    - --collector.ethtool  # ethtool 콜렉터 활성화
    # 특정 디바이스만 필터링하고 싶을 경우 아래 옵션 추가 (선택사항)
    # - --collector.ethtool.device-include=^rdma.* 
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
