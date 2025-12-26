
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
