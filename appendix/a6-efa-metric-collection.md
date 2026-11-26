# EFA 네트워크 지표 수집 해설

> "EFA 트래픽은 일반 이더넷 인터페이스 지표에 잡히지 않습니다. `/sys/class/infiniband/*/ports/1/hw_counters`의 카운터(또는 `rdma statistic`)를 node_exporter textfile collector로 수집해 Grafana에 올렸습니다."

## 1. EFA 트래픽이 이더넷 지표에 안 잡히는 이유

일반 TCP 통신은 애플리케이션에서 커널 TCP/IP 스택과 ENA 드라이버를 거쳐 NIC로 나갑니다. 커널이 모든 패킷을 처리하므로 인터페이스별 통계(`/proc/net/dev`)에 쌓입니다. node_exporter의 netdev collector는 이 통계를 `node_network_transmit_bytes_total` 같은 지표로 내보냅니다.

EFA 통신은 이 경로를 타지 않습니다.

- NCCL은 aws-ofi-nccl 플러그인과 libfabric(efa provider)을 거쳐, 사용자 공간에서 EFA 장치의 큐에 작업 요청을 직접 넣습니다(OS bypass). NIC는 GPU 메모리에서 데이터를 직접 읽어(GPUDirect RDMA) SRD 프로토콜로 보냅니다.
- 커널 드라이버(efa.ko)는 큐 생성이나 메모리 등록 같은 준비 단계에만 관여합니다. 데이터 패킷은 커널 네트워크 스택을 지나지 않으므로 `/proc/net/dev`에 남지 않습니다.
- EFA 인터페이스(EFA with ENA)를 붙이면 IP용 ENA 장치와 RDMA용 EFA 장치가 따로 생깁니다. ENA 장치는 `ens…` 같은 이더넷 인터페이스로 보이고, EFA 장치는 `rdmap47s0` 같은 RDMA 장치로 보입니다. EFA-only 인터페이스는 ENA 장치 없이 EFA 장치만 만들기 때문에 이더넷 인터페이스가 아예 없습니다.

그래서 ENA 지표에는 SSH, 스토리지 I/O, NCCL 초기 연결 같은 IP 트래픽만 보입니다. 학습 중 노드 사이를 오가는 통신량은 빠져 있습니다.

## 2. 카운터가 있는 곳

EFA 드라이버는 커널 RDMA 서브시스템에 장치를 등록하고, 장치에서 읽은 카운터를 sysfs 파일로 내놓습니다. 경로를 부분별로 보면 다음과 같습니다.

- `/sys/class/infiniband/`: 커널에 등록된 RDMA 장치 목록입니다. 이름은 InfiniBand지만 EFA도 여기에 나옵니다.
- `*`: EFA 장치 이름입니다. EFA 인터페이스마다 하나씩 있습니다(1-1의 노드는 16개). 이름은 PCI 위치에서 붙습니다. `lesson/3-gpu.md`에 나온 `rdmap47s0`은 PCI 주소 `0000:2f:00.0`의 장치입니다(0x2f = 47). 이 주소를 `nvidia-smi topo -m`과 맞춰 보면 어느 GPU 옆에 붙은 NIC인지 알 수 있습니다.
- `ports/1`: EFA 장치는 포트가 하나뿐이라 항상 1입니다.
- `hw_counters/`: 카운터마다 파일이 하나씩 있습니다. 값은 인스턴스 시작이나 드라이버 리셋 이후의 누적값입니다. `lifespan` 파일은 커널이 값을 캐시하는 시간(ms)이므로 카운터가 아닙니다.

`rdma statistic show`(iproute2의 `rdma` 도구)는 같은 드라이버 카운터를 netlink로 읽어 `link rdmap47s0/1 tx_bytes … rx_bytes …` 형태로 보여 줍니다. `-j`를 붙이면 JSON으로 나옵니다. 값은 sysfs와 같습니다. 스크립트에서는 도구를 따로 설치할 필요가 없는 sysfs가 더 간단합니다.

포트 카운터를 쓰임새별로 나누면 다음과 같습니다.

| 분류 | 카운터 | 의미 |
|---|---|---|
| 전체 트래픽 | `tx_bytes`, `rx_bytes`, `tx_pkts`, `rx_pkts` | 이 EFA 장치가 보내고 받은 바이트와 패킷. 처리량 패널은 이 값으로 만듭니다 |
| 연산별 | `send_bytes`, `send_wrs`, `recv_bytes`, `recv_wrs` | send/recv 연산으로 주고받은 바이트와 작업 요청(WR) 수 |
| | `rdma_read_bytes`, `rdma_read_resp_bytes`, `rdma_read_wrs` | RDMA read로 가져온 바이트(요청한 쪽), 상대의 read에 응답해 보낸 바이트(응답한 쪽), 완료된 read 수 |
| | `rdma_write_bytes`, `rdma_write_recv_bytes`, `rdma_write_wrs` | RDMA write로 상대에게 쓴 바이트(쓴 쪽), 상대가 이 노드에 쓴 바이트(받은 쪽), 완료된 write 수 |
| 오류 | `rx_drops`, `rdma_read_wr_err`, `rdma_write_wr_err` | 받은 뒤 버린 패킷, 로컬이나 원격 오류가 난 RDMA 연산 |
| SRD 상태 (Nitro v4 이상) | `retrans_bytes`, `retrans_pkts` | SRD가 재전송한 바이트와 패킷 |
| | `retrans_timeout_events` | 타임아웃 때문에 네트워크 경로를 바꾼 횟수 |
| | `impaired_remote_conn_events` | 연결이 impaired 상태가 되어 전송률 상한이 낮아진 횟수 |
| | `unresponsive_remote_events` | 상대 쪽 연결이 응답하지 않은 횟수 |

- NCCL이 send/recv, RDMA read, RDMA write 중 무엇을 쓰는지는 인스턴스 세대와 libfabric·aws-ofi-nccl 버전에 따라 다릅니다. 처리량은 `tx_bytes`/`rx_bytes`로 보고, 연산별 카운터는 트래픽이 어느 경로로 가는지 확인할 때 씁니다.
- EC2 문서는 `rdma_write_bytes`를 "다른 인스턴스가 써 넣은 바이트"라고 설명합니다. 하지만 받은 쪽 값인 `rdma_write_recv_bytes`가 따로 있고 CloudWatch 문서도 "transmitted"라고 적고 있으므로, 쓴 쪽 값으로 보는 것이 맞습니다.
- 장치 수준 카운터(`/sys/class/infiniband/<dev>/hw_counters`)도 따로 있습니다. `reg_mr_err`, `create_qp_err`, `cmds_err` 같은 제어 경로 오류 카운터라서 메모리 등록이나 QP 생성이 실패할 때 확인합니다.

## 3. node_exporter textfile collector를 쓴 이유

node_exporter의 기본 collector로는 EFA 카운터를 얻을 수 없습니다.

- netdev collector는 커널 네트워크 인터페이스 통계만 읽습니다(1번 참고).
- infiniband collector는 `/sys/class/infiniband`를 읽기는 합니다. 하지만 IB 표준 포트 카운터(`ports/*/counters`의 `port_xmit_data` 등)와, `hw_counters` 가운데 Mellanox 같은 RoCE NIC용으로 미리 정해 둔 이름(`np_cnp_sent`, `out_of_buffer` 등)만 해석합니다. EFA의 `tx_bytes`, `rdma_read_bytes`, `retrans_pkts`는 그 목록에 없어서 버려집니다.

textfile collector를 쓰면 가장 적게 바꾸고 이 빈틈을 메울 수 있습니다. node_exporter를 `--collector.textfile.directory=<디렉터리>`로 띄우면, 스크레이프할 때마다 그 디렉터리의 `*.prom` 파일을 읽어 자기 지표와 함께 내보냅니다. 따라서 sysfs 값을 Prometheus 텍스트 형식으로 바꿔 파일에 쓰는 스크립트 하나만 있으면 됩니다. 새 exporter, 새 포트, 새 스크레이프 설정은 필요 없습니다. 지표는 그 노드의 다른 node_exporter 지표와 같은 `instance` 라벨을 달고 들어옵니다.

운영할 때는 다음을 지켜야 합니다.

- 파일은 같은 디렉터리에 임시 이름으로 쓴 뒤 `mv`로 교체합니다. 쓰는 중인 파일을 node_exporter가 읽으면 파싱 오류가 납니다.
- textfile 형식은 타임스탬프를 지원하지 않습니다. 값은 파일을 마지막으로 쓴 시점의 것이므로, 쓰기 주기를 스크레이프 주기(예: 15초)보다 길지 않게 둡니다.
- 스크립트가 죽어도 파일은 남아 있어 값이 그대로 멈춥니다. 그러면 `rate()`가 0이 되어 트래픽이 끊긴 것처럼 보입니다. 이를 막으려면 `node_textfile_mtime_seconds`로 파일이 오래됐는지 경보를 겁니다.

같은 카운터는 CloudWatch agent로도 모을 수 있고, EKS에서는 Container Insights가 EFA 지표를 지원합니다. 다만 Container Insights는 `retrans_*`, `impaired_remote_conn_events`, `unresponsive_remote_events`를 수집하지 않습니다. 이미 Prometheus와 Grafana를 쓰는 환경이라면 textfile 방식이 바꿀 것이 가장 적습니다.

## 4. 수집 스크립트 예

```bash
#!/bin/bash
# /usr/local/bin/efa-textfile.sh: EFA 포트 카운터를 textfile collector 형식으로 씁니다.
DIR=/var/lib/node_exporter/textfile_collector
TMP="$DIR/.efa.prom.$$"          # *.prom 패턴에 걸리지 않는 임시 이름

ports=(/sys/class/infiniband/*/ports/*/hw_counters)
[ -d "${ports[0]}" ] || exit 0    # EFA가 없는 노드

{
  for f in "${ports[0]}"/*; do
    c=${f##*/}
    [ "$c" = lifespan ] && continue              # 캐시 주기라 제외
    echo "# TYPE node_efa_${c}_total counter"    # 같은 지표 이름의 줄은 한데 모아 씁니다
    for p in "${ports[@]}"; do
      dev=$(cut -d/ -f5 <<< "$p")                # 예: rdmap47s0
      port=$(cut -d/ -f7 <<< "$p")
      v=$(cat "$p/$c" 2>/dev/null) || continue
      echo "node_efa_${c}_total{device=\"$dev\",port=\"$port\"} $v"
    done
  done
} > "$TMP" && mv "$TMP" "$DIR/efa.prom"
```

출력은 `node_efa_tx_bytes_total{device="rdmap47s0",port="1"} 81234567890` 같은 줄이 카운터와 장치 수만큼 나옵니다.

이 스크립트는 15초마다 실행합니다(systemd 서비스 안의 루프나 타이머로). ParallelCluster에서는 설치 과정을 컴퓨트 큐의 `CustomActions.OnNodeConfigured` 스크립트에 넣으면 됩니다. 그러면 노드마다 손으로 설치할 필요가 없고, 자동 교체로 새로 들어온 노드(2-1)도 바로 수집을 시작합니다. EKS에서는 node-exporter DaemonSet에 이 스크립트를 돌리는 사이드카를 붙이고, emptyDir를 textfile 디렉터리로 함께 마운트합니다.

## 5. Grafana 쿼리

```promql
# NIC별 송신 처리량 (패널 단위: bits/sec(SI))
rate(node_efa_tx_bytes_total[1m]) * 8

# 노드 합계 (p5en은 NIC당 200Gbps, 노드당 3.2Tbps)
sum by (instance) (rate(node_efa_tx_bytes_total[1m])) * 8

# SRD 재전송 비율
rate(node_efa_retrans_pkts_total[5m]) / rate(node_efa_tx_pkts_total[5m])

# 드롭, 연결 이상 발생
increase(node_efa_rx_drops_total[5m]) > 0
increase(node_efa_impaired_remote_conn_events_total[5m]) > 0

# NCCL이 TCP로 떨어진 것 같음(4-1): EFA는 거의 0인데 ENA는 바쁨 (임계값은 예시)
sum by (instance) (rate(node_efa_tx_bytes_total[5m])) < 1e6
  and sum by (instance) (rate(node_network_transmit_bytes_total{device!="lo"}[5m])) > 1e8

# 수집 스크립트가 멈춤
time() - node_textfile_mtime_seconds{file=~".*efa.prom"} > 60
```

`rate()` 범위는 스크레이프 주기의 4배 이상으로 둡니다. 스크레이프 주기가 15초라면 `1m`입니다.

## 6. 1-1 사례에서 이 지표로 보이는 것과 안 보이는 것

- **GPU 지표가 정상으로 나온 이유:** GPU 사용률은 GPU에서 커널이 돌고 있는 시간의 비율입니다. NCCL 커널은 데이터를 기다리는 동안에도 계속 돌고 있으므로, 네트워크가 느려도 사용률은 높게 나옵니다.
- **보이는 것:** 64노드 작업이 느릴 때 EFA 처리량이 평소의 3분의 1 아래로 떨어진 것이 바로 보입니다. 그래서 병목이 계산이 아니라 네트워크라는 점을 확인할 수 있습니다. NCCL이 TCP로 떨어진 경우(4-1)는 EFA 처리량이 0 가까이 나오므로 바로 구분됩니다.
- **바로 안 보이는 것:** NCCL은 데이터를 채널별로 같은 크기로 나눠 NIC들에 고르게 싣습니다. 집합 연산은 가장 느린 NIC의 몫이 끝나야 완료됩니다. 그래서 1분 평균으로 보면 문제 NIC 2장, 같은 노드의 정상 NIC 14장, 다른 노드의 NIC가 모두 함께 낮아집니다. 바이트 카운터만으로는 어느 노드의 어느 NIC가 문제인지 드러나지 않습니다. `incident.md`의 "GPU 지표만으로는 보이지 않는다"는 말은 맞지만, 이 카운터를 모은다고 1-1 노드가 바로 찍히지는 않습니다.

문제 NIC를 짚어 내려면 다음을 더합니다.

1. NIC를 서로 독립적으로 쓰는 테스트를 돌리면서 NIC별 패널을 봅니다. 예를 들어 GPU 번호별로 따로 돌리는 2노드 테스트에서는 그 GPU 옆 NIC만 쓰입니다. 그러면 문제 GPU를 쓴 테스트에서만 NIC 2장이 낮게 나옵니다(1-1에서는 약 20%).
2. 같은 스크립트로 EFA 장치의 PCIe 링크 폭도 내보냅니다. 1-1은 x8이어야 할 링크가 x2로 붙어 있었으므로 아래 조건 하나로 경보가 됩니다. 8번 운영 교훈 1에 있는 PCIe 링크 확인을 상시 지표로 만드는 방법입니다.

```bash
# efa-textfile.sh의 { } 안에 추가
echo "# TYPE node_efa_pcie_link_width gauge"
for d in /sys/class/infiniband/*; do
  for k in current max; do
    v=$(cat "$d/device/${k}_link_width" 2>/dev/null) || continue
    echo "node_efa_pcie_link_width{device=\"${d##*/}\",kind=\"$k\"} $v"
  done
done
```

```promql
node_efa_pcie_link_width{kind="current"} < ignoring(kind) node_efa_pcie_link_width{kind="max"}
```

## 7. 참고: a2 문서와 맞지 않는 부분

`lesson/a2-efa-observability.md`에는 실제 카운터와 맞지 않는 부분이 있습니다.

- **카운터 이름:** EFA 드라이버의 카운터 이름은 `rx_drops`, `retrans_bytes`, `retrans_pkts`입니다. `tx_dropped`나 `rx_errors`에 해당하는 카운터는 없으므로, 그 이름을 쓰는 쿼리와 경보에는 값이 나오지 않습니다.
- **단위:** 쿼리는 `* 8 / 1024 / 1024 / 1024`로 이미 단위를 바꿨는데 패널 단위를 `bps`로 두어서, 화면에 나오는 값이 틀립니다. `* 8`만 곱하고 패널 단위를 bits/sec(SI)로 두면 됩니다.

## 출처

- [EC2 EFA 모니터링 문서](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-working-monitor.html): 카운터 의미, Container Insights 제외 항목
- [EFA 기본 문서](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html): ENA/EFA 장치 구분, EFA-only 인터페이스
- [커널 efa_verbs.c](https://github.com/torvalds/linux/blob/master/drivers/infiniband/hw/efa/efa_verbs.c): 카운터 이름
- [procfs class_infiniband.go](https://github.com/prometheus/procfs/blob/master/sysfs/class_infiniband.go): infiniband collector가 읽는 이름
- [node_exporter README](https://github.com/prometheus/node_exporter#textfile-collector): textfile collector
- [Container Insights EFA 지표](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Container-Insights-metrics-enhanced-EKS.html#Container-Insights-metrics-EFA): CloudWatch의 EFA 지표 설명
