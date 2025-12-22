2025년 기준, EKS Network Flow Monitor와 EFA 전용 모니터링을 설치하고 설정하는 핵심 절차를 정리해 드립니다.

#### 1. EKS Network Flow Monitor 설치 (관리형) ####
이 기능은 AWS 매니지드 서비스로, 별도의 복잡한 에이전트 개발 없이 EKS Add-on으로 설치합니다.
IAM 권한 설정: 워커 노드가 CloudWatch에 데이터를 보낼 수 있도록 CloudWatchAgentServerPolicy가 포함된 IAM 역할을 노드에 연결하거나, EKS Pod Identity를 사용하여 권한을 부여합니다.
Add-on 활성화:
AWS 콘솔: EKS 클러스터 선택 → '추가 기능(Add-ons)' 탭 → '추가 기능 더 보기' → Amazon CloudWatch Observability 선택 후 설치.
CLI 명령:
```
aws eks create-addon --cluster-name <클러스터명> --addon-name amazon-cloudwatch-observability
```

결과 확인: CloudWatch 콘솔의 Network Flow Monitor 메뉴에서 대시보드가 자동 생성됩니다.

#### 2. EFA 전용 모니터링 설치 (2025년 신규 지표 대응) ####
EFA의 로우 레벨 지표(SRD 재전송 등)를 수집하려면 CloudWatch Agent를 노드에 배포하여 시스템 파일을 읽어야 합니다.
EFA 드라이버 확인: GPU 인스턴스(P4, P5 등)에 최신 EFA 커널 모듈이 설치되어 있는지 확인합니다.

* CloudWatch Agent 구성:
/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json 설정 파일에 ethtool 관련 설정을 추가하여 EFA 인터페이스 지표를 수집하도록 지정합니다.

* 지표 수집 설정 예시 (2025년 최신 지표 포함):
에이전트 설정에서 proc 및 sysfs 내의 /sys/class/infiniband/ 경로에 접근하여 retransmits, rdma_read_bytes 등의 지표를 수집하도록 구성합니다.

* Prometheus/Grafana 사용 시 (권장):
Node Exporter를 설치하고, ethtool 콜렉터를 활성화하면 EFA 지표가 자동으로 수집됩니다.
Grafana에서 EFA Performance Dashboard 템플릿을 불러와 연결합니다.

#### 3. 요약: 단계별 실행 가이드 ####
* 1단계: EKS 콘솔에서 CloudWatch Observability 애드온을 활성화하여 전체 네트워크 흐름을 먼저 잡습니다.
* 2단계: GPU 노드 그룹에 CloudWatch Agent 또는 Prometheus Node Exporter를 데몬셋(DaemonSet)으로 배포합니다.
* 3단계: Amazon CloudWatch 콘솔에서 Network Flow Monitor 탭과 Metrics 탭을 각각 확인하여 통합 모니터링 체계를 구축합니다.
구체적인 가이드는 Amazon EKS 관측성 가이드 및 CloudWatch Network Flow Monitor 공식 문서에서 더 자세히 확인하실 수 있습니다.
