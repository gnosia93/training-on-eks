## helm ##
```
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4
sh get_helm.sh

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```



## 설명 ##

PyTorchJob을 모니터링하는 환경을 구축하려면 Kubernetes 클러스터의 표준 모니터링 스택인 Prometheus와 Grafana를 활용하는 것이 일반적입니다. Kubeflow 환경에서는 이러한 도구들이 이미 통합되어 있을 가능성이 높습니다.
모니터링 환경 구축 및 설정 방법은 다음과 같습니다.
1. 전제 조건 및 구성 요소
PyTorchJob의 상태와 성능 지표를 수집하고 시각화하려면 다음 구성 요소가 필요합니다.
Kubernetes 클러스터: EKS 등 쿠버네티스 환경.
Kubeflow Training Operator: PyTorchJob CRD 및 컨트롤러가 설치되어 있어야 합니다.
Prometheus: 메트릭(지표)을 수집하고 저장합니다.
Grafana: Prometheus에서 수집한 메트릭을 시각화하여 대시보드를 제공합니다.
NVIDIA GPU Operator (선택): GPU 활용률(Utilization), 온도, 메모리 사용량 같은 하드웨어 메트릭을 수집하려면 NVIDIA GPU Operator 설치가 권장됩니다.
2. 모니터링 방법 및 단계
모니터링은 크게 두 가지 관점에서 이루어집니다.
A. PyTorchJob 상태 및 라이프사이클 모니터링 (Kubernetes 레벨)
PyTorchJob 리소스 자체의 상태 변화(생성, 진행 중, 성공, 실패 등)를 모니터링합니다.
Kubectl 명령어 활용:
기본적으로 kubectl 명령어로 현재 상태를 확인할 수 있습니다.
bash
kubectl get pytorchjobs -n pytorch
kubectl describe pytorchjob pytorch-dist-job -n pytorch
kubectl logs pytorch-dist-job-worker-0 -n pytorch
코드를 사용할 때는 주의가 필요합니다.

Kubernetes Event 모니터링:
Job의 라이프사이클 변경 시 발생하는 이벤트를 모니터링 시스템(예: Prometheus Alertmanager)과 연동하여 알림을 받을 수 있습니다.
bash
kubectl get events -n pytorch --field-selector involvedObject.name=pytorch-dist-job
코드를 사용할 때는 주의가 필요합니다.

B. 학습 성능 및 GPU/CPU 리소스 메트릭 모니터링 (Prometheus/Grafana 레벨)
학습 중인 Pod들의 리소스 사용량과 성능을 시각적으로 모니터링합니다.
단계 1: 메트릭 수집 활성화
쿠버네티스 클러스터에 Prometheus가 설치되어 있다면, Prometheus Operator가 Pod의 메트릭을 자동으로 스크래핑하도록 설정해야 합니다.
cAdvisor 메트릭: 쿠버네티스는 기본적으로 각 노드의 리소스 사용량(CPU, 메모리)을 cAdvisor를 통해 Prometheus 형식으로 노출합니다.
GPU 메트릭: NVIDIA GPU Operator를 설치하면 dcgm-exporter가 배포되어 GPU 사용률, 온도, 전력 등 상세 메트릭을 Prometheus로 보냅니다.
단계 2: Grafana 대시보드 구축
수집된 메트릭을 Grafana에서 시각화합니다.
사전 구축된 대시보드 활용: Grafana Labs에서 제공하는 쿠버네티스 및 GPU 관련 공개 대시보드 템플릿(예: Node Exporter Full, NVIDIA GPU Metrics)을 가져와서 사용할 수 있습니다.
커스텀 대시보드: Job 이름이나 Pod 레이블을 기준으로 필터링하여 특정 PyTorchJob의 리소스 사용량을 집중 모니터링하는 대시보드를 직접 구성할 수 있습니다.
3. 알림(Alert) 설정
Prometheus Alertmanager를 사용하여 특정 조건(예: GPU 사용률이 5분 이상 0%일 때, Pod이 Failed 상태로 전환될 때)에서 Slack, 이메일, PagerDuty 등으로 알림을 받도록 설정합니다.

요약
가장 효과적인 모니터링 환경은 Kubeflow, Prometheus, Grafana 스택을 활용하는 것입니다. 이를 통해 Job의 성공/실패 여부뿐만 아니라, GPU가 실제로 효율적으로 사용되고 있는지 실시간으로 확인할 수 있습니다.
