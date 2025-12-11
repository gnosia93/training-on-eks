EKS 오토모드에서는 Bottlerocket 기반의 AMI(Amazon Machine Image)를 사용하고 있다.
GPU 메토릭 정보를 추출하기 위해서 DCGM exporter 파드를 스케줄하는 경우, 아래의 POD 로그에서 보이는 것 처럼 libdcgm.so.4 파일이 설치되어 있지 않아서 파드가 크래시 된다.
```
Defaulted container "nvidia-dcgm-exporter" out of: nvidia-dcgm-exporter, toolkit-validation (init)
time=2025-12-11T06:53:46.917Z level=INFO msg="Starting dcgm-exporter" Version=4.4.2-4.7.0
time=2025-12-11T06:53:46.918Z level=ERROR msg="the libdcgm.so.4 library was not found. Install Data Center GPU Manager (DCGM)."
```

## 어떻게 GPU 정보는 뽑아오지 ? ##






헬름을 먼저 설치하고 NVIDIA 레포지토리를 등록한다. 
```
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update
```





## 레퍼런스 ##
* [Tracking GPU Usage in K8s with Prometheus and DCGM: A Complete Guide](https://medium.com/@penkow/tracking-gpu-usage-in-k8s-with-prometheus-and-dcgm-a-complete-guide-7c8590809d7c)
* https://tech.inflab.com/20250827-bottlerocket-ami-gpu-issue/



## 프로메테우스 설치 ##
helm 차트를 설치한다. 
```
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4
sh get_helm.sh

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```
helm 설치후 프로메테우스 스택(kube-prometheus-stack)을 설치한다. 이 스택에는 Prometheus Operator, Prometheus, Grafana, Alertmanager 등이 포함되어 있다.
```
helm install kube-prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
kubectl get pods -n monitoring
```
[결과]
```
NAME                                                     READY   STATUS    RESTARTS   AGE
alertmanager-kube-prometheus-kube-prome-alertmanager-0   2/2     Running   0          2m21s
kube-prometheus-grafana-7bd844d876-97bvc                 3/3     Running   0          2m41s
kube-prometheus-kube-prome-operator-5698d75fcb-rgt58     1/1     Running   0          2m41s
kube-prometheus-kube-state-metrics-5849b6fdb-5j5qw       1/1     Running   0          2m41s
kube-prometheus-prometheus-node-exporter-9f98j           1/1     Running   0          2m41s
kube-prometheus-prometheus-node-exporter-mzx7j           1/1     Running   0          2m41s
prometheus-kube-prometheus-kube-prome-prometheus-0       2/2     Running   0          2m21s
```

그라파나 서비스를 로드밸런스 타입으로 변경하여 외부로 노출시킨다.

[grafana-service.yaml]
```
apiVersion: v1
kind: Service
metadata:
  name: kube-prometheus-grafana
  namespace: monitoring
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "external" # ALB/NLB용으로 프로비저닝
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing" # 인터넷 연결형
spec:
  ports:
  - name: http-web
    port: 80
    protocol: TCP
    targetPort: grafana
  selector:
    app.kubernetes.io/instance: kube-prometheus
    app.kubernetes.io/name: grafana
  sessionAffinity: None
  type: LoadBalancer
  loadBalancerSourceRanges:
    - "122.36.213.114/32" # 특정 회사/집 IP 대역만 허용
```
접속 가능한 출발지 IP 대역폭은 loadBalancerSourceRanges 필드를 이용하여 정의한다. 
```
kubectl apply -f grafana-service.yaml
```




 

echo "----------------------------------------------------"
echo "설치가 완료되었습니다."
echo "Grafana에 접속하려면 다음 포트 포워딩 명령어를 새 터미널에서 실행하십시오:"
echo "kubectl port-forward svc/prometheus-grafana 8080:80 -n monitoring"
echo "그 후 웹 브라우저에서 http://localhost:8080 으로 접속하십시오."
echo "기본 사용자 이름: admin, 기본 비밀번호: prom-operator"
echo "----------------------------------------------------"

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
