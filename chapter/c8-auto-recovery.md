## 1. Node Monitoring Agent 설치 ##
클러스터 설정에서 eks-node-monitoring-agent 애드온을 추가하여 설치해야 합니다. 이 에이전트가 노드의 로그를 분석하여 장애를 감지하는 역할을 합니다.
```
export CLUSTER_NAME="training-on-eks"
export K8S_VERSION="1.34"

NMA_VERSION=$(aws eks describe-addon-versions \
    --kubernetes-version ${K8S_VERSION} \
    --addon-name eks-node-monitoring-agent \
    --query 'addons[0].addonVersions[?compatibilities[0].defaultVersion==`true`].addonVersion' \
    --output text)
echo "Node Monitoring Agent Version: "${NMA_VERSION}

aws eks create-addon \
    --cluster-name ${CLUSTER_NAME} \
    --addon-name eks-node-monitoring-agent \
    --addon-version ${NMA_VERSION}  # 2025년 최신 버전 확인 필요

# 에이전트 포드 확인
kubectl get pods -n kube-system | grep node-monitoring-agent
```
Node Monitoring Agent는 GPU 노드의 상태를 정밀하게 진단하기 위해 내부적으로 NVIDIA의 핵심 라이브러리인 NVML(NVIDIA Management Library)과 DCGM을 활용한다.


## 2. 노드 타입별 설정 ##

### 2-1. 매니지드 노드 ###
관리형 노드 그룹에서는 오토 모드와 달리 사용자가 직접 기능을 활성화해야 한다.  
NMA 에이전트가 노드의 커널, 네트워크, 스토리지, GPU 상태를 모니터링하다가 문제가 발견되면 Kubernetes NodeCondition을 업데이트한다.
상태가 나빠진 노드를 발견하면 EKS 컨트롤 플래인이 해당 노드를 자동으로 격리(Cordon) 및 비우기(Drain)한 후, 새로운 인스턴스로 교체한다. 

#### 노드 그룹의 'Node Repair' 활성화 ####
에이전트만 설치한다고 복구가 자동으로 수행되지 않습니다. 매니지드 노드 그룹 설정에서 node-repair 기능을 Enabled로 변경해야 한다.
```
aws eks update-nodegroup-config --cluster-name <클러스터명> \
  --nodegroup-name <노드그룹명> \
  --node-repair-config enabled=true
```

### 2-2. 카펜터 노드 ###

카펜터는 기본적으로 NotReady 상태인 노드를 감지하여 교체한다. Node Monitoring Agent를 설치하면 NotReady가 되기 전의 미세한 하드웨어 장애(GPU 오류, 디스크 읽기 전용 등) 단계에서 더 빠르게 대응할 수 있다. 매니지드 노드 그룹에서는 AWS 백엔드가 복구를 수행하지만, 카펜터 환경에서는 Node Monitoring Agent 가 문제를 감지하고 카펜터가 노드를 교체하는 협업 구조로 작동한다. 
Node Monitoring Agent는 노드 장애를 발견되면, 해당 노드의 NodeCondition을 업데이트 한다. (예: StorageReady=False).
카펜터의 Disruption(중단) 컨트롤러가 이를 감지하고, 자동으로 교체(Terminate & Provision)해 준다. 

#### Disruption 컨트롤러의 역할 ####
Disruption 컨트롤러 클러스터의 상태를 계속 감지하면서 "지금 이 노드를 삭제하거나 교체해야 하는가?"를 결정하는데, 다음 3가지의 방식이 존재한다. 
* Drift (드리프트): 사용자가 NodePool 설정을 변경하여 실제 노드 사양과 설정이 달라졌을 때.
* Consolidation (최적화): 노드가 비어 있거나 더 저렴한 인스턴스로 합칠 수 있을 때.
* Repair (복구 - Node Monitoring Agent 연동): Node Monitoring Agent가 노드 장애를 보고하여 NodeCondition이 나빠졌을 때.

#### Node Repair 기능 활성화 ####
카펜터 버전 v1.1.0 이상부터는 기본적으로 Node Auto Repair 기능이 포함되어져 있어 별도의 설정이 필요하지 않다. 
다만, 카펜터가 노드를 너무 공격적으로 노드를 교체하지 않도록 NodePool 리소스에 Disruption Budgets를 설정하는 것을 고려해 볼 수도 있다.
```
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    budgets:
      - nodes: "10%" # 한 번에 교체될 수 있는 최대 노드 비율
```

## 장애 시뮬레이션 ##

* 3. Chaos Mesh 등 카오스 엔지니어링 도구 사용
파드 수준에서 더 정교한 장애(네트워크 지연, 특정 컨테이너 강제 종료 등)를 시뮬레이션하고 싶다면 전용 도구를 사용하는 것이 좋습니다. 
Chaos Mesh: PodChaos를 생성하여 특정 파드를 주기적으로 죽이거나(PodKill), 컨테이너를 강제 종료(ContainerKill)하여 EKS의 자가 치유 성능을 측정할 수 있습니다.

* 4. GPU 스트레스 테스트 (과부하 유도)
장애가 아닌 '성능 저하' 및 '과열로 인한 노드 이탈' 시나리오를 테스트하고 싶다면 파드 내에서 연산 부하를 최대치로 올릴 수 있습니다.
```
# 파드 내에서 무한 루프로 GPU 연산 수행 (유틸리티 상승)
bash -c "for i in {1..100}; do nvidia-smi; sleep 0.1; done"
```
파드 내부에서 "GPU 고장" 명령을 날리는 대신, NVIDIA_VISIBLE_DEVICES=none을 설정하여 앱의 오류 처리를 테스트하거나, Chaos Mesh 같은 도구로 파드 자체를 강제 종료하는 방식이 실제 EKS 복구 테스트에 더 적합합니다.


#### 에이전트 로그 실시간 모니터링 ####
에이전트가 노드의 커널 메시지나 시스템 로그를 제대로 파싱하고 있는지 확인합니다.
```
# stern 설치 시
kubectl stern -n kube-system -l app.kubernetes.io/instance=eks-node-monitoring-agent

# 기본 kubectl 사용 시
kubectl logs -f -n kube-system -l app.kubernetes.io/instance=eks-node-monitoring-agent
```

#### GPU 장애 감지 시뮬레이션 (검증 테스트) ####
NMA는 노드 로그를 모니터링하여 특정 오류 메시지가 발생하면 NodeCondition을 변경합니다. GPU 노드의 경우 NVIDIA 드라이버 오류(XID 메시지 등)를 감지하도록 설계되어 있습니다. 

A. 가상 오류 메시지 주입 (노드 레벨)
노드에 직접 접속하여 에이전트가 감시하는 커널 로그(kmsg)에 강제로 오류 메시지를 써서 감지 여부를 테스트할 수 있습니다. 
```
# 노드 SSH 접속 후 실행 (NVIDIA GPU 오류 메시지 예시)
echo "NVRM: Xid (PCI:0000:00:00): 31, GPU termination" | sudo tee /dev/kmsg
```

B. NodeCondition 변화 확인
메시지 주입 후 에이전트가 이를 감지하면 해당 노드의 상태값이 변경됩니다
```
kubectl describe node <노드명> | grep -A 5 Conditions
```
정상 감지 시: AcceleratedHardwareReady 또는 관련 조건이 False로 변경되거나 특정 오류 테인트(Taint)가 붙는지 확인합니다. 

* 관전 포인트: 오류 주입 후 수 분 내에 노드가 Cordon(스케줄링 중단) 상태가 되고, 새로운 노드가 프로비저닝되는지 확인합니다. 
*  CloudWatch 지표 확인
NMA는 감지된 문제에 대해 CloudWatch 메트릭을 발행할 수 있습니다. CloudWatch 콘솔의 ContainerInsights 네임스페이스에서 관련 성능 지표나 로그가 수집되고 있는지 점검

---
## 프로메테우스 연동 ##
EKS Node Monitoring Agent(NMA)는 내부적으로 GPU 메트릭을 수집하지만, 기본적으로는 Prometheus가 아닌 Amazon CloudWatch Container Insights로 데이터를 보내도록 설계되어 있습니다. 그라파나(Grafana) 연동을 위해 프로메테우스(Prometheus)가 NMA의 데이터를 읽어오게 하려면, NMA가 노출하는 메트릭 엔드포인트를 프로메테우스 스크랩(Scrape) 대상에 추가해야 합니다.

#### 1. NMA 메트릭 엔드포인트 확인 ####
NMA는 각 노드에서 DaemonSet 형태로 실행되며, 일반적으로 다음 포트를 통해 메트릭을 노출합니다: 
* 포트: 8080 (또는 설정에 따라 다를 수 있음)
* 경로: /metrics 

#### 2. Prometheus 설정 (Scrape Config) #### 
프로메테우스 설정(prometheus.yml) 또는 ServiceMonitor(Prometheus Operator 사용 시)에 NMA를 타겟으로 추가합니다. 

방법 A: prometheus.yml에 직접 추가
프로메테우스 설정 파일의 scrape_configs 섹션에 아래 내용을 추가합니다. 
```
scrape_configs:
  - job_name: 'eks-node-monitoring-agent'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      # NMA 포드가 있는 kube-system 네임스페이스와 앱 라벨 필터링
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_pod_label_app_kubernetes_io_name]
        action: keep
        regex: kube-system;eks-node-monitoring-agent
      # 포트 8080 사용 설정
      - source_labels: [__address__]
        action: replace
        regex: ([^:]+)(?::\d+)?
        replacement: $1:8080
        target_label: __address__

```

방법 B: ServiceMonitor 사용 (Prometheus Operator 환경) 
가장 권장되는 현대적인 방식입니다. 아래 YAML을 적용하여 프로메테우스가 자동으로 NMA를 발견하게 합니다.
```
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: eks-node-monitoring-agent
  namespace: kube-system # NMA가 설치된 네임스페이스
  labels:
    release: prometheus # 사용하는 프로메테우스 릴리즈 라벨에 맞게 수정
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: eks-node-monitoring-agent
  endpoints:
  - port: metrics # NMA 서비스/포드에 정의된 포트 이름
    interval: 30s
    path: /metrics

```



## 레퍼런스 ##
* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/
* https://dev.to/aws-builders/node-health-monitoring-and-auto-repair-for-amazon-eks-3eja
