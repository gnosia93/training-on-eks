### 1. 필수 설정 2단계 ###
관리형 노드 그룹에서는 오토 모드와 달리 사용자가 직접 기능을 활성화해야 합니다. 

#### Node Monitoring Agent(NMA) 설치: ####
EKS Add-on으로 제공됩니다.
클러스터 설정에서 eks-node-monitoring-agent 애드온을 추가하여 설치해야 합니다. 이 에이전트가 노드의 로그를 분석하여 장애를 감지하는 역할을 합니다.

#### 노드 그룹의 'Node Repair' 활성화: ####
에이전트만 설치한다고 복구가 자동으로 수행되지 않습니다. 매니지드 노드 그룹 설정에서 node-repair 기능을 Enabled로 변경해야 합니다.
```
aws eks update-nodegroup-config --cluster-name <클러스터명> \
  --nodegroup-name <노드그룹명> \
  --node-repair-config enabled=true
```

### 2. 작동 원리 ###
* 감지: NMA 에이전트가 노드의 커널, 네트워크, 스토리지 상태를 모니터링하다가 문제가 발견되면 Kubernetes NodeCondition을 업데이트합니다.
* 복구: 상태가 나빠진 노드를 발견하면 EKS 제어 평면이 해당 노드를 자동으로 격리(Cordon) 및 비우기(Drain)한 후, 새로운 인스턴스로 교체합니다. 













```
# using the github chart repository
helm repo add eks-node-monitoring-agent https://aws.github.io/eks-node-monitoring-agent
helm install eks-node-monitoring-agent eks-node-monitoring-agent/eks-node-monitoring-agent --namespace kube-system
```

```
helm uninstall eks-node-monitoring-agent --namespace kube-system
```

* 
```
aws eks create-addon --cluster-name <name of the EKS cluster> --addon-name eks-node-monitoring-agent
```
```
aws eks create-addon \
    --cluster-name <클러스터_이름> \
    --addon-name eks-node-monitoring-agent \
    --addon-version v1.0.0-eksbuild.1  # 2025년 최신 버전 확인 필요

# 에이전트 포드 확인
kubectl get pods -n kube-system | grep node-monitoring-agent

. 핵심 주의사항
Karpenter 연동: 에이전트만 설치한다고 복구가 완료되지 않습니다. Karpenter v1.1.0 이상이 설치되어 있어야 에이전트가 보낸 "아파요" 신호를 보고 Karpenter가 노드를 교체합니다.
권한(IAM): 에이전트가 장애 정보를 AWS로 보고할 수 있도록 노드 IAM 역할(Node Role)에 관련 권한(AmazonEKSWorkerNodePolicy 등)이 포함되어 있어야 합니다.
요약하자면: 2025년에는 EKS Managed Add-on 목록에서 클릭 한 번으로 설치하는 것이 표준이며, 가장 안전한 방법입니다.
```

## 레퍼런스 ##
* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/

---
2025년 기준 Amazon EKS의 노드 자동 복구(Node Auto-repair)는 한층 정교해진 모니터링 에이전트와 자동화된 교체 로직을 통해 클러스터의 가용성을 유지합니다. 상세 작동 원리와 주요 특징은 다음과 같습니다.

### 1. 작동 원리 및 프로세스 ###
노드 자동 복구는 '감지 → 진단 → 복구'의 3단계로 진행됩니다. 
#### 감지 (Detection): #### 
노드에 설치된 Node Monitoring Agent가 시스템 로그와 메트릭을 실시간으로 분석합니다.
* Kubelet 응답 없음: 노드와 마스터 간 통신이 끊긴 경우.
* 리소스 고갈: 디스크 가득 참(DiskPressure)이나 메모리 부족(MemoryPressure).
* 네트워크/스토리지 이슈: NetworkingReady, StorageReady 등의 조건이 비정상인 경우.
* GPU 장애: 인공지능(AI) 워크로드에 중요한 GPU 하드웨어 결함.

#### 진단 및 격리 (Cordoning): ####
문제가 확인되면 해당 노드를 즉시 Cordon(신규 포드 스케줄링 중단) 상태로 변경하여 추가 피해를 막습니다.

#### 복구 (Repair): ####
문제가 지속될 경우 AWS가 자동으로 해당 EC2 인스턴스를 종료하고 동일한 설정의 새 인스턴스를 실행합니다. 

### 2. 주요 복구 기준 및 시간 (2025년 기준) ###
* 일반 노드: 상태 이상 감지 후 최대 30분 이내에 복구 조치(재부팅 또는 교체)가 완료됩니다.
* GPU 노드: 고가의 리소스를 사용하는 AI/ML 환경을 위해 더 빠른 10분 이내 복구를 보장합니다.
* 복구 중단 조건: 안전을 위해 다음과 같은 상황에서는 복구가 일시 중지됩니다.
노드 그룹 내 노드가 5개 이상이고, 20% 이상의 노드가 동시에 실패한 경우 (대규모 장애 예방).
가용 영역(AZ) 장애로 인해 'Zonal Shift'가 활성화된 경우. 

### 3. 유형별 자동 복구 방식 ###
사용하는 컴퓨팅 방식에 따라 복구 매커니즘이 다릅니다.
* EKS Auto Mode	기본 활성화. AWS가 인프라 전체를 관리하며 노드 모니터링 에이전트와 복구 기능이 별도 설정 없이 통합 제공됩니다.
* Managed Node Groups	선택 활성화. API나 콘솔에서 Node Auto Repair 옵션을 켜야 하며, AWS가 관리하는 Auto Scaling Group(ASG)과 연동되어 복구됩니다.
* Karpenter	2025년 현재 Alpha 기능(v1.10+)으로 제공되며, 건강하지 않은 노드를 자동으로 감지하여 교체하는 기능을 포함하고 있습니다.

### 4. 관리자 고려 사항 ###
* Pod Disruption Budgets(PDB): 자동 복구 시에도 사용자가 설정한 PDB를 준수합니다. 서비스 가용성을 보장하기 위해 앱마다 최소 실행 단위(PDB)를 설정해두는 것이 권장됩니다.
* 로그 및 감사: 모든 자동 복구 조치는 AWS CloudTrail이나 EKS 이벤트 로그에 기록되어 사후 분석이 가능합니다. 
이 기능은 특히 관리자가 직접 인프라를 일일이 확인하기 어려운 대규모 클러스터나 GPU 기반의 AI 모델 학습 환경에서 운영 부담을 획기적으로 줄여줍니다. 

## 레퍼런스 ##

* https://docs.aws.amazon.com/eks/latest/userguide/node-health.html
