
```
aws eks create-addon \
    --cluster-name <클러스터_이름> \
    --addon-name eks-node-monitoring-agent \
    --addon-version v1.0.0-eksbuild.1  # 2025년 최신 버전 확인 필요

# 에이전트 포드 확인
kubectl get pods -n kube-system | grep node-monitoring-agent

```




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
