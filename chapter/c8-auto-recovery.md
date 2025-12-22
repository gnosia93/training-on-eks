### 1. 구성하기 ###
관리형 노드 그룹에서는 오토 모드와 달리 사용자가 직접 기능을 활성화해야 합니다. 

#### Node Monitoring Agent 설치 ####
클러스터 설정에서 eks-node-monitoring-agent 애드온을 추가하여 설치해야 합니다. 이 에이전트가 노드의 로그를 분석하여 장애를 감지하는 역할을 합니다.
```
aws eks create-addon \
    --cluster-name <클러스터_이름> \
    --addon-name eks-node-monitoring-agent \
    --addon-version v1.0.0-eksbuild.1  # 2025년 최신 버전 확인 필요

# 에이전트 포드 확인
kubectl get pods -n kube-system | grep node-monitoring-agent
```

#### 노드 그룹의 'Node Repair' 활성화 ####
에이전트만 설치한다고 복구가 자동으로 수행되지 않습니다. 매니지드 노드 그룹 설정에서 node-repair 기능을 Enabled로 변경해야 합니다.
```
aws eks update-nodegroup-config --cluster-name <클러스터명> \
  --nodegroup-name <노드그룹명> \
  --node-repair-config enabled=true
```

### 2. 작동 원리 ###
* 감지: NMA 에이전트가 노드의 커널, 네트워크, 스토리지 상태를 모니터링하다가 문제가 발견되면 Kubernetes NodeCondition을 업데이트합니다.
* 복구: 상태가 나빠진 노드를 발견하면 EKS 제어 평면이 해당 노드를 자동으로 격리(Cordon) 및 비우기(Drain)한 후, 새로운 인스턴스로 교체합니다. 



## 카펜터 ##

매니지드 노드 그룹에서는 AWS 백엔드가 복구를 수행하지만, 카펜터 환경에서는 NMA가 문제를 감지하고 카펜터가 노드를 교체하는 협업 구조로 작동합니다. 
* 감지: NMA가 노드 장애를 발견하면 해당 노드의 NodeCondition을 업데이트합니다 (예: StorageReady=False).
* 복구: 카펜터의 Disruption(중단) 컨트롤러가 상태가 나빠진 노드를 감지하고, 이를 자동으로 교체(Terminate & Provision)합니다. 

### 카펜터 NodeRepair 기능 활성화 ###
카펜터 버전 v1.1.0 이상부터는 NMA가 마킹한 비정상 노드를 자동으로 감지하여 교체하는 Node Auto Repair 기능이 포함되어 있습니다. 이를 위해 카펜터 설정에서 해당 기능 플래그를 확인해야 할 수 있습니다 (보통 기본값이거나 Disruption 설정에 포함됨). 

* NodePool 설정 (선택 사항)
카펜터가 노드를 너무 공격적으로 교체하지 않도록 NodePool 리소스에 Disruption Budgets를 설정하는 것이 좋습니다.

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

### 3. 주의사항 ###
* 버전 확인: 카펜터에서 NMA 기반의 자동 복구 기능을 원활하게 사용하려면 카펜터 v1.1.0 이상의 버전을 사용하는 것이 권장됩니다.
* Health Checks: 카펜터는 기본적으로 NotReady 상태인 노드를 감지하여 교체하려고 시도하지만, NMA를 설치하면 NotReady가 되기 전의 미세한 하드웨어 장애(GPU 오류, 디스크 읽기 전용 등) 단계에서 더 빠르게 대응할 수 있습니다. 
카펜터 사용자라면 NMA 애드온만 설치하면 됩니다. 그러면 NMA가 노드 상태를 업데이트하고, 카펜터가 자신의 메커니즘(Disruption)을 통해 해당 노드를 새 노드로 알아서 갈아끼우게 됩니다.
* 권한(IAM): 에이전트가 장애 정보를 AWS로 보고할 수 있도록 노드 IAM 역할(Node Role)에 관련 권한(AmazonEKSWorkerNodePolicy 등)이 포함되어 있어야 합니다.



## 레퍼런스 ##
* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/

