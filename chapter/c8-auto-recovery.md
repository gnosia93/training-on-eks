## 매니지드 노드 자동복구 ##
관리형 노드 그룹에서는 오토 모드와 달리 사용자가 직접 기능을 활성화해야 합니다. 
NMA 에이전트가 노드의 커널, 네트워크, 스토리지, GPU 상태를 모니터링하다가 문제가 발견되면 Kubernetes NodeCondition을 업데이트합니다.
상태가 나빠진 노드를 발견하면 EKS 컨트롤 플래인이 해당 노드를 자동으로 격리(Cordon) 및 비우기(Drain)한 후, 새로운 인스턴스로 교체합니다. 

#### 1. Node Monitoring Agent 설치 ####
클러스터 설정에서 eks-node-monitoring-agent 애드온을 추가하여 설치해야 합니다. 이 에이전트가 노드의 로그를 분석하여 장애를 감지하는 역할을 합니다.
```
export CLUSTER_NAME="training-on-eks"
export K8S_VERSION="1.34"

NMA_VERSION=$(aws eks describe-addon-versions --kubernetes-version ${K8S_VERSION} --addon-name eks-node-monitoring-agent)
echo "Node Monitoring Agent Version: "${NMA_VERSION}

aws eks create-addon \
    --cluster-name ${CLUSTER_NAME} \
    --addon-name eks-node-monitoring-agent \
    --addon-version ${NMA_VERSION}  # 2025년 최신 버전 확인 필요

# 에이전트 포드 확인
kubectl get pods -n kube-system | grep node-monitoring-agent
```

#### 2. 노드 그룹의 'Node Repair' 활성화 ####
에이전트만 설치한다고 복구가 자동으로 수행되지 않습니다. 매니지드 노드 그룹 설정에서 node-repair 기능을 Enabled로 변경해야 합니다.
```
aws eks update-nodegroup-config --cluster-name <클러스터명> \
  --nodegroup-name <노드그룹명> \
  --node-repair-config enabled=true
```

## 카펜터 노드 자동복구 ##

카펜터는 기본적으로 NotReady 상태인 노드를 감지하여 교체한다. Node Monitoring Agent를 설치하면 NotReady가 되기 전의 미세한 하드웨어 장애(GPU 오류, 디스크 읽기 전용 등) 단계에서 더 빠르게 대응할 수 있다. 매니지드 노드 그룹에서는 AWS 백엔드가 복구를 수행하지만, 카펜터 환경에서는 Node Monitoring Agent 가 문제를 감지하고 카펜터가 노드를 교체하는 협업 구조로 작동한다. 
Node Monitoring Agent는 노드 장애를 발견되면, 해당 노드의 NodeCondition을 업데이트 한다. (예: StorageReady=False).
카펜터의 Disruption(중단) 컨트롤러가 이를 감지하고, 자동으로 교체(Terminate & Provision)해 준다. 

#### Disruption 컨트롤러의 역할 ###
Disruption 컨트롤러 클러스터의 상태를 계속 감지하면서 "지금 이 노드를 삭제하거나 교체해야 하는가?"를 결정하는데, 다음 3가지의 방식이 존재한다. 
* Drift (드리프트): 사용자가 NodePool 설정을 변경하여 실제 노드 사양과 설정이 달라졌을 때.
* Consolidation (최적화): 노드가 비어 있거나 더 저렴한 인스턴스로 합칠 수 있을 때.
* Repair (복구 - Node Monitoring Agent 연동): Node Monitoring Agent가 노드 장애를 보고하여 NodeCondition이 나빠졌을 때.

### NodeRepair 기능 활성화 ###
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


## 레퍼런스 ##
* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/

