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


### 핵심 주의사항 ###
* Karpenter 연동: 에이전트만 설치한다고 복구가 완료되지 않습니다. Karpenter v1.1.0 이상이 설치되어 있어야 에이전트가 보낸 "아파요" 신호를 보고 Karpenter가 노드를 교체합니다.
* 권한(IAM): 에이전트가 장애 정보를 AWS로 보고할 수 있도록 노드 IAM 역할(Node Role)에 관련 권한(AmazonEKSWorkerNodePolicy 등)이 포함되어 있어야 합니다.

## 레퍼런스 ##
* https://github.com/aws/eks-node-monitoring-agent/tree/main/charts/eks-node-monitoring-agent
* https://aws.amazon.com/ko/blogs/containers/amazon-eks-introduces-node-monitoring-and-auto-repair-capabilities/

