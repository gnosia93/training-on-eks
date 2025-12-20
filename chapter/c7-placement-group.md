AWS EC2 배치 그룹(Placement Group)은 EC2 인스턴스들이 AWS 인프라 내 물리적으로 어떻게 배치될지 제어하여, 워크로드의 성능 및 가용성 요구 사항(낮은 지연 시간, 내결함성 등)에 맞춰 최적의 성능을 얻기 위한 기능입니다. 배치 그룹은 인스턴스를 동일한 네트워크 대역에 모으거나(클러스터), 여러 가용 영역(AZ)의 다른 하드웨어에 분산시키는(분산, 파티션) 등의 전략을 제공하며, 이는 고성능 컴퓨팅(HPC), 분산 데이터베이스 등 특정 워크로드에 필수적입니다. 

### 배치그룹의 주요 전략 ###
배치그룹 자체는 무료이다. 

* 클러스터(Cluster)-성능 최적화: 인스턴스를 동일한 가용 영역(AZ) 내에서 서로 가깝게 배치하여, 매우 낮은 네트워크 지연 시간이 필요한 HPC(High-Performance Computing) 작업에 적합합니다. 모든 인스턴스가 동일한 랙(rack)에 위치하여 성능은 높지만, 해당 랙의 장애 시 동시 장애 위험이 있다. AZ 당 사용할 수 있는 인스턴스의 개수에는 제한이 없다. 

* 파티션(Partition)-대규모 가용성 관리: 인스턴스를 논리적 파티션으로 분산시켜, 각 파티션이 독립된 하드웨어(랙)를 공유하지 않도록 합니다. Hadoop, Kafka, Cassandra 같은 대규모 분산 워크로드에서 노드 간 장애 영향을 최소화합니다.

* 분산(Spread)-가용성 극대화: 소규모 인스턴스 그룹을 서로 다른 하드웨어에 분산 배치하여, 상호 연관된 장애(예: 랙 장애)의 위험을 최소화합니다. 가장 높은 수준의 내결함성을 제공하며, 주로 중요한 애플리케이션에 사용됩니다. 

#### 사용 목적 ####
* 고성능 컴퓨팅(HPC): 클러스터 전략을 통해 노드 간 통신 속도를 극대화합니다.
* 대규모 분산 시스템: 파티션/분산 전략으로 단일 하드웨어 장애가 전체 시스템에 미치는 영향을 줄입니다.
* 내결함성 강화: 분산/파티션 전략으로 하드웨어 장애로부터 애플리케이션을 보호합니다. 
* 제한: 배치 그룹 전략에 따라 인스턴스 수, 배치 그룹 유형 등에 제약이 있을 수 있으며, 클러스터 전략은 높은 성능과 함께 장애 위험 증가라는 상반된 특징을 가집니다. 

## 예시 ##
#### ec2 예시 ####
```
aws ec2 create-placement-group \
    --group-name "MyHPCCluster" \
    --strategy cluster

aws ec2 run-instances \
    --image-id ami-xxxxxxxxxxxxxxxxx \
    --count 2 \
    --instance-type c5n.9xlarge \
    --key-name MyKeyPair \
    --placement "GroupName=MyHPCCluster" \
    --subnet-id subnet-xxxxxxxx
```

#### 카펜터 예시 ####
카펜터(Karpenter)를 사용하여 EC2 인스턴스를 프로비저닝할 때 클러스터 배치 그룹(Cluster Placement Group)을 적용하려면, Karpenter의 EC2NodeClass 설정에서 배치 그룹 이름을 지정하면 된다.

#### 1. 사전 준비: 배치 그룹 생성 ####
Karpenter는 현재 배치 그룹을 직접 '생성'해주지는 않으므로, AWS CLI나 Terraform으로 배치 그룹을 미리 만들어 두어야 합니다.
```
# 클러스터 전략의 배치 그룹 생성
aws ec2 create-placement-group --group-name "karpenter-hpc-group" --strategy cluster
```

#### 2. Karpenter 설정 (YAML) ####
Karpenter의 EC2NodeClass 리소스에 placementGroupName 속성을 추가합니다.

```
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: hpc-node-class
spec:
  amiFamily: AL2
  role: "KarpenterNodeRole-MyCluster" # 실제 역할 이름으로 변경
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "my-cluster"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "my-cluster"
  # --- 배치 그룹 설정 부분 ---
  placementGroupName: "karpenter-hpc-group"
```
그 다음, NodePool에서 특정 워크로드(HPC, ML 등)가 이 노드 클래스를 사용하도록 연결합니다.
```
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: hpc-nodepool
spec:
  template:
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: hpc-node-class
      requirements:
        - key: "karpenter.k8s.aws/instance-category"
          operator: In
          values: ["c", "p", "g"] # 클러스터 그룹에 적합한 컴퓨팅/GPU 인스턴스
        - key: "karpenter.sh/capacity-type"
          operator: In
          values: ["on-demand"] # 클러스터 배치는 안정성을 위해 온디맨드 권장
      # 중요: 클러스터 배치 그룹은 단일 AZ 내에서만 작동하므로 하나만 지정
      - key: "topology.kubernetes.io/zone"
        operator: In
        values: ["ap-northeast-2a"]
```

#### Karpenter 사용 시 주의사항 ####
* 용량 부족(ICE) 발생 확률: 클러스터 배치 그룹은 특정 랙(Rack)에 인스턴스를 몰아넣기 때문에, Karpenter가 노드를 추가하려고 할 때 AWS에 해당 랙의 자리가 없으면  InsufficientInstanceCapacity 오류가 발생할 가능성이 매우 높습니다.
* 단일 가용 영역(Single AZ): 클러스터 배치 그룹은 여러 AZ에 걸칠 수 없습니다. 따라서 NodePool 설정에서 반드시 하나의 zone으로 제한해야 합니다.
* 인스턴스 타입 제한: 배치 그룹 내에서 너무 다양한 인스턴스 타입을 섞어 쓰면 배치가 실패할 수 있으므로, 유사한 제품군(예: c5.xlarge, c5.2xlarge)으로 제한하는 것이 좋습니다

