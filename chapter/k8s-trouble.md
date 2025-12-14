## 노드 조인 되지 않음 ##

### 1단계: 노드의 IAM 역할 및 권한 확인 ####
Karpenter 설정 시 지정했던 IAM 역할에 필수 정책이 연결되어 있는지 확인합니다.
```
aws iam list-attached-role-policies --role-name KarpenterNodeRole-training-on-eks
```

* 필수 정책:
  - AmazonEKSWorkerNodePolicy
  - AmazonEC2ContainerRegistryReadOnly
만약 정책이 누락되었다면, AWS CLI로 추가해 줍니다.
```
aws iam attach-role-policy --role-name KarpenterNodeRole-training-on-eks --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
aws iam attach-role-policy --role-name KarpenterNodeRole-training-on-eks --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
```

### 2단계: aws-auth ConfigMap 확인 (가장 중요) ###
새 노드가 클러스터에 합류하려면, 노드의 IAM 역할이 aws-auth ConfigMap에 매핑되어야 합니다. Karpenter는 이 작업을 자동으로 처리하지만, 때때로 문제가 발생할 수 있습니다.
```
kubectl edit configmap aws-auth -n kube-system
```
```
mapRoles 섹션에 KarpenterNodeRole에 대한 항목이 있는지 확인하십시오. 다음 예시와 유사해야 합니다.
yaml
data:
  mapRoles: |
    - rolearn: arn:aws:iam::<ACCOUNT_ID>:role/KarpenterNodeRole-<YOUR_CLUSTER_NAME>
      username: system:node:{{EC2PrivateDNSName}}
      groups:
        - system:bootstrappers
        - system:nodes
```

### 3단계: EC2 시스템 로그 확인 ###
AWS 콘솔에서 실행 중인 GPU 인스턴스를 선택하고, **작업(Actions) -> 모니터링 및 문제 해결(Monitor and troubleshoot) -> 시스템 로그 가져오기(Get system log)**를 클릭합니다.

로그 하단에서 kubelet이 성공적으로 시작되었는지, EKS 엔드포인트 연결에 실패했다는 오류 메시지가 있는지 확인합니다.

이 문제들을 해결한 후, 해당 EC2 인스턴스를 재부팅하거나 Karpenter가 새 인스턴스를 띄우도록 유도하면 노드가 정상적으로 합류할 것입니다.

