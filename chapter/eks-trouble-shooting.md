## EKS 보안 그룹 ##

eksctl 로 클러스터를 생성하는 경우 EKS 클러스터 보안 그룹에 karpenter.sh/discovery 와 kubernetes.io/cluster/${CLUSTER_NAME} 모두 자동으로 기록되어 진다.
카펜터에서 시큐리티 그룹 참조시 둘 중 하나를 사용하면 된다. 

### 1. eksctl에서 아무 설정도 안 하면 생기는 일 ###
* 클러스터 보안 그룹(Shared SG) 자동 생성: eksctl이 443, 10250 및 노드 간 통신 규칙이 완벽히 세팅된 보안 그룹을 만듭니다 [1, 2].
* 태그 자동 부여: 이 보안 그룹에는 kubernetes.io/cluster/<name>: owned 태그가 자동으로 붙습니다 [2].
* 노드 그룹에 자동 적용: 작성하신 managedNodeGroups에도 withShared: true가 기본으로 적용되어, 노드들이 이 보안 그룹을 입고 태어납니다 [3].

### 2. 카펜터(Karpenter) 설정과의 연결 ###
이제 카펜터 설정(EC2NodeClass)에서 eksctl이만든 그 태그만 바라보게 하면 모든 것이 끝납니다.
```
# EC2NodeClass (카펜터 설정)
spec:
  securityGroupSelectorTerms:
    - tags:
        kubernetes.io/cluster/${CLUSTER_NAME}: "owned"
```

### 3. 결론: "최소한의 설정이 최고의 설정" ###
cluster.yaml에서 보안 그룹 관련 코드를 추가하지 않는 것은 "AWS와 eksctl이 권장하는 표준 보안 아키텍처를 그대로 따르겠다"는 의미입니다.
* 장점: 설정 오류로 인해 노드가 NotReady에 빠질 위험이 거의 없습니다.
* 추가 작업: 나중에 특정 DB나 외부 서비스에 접속해야 할 때만 별도의 보안 그룹을 만들어 attachIDs로 추가해 주시면 됩니다.
결론적으로, 보안 그룹 부분은 아무것도 건드리지 않고 owned 태그로 카펜터를 연결하는 것이 2025년 현재 가장 깔끔한 베스트 프랙티스입니다.

### 4. 부연설명 ###

혹시나 클러스터 시큐리티 그룹에 karpenter.sh/discovery 태깅이 없는 경우 아래 스크립트를 돌리면 된다.

* karpenter.sh/discovery 태깅
클러스터 생성시 만들어진 시큐리티 그룹에 karpenter.sh/discovery={cluster name} 로 태깅한다.
카펜터가 신규 노드를 프로비저닝 하면, 이 값으로 태킹된 시큐리티 그룹을 찾아 신규 노드에 할당하게 된다.  
노드가 위치하게 되는 서브넷 역시 동일 매커니즘으로 동작하는데, 테라폼에서 이미 karpenter.sh/discovery={cluster name} 태깅을 완료하였다. 
```
NODEGROUP=$(aws eks list-nodegroups --cluster-name "${CLUSTER_NAME}" \
    --query 'nodegroups[0]' --output text)

LAUNCH_TEMPLATE=$(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name "${NODEGROUP}" --query 'nodegroup.launchTemplate.{id:id,version:version}' \
    --output text | tr -s "\t" ",")

SECURITY_GROUPS=$(aws eks describe-cluster \
    --name "${CLUSTER_NAME}" --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

SECURITY_GROUPS="$(aws ec2 describe-launch-template-versions \
    --launch-template-id "${LAUNCH_TEMPLATE%,*}" --versions "${LAUNCH_TEMPLATE#*,}" \
    --query 'LaunchTemplateVersions[0].LaunchTemplateData.[NetworkInterfaces[0].Groups||SecurityGroupIds]' \
    --output text)"

aws ec2 create-tags \
    --tags "Key=karpenter.sh/discovery,Value=${CLUSTER_NAME}" \
    --resources "${SECURITY_GROUPS}"
```
```



## 로드밸런서 비정상적인 동작 - Timeout / 작동은 하나 Headlth 상태 Not Applicable ##
#### 현상 ####
* OIDC 만 설정된 클러스터에서 로드 밸런서 타입의 서비스가 생성된 후, 상당히 오랜 시간이 흘러야 CLB 의 타겟노드들이 In-Service 상태로 바뀜, 하지만 Health status description 칼럼의 값은 Not applicable 상태, 즉 헬스 체킹이 실패하고 있는 상태로 유지됨.. 하지만 서비스는 가능한 상태..
* In-tree 방식(현재 작동 중): 이 방식은 로드밸런서를 만들 때 "워커 노드의 IAM 역할(Node Role)"을 가로채서 사용합니다. Pod Identity 설정 여부와 상관없이 노드 자체에 권한이 있으면 작동합하는중. Deprecated 예정.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/clb-target.png)
* POD Identity 방식에서는 아예 동작하지 않는다. 아래 해결책으로 해결해야 한다.
  
#### 해결책 (2025년 권장 아키텍처) ####
Pod Identity 모드에서 정상적으로 로드밸런서를 쓰시려면 아래 단계를 밟으세요.
* 애드온 설치: eks-pod-identity-agent 설치.
* 컨트롤러 배포: helm으로 aws-load-balancer-controller 설치.
* 권한 연결: eksctl create podidentityassociation으로 컨트롤러와 IAM Role 연결.
* 서비스 수정: 어노테이션에 aws-load-balancer-type: external 추가 (NLB 생성 유도).
요약하자면: Pod Identity를 쓰면 "노드 권한"을 사용하는 구식 방식이 막히는 것이 정상입니다. 따라서 그 권한을 정당하게 이어받을 전용 컨트롤러를 설치해야만 로드밸런서가 다시 동작하게 됩니다. AWS EKS Pod Identity 가이드를 참고하여 컨트롤러를 추가해 보시기 바랍니다.


## 클러스터 생성시 카펜터 설치 실패 ##
#### 원인 ####
eksctl-KarpenterNodeRole-training-on-eks 지워지지 않고 남아 있어서 발생함

#### 해결방법 ####
```
aws iam list-instance-profiles-for-role \
    --role-name eksctl-KarpenterNodeRole-training-on-eks \
    --query 'InstanceProfiles[*].InstanceProfileName' \
    --output text

aws iam remove-role-from-instance-profile \
    --instance-profile-name training-on-eks_14123795526445225688 \
    --role-name eksctl-KarpenterNodeRole-training-on-eks

aws iam delete-instance-profile --instance-profile-name training-on-eks_14123795526445225688

aws iam delete-role --role-name eksctl-KarpenterNodeRole-training-on-eks
```


## 문제 Node Not Initialized ##
Karpenter가 EC2 인스턴스를 성공적으로 생성하여 kubectl get nodes에는 나타나지만, Not Initialized 상태(또는 NoSchedule 테인트 유지)로 수 분 이상 머무는 이유는 크게 네트워크, 권한, 설정 세 가지 측면에서 발생합니다.
2025년 기준, 특히 EKS 1.34 버전 환경에서 발생할 수 있는 주요 원인들은 다음과 같습니다.

### 1. 네트워크 및 CNI 이슈 (가장 흔함) ###
노드가 생성되어도 내부 통신이 안 되면 초기화가 완료되지 않습니다.
* VPC CNI (aws-node) 할당 실패: 새 노드가 생성되었지만, 해당 서브넷에 사용 가능한 IP가 부족하여 aws-node 파드가 실행되지 못할 때 발생합니다.
* Security Group 차단: Control Plane과 노드 간의 443 또는 10250 포트가 막혀 있으면 kubelet이 API 서버와 통신하지 못해 초기화 단계에서 멈춥니다.
* NAT Gateway/프록시 문제: Private Subnet에 노드가 생성되었으나, 인터넷으로 나가는 경로(NAT Gateway)가 없어 kubelet이 EKS API 서버나 컨테이너 레지스트리에 접속하지 못하는 경우입니다.

### 2. IAM 권한 설정 (EC2 Node Role) ###
Karpenter가 노드를 띄울 때 부여하는 IAM Role에 필수 정책이 누락된 경우입니다.
* 필수 정책 누락: AmazonEKSWorkerNodePolicy, AmazonEKS_CNI_Policy, AmazonEC2ContainerRegistryReadOnly 등의 권한이 노드 역할(Instance Profile)에 포함되어야 합니다.
* Karpenter Controller 권한: Karpenter가 인스턴스를 프로비저닝할 때 필요한 ssm:GetParameter 권한 등이 부족하면 노드가 부팅된 후 설정을 가져오지 못합니다.

### 3. NodePool의 Taints와 Pod의 Tolerations 불일치 ###
* Karpenter 전용 테인트: Karpenter는 노드가 완전히 준비되기 전까지 karpenter.sh/not-ready 또는 node.cloudprovider.kubernetes.io/uninitialized 테인트를 붙입니다.
* 초기화 파드 실행 실패: 노드가 'Ready'가 되기 위해 필요한 kube-proxy나 aws-node 같은 시스템 데몬셋이 특정 테인트 때문에 스케줄링되지 못하면 노드는 영원히 초기화되지 않습니다.

### 4. 사용자 데이터(User Data) 및 부팅 스크립트 오류 ###
* 노드가 부팅될 때 실행되는 스크립트가 실패하면 kubelet이 클러스터에 합류하지 못합니다.
* 잘못된 AMI 사용: EKS 1.34 버전과 호환되지 않는 이전 버전의 AMI를 강제로 지정했을 때 발생합니다.
* UserData 문법 오류: EC2NodeClass에서 커스텀 userData를 사용하는 경우, 스크립트 오류로 인해 kubelet 설정이 완료되지 않을 수 있습니다.

### 5. 가용 영역(AZ) 및 서브넷 태그 문제 ###
Karpenter는 서브넷을 선택할 때 특정 태그를 참조합니다.
* kubernetes.io/cluster/<cluster-name>: shared 또는 owned 태그가 서브넷에 없거나, Karpenter EC2NodeClass의 subnetSelector가 서브넷을 찾지 못해 노드가 기형적으로 생성되는 경우 초기화가 지연될 수 있습니다.
확인 방법

### 해결방법 ###
* 이벤트 확인: kubectl describe node <노드이름>을 통해 Events 섹션의 에러 메시지를 확인하세요.
* 시스템 파드 상태: kubectl get pods -n kube-system -o wide 명령어로 해당 노드에 aws-node와 kube-proxy가 정상적으로 떠 있는지 확인하세요.
Karpenter 로그: kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter를 통해 인스턴스 초기화 과정에서 발생하는 에러를 모니터링하세요.
현재 발생한 "incompatible requirements" 에러는 Karpenter가 Pod를 배치할 적절한 노드 사양을 결정하지 못해 발생하는 것으로, NodePool의 requirements 설정과 Pod의 nodeSelector를 먼저 일치시키면 노드가 정상적으로 초기화 단계를 통과할 것입니다.


## Karpenter/NodePool 에서의 EFA 식별 ##
Karpenter는 다음과 같은 방식으로 노드를 식별하고 플러그인과 연동합니다.
* EC2NodeClass/NodePool 설정: 사용자가 NodePool에서 instance-family: [p5, p4d] 등을 지정하거나 requirements에 efa: true 관련 설정을 넣습니다.
* 프로비저닝: Karpenter가 EFA 지원 인스턴스를 띄우면, 해당 노드에 aws-efa-k8s-device-plugin이 실행됩니다.
* 리소스 보고: 플러그인이 실제 EFA 하드웨어를 찾으면 vpc.amazonaws.com 리소스를 노드에 할당합니다. 문
