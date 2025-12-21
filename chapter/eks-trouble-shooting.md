## 클러스터 생성시 카펜터 설치 실패 ##

```
aws iam list-instance-profiles-for-role --role-name eksctl-KarpenterNodeRole-training-on-eks

aws iam remove-role-from-instance-profile \
    --instance-profile-name <확인한_프로파일_이름> \
    --role-name eksctl-KarpenterNodeRole-training-on-eks

aws iam delete-instance-profile --instance-profile-name <확인한_프로파일_이름>

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
