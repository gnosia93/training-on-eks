## 카펜터 로그 확인 ##

![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/karpenter-message-1.png)

```
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter
```

## 오류 메시지 및 현상 ##

### 노드의 잦은 Not Ready 상태로의 변경 ###
훈련 도중 노드가 Not Ready 상태로 변경되면, 해당 노드에서 실행중인 파드는 쿠버네티스로 부터 종료 시그널을 받게 된다. 종료 시그널을 받은 파드는 진행중인 작업을 중단하고 종료하게 되는데, 이경우 NCCL 통신이 broken 되어 전체 작업이 비정상적으로 종료한다. 쿠버네티스에서는 이를 방지하게 위해서 아래와 같이 두가지 설정이 필요하다.

#### 1. 파드 annotation 설정 ####
```
  metadata:
    annotations:
      karpenter.sh/do-not-disrupt: "true"                  # Karpenter의 노드 회수 방지
```

#### 2. 카펜터 Consolidation 정책 조정 ####
```
disruption:
    consolidationPolicy: WhenEmpty                         # 이전 설정값은 WhenEmptyOrUnderutilized / 노드의 잦은 Not Ready 상태로의 변경으로 인해 수정  
    consolidateAfter: 10m
```




### "error":"no instance type which had enough resources and the required offering met the scheduling requirements ###

이 에러는 Kubernetes의 Karpenter 가 Pod 에서 요구한 리소스를 제공할 수 있는 EC2 인스턴스를 찾지 못했을 때 발생한다.
주로 다음의 설정이 충돌할 때 발생하며, 최신 가이드에 따른 해결 방법은 다음과 같다.

#### 1. 리소스 요구량과 인스턴스 사양 불일치 ####
YAML에 정의한 limits 값이 지정한 instance-type의 실제 물리적 사양보다 큰 경우이다.
* 체크: nvidia.com/gpu 나 vpc.amazonaws.com/efa 가 p4d.24xlarge 또는 p5.48xlarge 같은 실제 인스턴스가 제공하는 갯수와 일치하는지 확인.
* 예: p4d.24xlarge는 GPU가 8개인데, 실수로 nvidia.com/gpu 필드에 16개를 요청하면 위 에러가 발생.

#### 2. 가용 영역(AZ) 및 구매 옵션(Spot/On-Demand) 불일치 ####
가장 빈번한 원인으로, 지정한 AZ에 해당 인스턴스 재고가 없는 경우이다.
* 해결: topology.kubernetes.io/zone: ap-northeast-2a 설정을 제거하거나, 여러 AZ를 허용하도록 수정.
* 스팟 사용시: 스팟 인스턴스로 요청했는데 해당 리전에 스팟 물량이 없다면 capacity-type: on-demand로 변경.

#### 3. EFA 장치 요청 오류 ####
YAML에 vpc.amazonaws.com/efa: 8 과 같이 명시했다면, 해당 노드 그룹(또는 Karpenter Provisioner)에 EFA 드라이버와 관련 설정이 되어 있어야 함.
일반 GPU 인스턴스(EFA 미지원)를 쓰면서 EFA 리소스를 요청하면 스케줄링이 불가능.
* 테스트: vpc.amazonaws.com/efa 부분을 주석 처리하고 학습이 시작되는지 확인.

#### 4. Karpenter NodePool/Provisioner 제약 ####
Karpenter를 사용 중이라면, NodePool 설정에서 해당 인스턴스 타입을 허용하고 있는지 확인.
```
kubectl get nodepool -o yaml
```
spec.template.spec.requirements에 node.kubernetes.io/instance-type과 karpenter.sh/capacity-type이 Pod의 nodeSelector와 일치하는지 확인.
