## 노드 주요 레이블 ##
```
feature.node.kubernetes.io/pci-10de.present=true                           # HW - NVIDIA
feature.node.kubernetes.io/pci-1d0f.present=true                           # HW - EFA
feature.node.kubernetes.io/rdma.available=true                             # HW - RDMA 기능 탑재  
node.kubernetes.io/instance-type=p4d.24xlarge
topology.ebs.csi.aws.com/zone=ap-northeast-2b
topology.k8s.aws/network-node-layer-1=nn-c79422f4e61deb9ca                 # TOR 스위치
topology.k8s.aws/network-node-layer-2=nn-6ff5053c3a4d86db7                 # 어그리게이션 스위치
topology.k8s.aws/network-node-layer-3=nn-dcbcca51aebf4c95b                 # 백본 스위치
topology.k8s.aws/zone-id=apne2-az2
topology.kubernetes.io/region=ap-northeast-2
topology.kubernetes.io/zone=ap-northeast-2b

Capacity:
  nvidia.com/gpu:     8
Allocatable:
  nvidia.com/gpu:     8
System Info:
  Machine ID:                 ec2f7360c2f7b4c41b8304f57e0dee91             	# 동일 하드웨어/VM 이라도 OS 재설치 시 바뀜
  System UUID:                ec2f7360-c2f7-b4c4-1b83-04f57e0dee91          # 노드(호스트)의 하드웨어나 가상 머신(VM)을 전 세계적으로 고유하게 식별하기 위해 부여된 128비트 길이의 식별자
```

### 리눅스 OS 에서 제공하는 ID (Node 식별용) ###

* /etc/machine-id: 운영체제 설치 시 생성되는 OS 레벨의 고유 ID입니다. 시스템 전체에서 가장 흔하게 '시스템 ID'로 사용됩니다.
* /sys/class/dmi/id/product_uuid: 하드웨어(BIOS)에서 제공하는 System UUID입니다. 가상 머신이나 물리 서버 자체를 식별할 때 가장 정확합니다.
* /proc/sys/kernel/random/boot_id: 부팅할 때마다 새로 생성되는 ID입니다. 시스템이 재부팅되었는지 여부를 확인하는 용도로 유용합니다.

## 노드 배제 ##

#### 1. 노드에 "Cordon" 설정 (가장 확실한 방법) ####
재부팅 전에 해당 노드를 미리 제외하고 싶다면, UUID를 확인한 뒤 해당 노드 이름에 uncschedulable 마킹을 하는 것입니다.
```
kubectl cordon <노드이름>
```
* 효과: 노드가 재부팅되어 다시 클러스터에 붙더라도, SchedulingDisabled 상태가 유지되어 새로운 Pod가 배치되지 않습니다.
* 복구: 점검이 끝난 후 kubectl uncordon <노드이름>을 해야 다시 사용 가능합니다.

#### 2. 고유 식별자를 레이블(Label)로 활용 ####
System UUID 값을 노드의 레이블로 등록해두면, 특정 노드를 타겟팅하거나 제외하는 스케줄링이 가능합니다.
* 설정: kubectl label nodes <노드이름> hardware-id=ec2f7360-c2f7-b4c4-1b83-04f57e0dee91
* 활용: Pod를 배포할 때 nodeAffinity를 사용하여 해당 ID를 가진 노드를 피하도록(DoesNotExist) 설정할 수 있습니다.

#### 3. Taints (Node Taints) 사용 ####
특정 노드에 "오염(Taint)"을 표시하여, 일반적인 Pod들이 들어오지 못하게 막을 수 있습니다.
* 명령: kubectl taint nodes <노드이름> maintenance=true:NoSchedule
* 효과: 이 노드는 재부팅 후에도 maintenance=true라는 속성을 가지고 있으므로, 이 Taint를 견딜 수 있는(Toleration) 특별한 Pod 외에는 배치되지 않습니다.


## GPU 배제 ##

#### 1. GPU UUID 확인 ####
인덱스(0, 1, 2...)는 하드웨어 변경 시 바뀔 수 있으므로, 확실한 배제를 위해서는 nvidia-smi 를 이용하여 UUID 를 확인해야 한다. GPU UUID 를 알기위해서는 다음과 같이 두가지 방법이 가능하다.   

* NVIDIA_VISIBLE_DEVICES 환경변수 이용
```
kubectl exec -it llama-3-8b-node-0-3-f86kr -- env | grep NVIDIA_VISIBLE_DEVICES
```
* nvidia-smi 이용
```
nvidia-smi -L 
```

#### 2. ConfigMap 생성 #### 

```
# values.yaml
config:
  name: "nvidia-device-plugin-configs"
  default: "default"
  map:
    # (1) 기본 설정
    default: |-
      version: v1
      flags:
        migStrategy: none
    
    # (2) node1 전용: 2번 GPU 제외 (0, 1, 3번만 기재)
    node1-gpu-skip: |-
      version: v1
      devices:
        all: false
        uuids:
          - "GPU-1111-0000-XXXX"
          - "GPU-1111-1111-XXXX"
          - "GPU-1111-3333-XXXX"

    # (3) node3 전용: 0, 1번 GPU만 사용
    node3-gpu-skip: |-
      version: v1
      devices:
        all: false
        uuids:
          - "GPU-3333-0000-XXXX"
          - "GPU-3333-1111-XXXX"
```

```
# 레포지토리 추가 (이미 되어 있다면 생략)
helm repo add nvdp nvidia.github.io
helm repo update

# 설치 또는 업데이트
helm upgrade -i nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system -f values.yaml
```

#### 3. 노드별 차등 적용 ####
스케줄링 배제해야 하는 GPU 를 가진 노드에 아래와 같이 nvida.com 레이블을 설정한다. 
```
# 형식: kubectl label node <노드명> nvidia.com<설정명>

kubectl label node node1 nvidia.com/device-plugin.config=node1-gpu-skip
kubectl label node node3 nvidia.com/device-plugin.config=node3-gpu-skip
```

```
kubectl get node node1 -o jsonpath='{.metadata.labels["nvidia.com/device-plugin.config"]}'
```


#### 4. 디바이스 플러그인 재시작 ####
설정을 변경한 후에는 nvidia-device-plugin 파드를 재시작해야 변경 사항이 반영된다.

* 전체 재시작
```
kubectl rollout restart daemonset -n kube-system nvidia-device-plugin-daemonset
```
* 특정 노드의 Pod만 재시작
```
kubectl get pods -n kube-system -l component=nvidia-device-plugin -o wide
kubectl delete pod <파드_이름> -n kube-system
```

#### 5. GPU capacity 확인 ####
```
kubectl describe node <노드명> | grep -A 5 Capacity
```




