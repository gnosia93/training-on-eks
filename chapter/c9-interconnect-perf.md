```
# overlays/custom-url/kustomization.yaml

resources:
- ../../base

# 1. 여기에 파라미터 역할을 할 임시 값을 정의합니다.
configMapGenerator:
- name: job-params
  literals:
  - GIT_URL=https://github.com/gnosia93/training-on-eks
  - NUM_WORKERS=3
  - INSTANCE_TYPE=p4d.24xlarge

# 2. 이 값을 필요한 곳(Master args, Worker args 등)에 뿌려줍니다.
replacements:
- source:
    kind: ConfigMap
    name: job-params
    fieldPath: data.GIT_URL
  targets:
  - select: {kind: PyTorchJob, name: pytorch-dist-job}
    fieldPaths: 
    - spec.pytorchReplicaSpecs.Master.template.spec.containers.0.args.0
    - spec.pytorchReplicaSpecs.Worker.template.spec.containers.0.args.0
    # 주의: 이 경우 args 전체가 GIT_URL로 바뀌므로, args가 단순 문자열일 때 유용합니다.

- source:
    kind: ConfigMap
    name: job-params
    fieldPath: data.NUM_WORKERS
  targets:
  - select: {kind: PyTorchJob, name: pytorch-dist-job}
    fieldPaths: ["spec.pytorchReplicaSpecs.Worker.replicas"]
```

### "파라미터 전달"을 위한 실행 명령어 ###
위와 같이 설정해두면, 이제 외부에서 kustomize edit 명령어로 값을 주입할 수 있습니다.

#### 1. 레플리카 수 파라미터 변경 ####
kustomize edit add configmap-item job-params --from-literal=NUM_WORKERS=5 --append-hash

#### 2. 인스턴스 타입 파라미터 변경 ####
kustomize edit add configmap-item job-params --from-literal=INSTANCE_TYPE=g5.48xlarge --append-hash

#### 3. 빌드 및 배포 ####
kustomize build . | kubectl apply -f -


### 만약 스크립트 내부의 "일부 문자열"만 바꾸고 싶다면? ###
작성하신 args는 긴 셸 스크립트 덩어리입니다. 이 안의 특정 단어만 바꾸는 것은 Kustomize가 매우 서툽니다. 이럴 때는 앞서 말씀드린 envsubst 조합이 정신 건강에 가장 좋습니다.

#### 추천 워크플로우: ####
kustomization.yaml의 패치 문구 안에 ${GIT_URL} 같은 변수를 적어둡니다.
```
export GIT_URL="github.com"
export WORKERS=4

kustomize build . | envsubst | kubectl apply -f -
```

## 인터케넥트 타입별 ## 

분산 학습 및 고성능 컴퓨팅(HPC) 환경에서 NVLINK, EFA, PCIe, ENI 등을 통칭할 때는 목적에 따라 다음과 같은 용어들을 사용합니다.

### 1. 하드웨어적 인터페이스 측면: 상호 연결 기술 (Interconnects) ###
가장 일반적으로 사용되는 기술 용어입니다. 서버 내부 또는 서버 간에 데이터를 주고받기 위한 하드웨어적 통로를 의미합니다.
* Node Interconnect: 서버와 서버 사이의 연결 (EFA, ENI, InfiniBand 등)
* GPU Interconnect: GPU와 GPU 사이의 연결 (NVLink)
* System Interconnect: CPU, GPU, 메모리 간의 연결 (PCIe) 

### 2. 물리적 연결 체계: 네트워크 패브릭 (Network Fabric) ###
단순한 선 연결을 넘어, 수많은 노드가 거대한 그물망처럼 얽혀 고속으로 데이터를 주고받는 전체적인 구조를 부를 때 사용합니다.
* 예: "AWS는 EFA 패브릭을 통해 수천 개의 GPU를 하나로 묶는다."

### 3. 논리적/물리적 구성: 네트워크 토폴로지 (Network Topology) ### 
이러한 장치들이 어떤 모양(링, 트리, 메시 등)으로 연결되어 있는지를 말할 때 사용합니다. 분산 학습 효율을 결정짓는 핵심 요소입니다. 
* 예: "NVLink 토폴로지를 확인하여 올바른 프로세스 매핑을 수행한다."

### 4. 데이터 전송 통로: 입출력 계층 (I/O Hierarchy / I/O Path) ###
데이터가 CPU에서 메모리를 거쳐 GPU나 네트워크 카드로 흘러가는 경로를 통칭할 때 사용합니다


