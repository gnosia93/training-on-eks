GPU 분야에서 P2P(Peer-to-Peer) Access란, 한 컴퓨터 안에 장착된 여러 개의 GPU가 CPU나 시스템 메모리(RAM)를 거치지 않고 서로 직접 데이터를 주고받는 기술을 의미합니다.
일반적으로 GPU들끼리 데이터를 교환하려면 GPU A → CPU/RAM → GPU B라는 복잡한 경로를 거쳐야 하지만, P2P가 활성화되면 GPU A ↔ GPU B로 직접 연결됩니다.

1. 주요 장점
지연 시간(Latency) 감소: 데이터가 이동하는 경로가 짧아져 응답 속도가 훨씬 빨라집니다.
대역폭(Bandwidth) 최적화: 시스템 메모리(RAM)와 CPU의 부하를 줄여 전체 연산 효율이 높아집니다.
성능 향상: 대규모 AI 모델 학습(Deep Learning)이나 복잡한 시뮬레이션에서 GPU 간 데이터 동기화 속도가 비약적으로 상승합니다.
2. 구현 방식 (연결 통로)
GPU 간에 직접 대화하기 위해 주로 다음과 같은 물리적 통로를 사용합니다.
NVLink (NVIDIA): NVIDIA 전용 초고속 연결 브릿지로, 일반 PCIe보다 수배 이상 빠릅니다.
PCIe P2P: 별도의 브릿지 없이 마더보드의 PCIe 슬롯 경로를 통해 직접 통신합니다.
3. 왜 지금 필요한가요? (Kubernetes 환경)
질문하신 맥락(cert-manager, Helm 등)으로 보아 쿠버네티스에서 GPU 클러스터를 구축 중이신 것으로 보입니다.
분산 학습: 여러 GPU가 하나의 거대한 모델을 학습할 때 P2P가 꺼져 있으면 GPU 성능이 아무리 좋아도 병목 현상 때문에 속도가 나지 않습니다.
NVIDIA Device Plugin: 쿠버네티스에서 GPU를 사용할 때, 이 P2P 기능을 활성화해야 컨테이너 내부의 프로세스들이 서로의 GPU 메모리에 직접 접근할 수 있습니다.
4. 확인 방법
리눅스 터미널에서 다음 명령어를 입력하면 GPU 간의 연결 상태(P2P 지원 여부)를 확인할 수 있습니다.
bash
nvidia-smi topo -m
코드를 사용할 때는 주의가 필요합니다.

결과 화면에서 OK 또는 NV#라고 표시되면 P2P 통신이 가능한 상태입니다.
SYS라고 표시되면 CPU를 거쳐야 하는 상태(P2P 미지원 또는 비활성)입니다.
요약
P2P Access는 GPU들끼리 "중간 관리자(CPU)" 없이 직통 전화(Direct Line)를 개설하는 것이라고 이해하시면 됩니다. AI 학습이나 고성능 연산 시 필수적인 설정입니다.

---








## ClearML ##
ClearML은 인프라 정보(GPU/EFA)부터 모델 실험 데이터까지 훈련의 모든 과정을 자동으로 추적·관리하고 실행 환경까지 제어할 수 있는 올인원(All-in-One) 오픈소스 MLOps 플랫폼으로 
실험 관리(Experiment Tracking)뿐만 아니라 MLOps 인프라 제어 기능을 포함하고 있어, 다른 도구보다 하드웨어 레벨의 정보를 더 깊게 자동 기록한다. 
* 인프라 자동 기록: 별도의 코드 없이도 훈련이 수행된 인스턴스 정보, CPU/GPU 모델 및 개수, 메모리 사양을 자동으로 감지하여 저장.
* 하드웨어 메트릭 (GPU/EFA): GPU 사용량뿐만 아니라 네트워크(Network) 사용량을 실시간으로 로깅하므로, EFA 활성화 여부와 데이터 전송 효율을 시각적으로 확인.
* 상세 훈련 이력: 에폭(Epoch), 모델 하이퍼파라미터, 소요 시간, 콘솔 출력 로그(stdout/stderr)를 모두 보관.
* 오픈 소스: 서버를 직접 구축(Self-hosted)할 수 있는 무료 오픈 소스 버전이 제공.

### 설치 ###
```
helm repo add clearml https://clearml.github.io/clearml-helm-charts
helm repo update
helm install clearml clearml/clearml-server -f values.yaml --namespace clearml --create-namespace
```

```
pip install clearml
clearml-init
```
* 명령어를 실행하면 URL 입력창이 뜹니다. ClearML 웹 UI의 Settings > Workspace > Create new credentials에서 복사한 API 키를 붙여넣으세요.


[values.yaml]
```
config:
  # 스토리지 설정을 S3로 지정
  files_host: "https://s3.amazonaws.com"
  
  # AWS 자격 증명 (IRSA 사용 시 생략 가능하지만, 명시적 설정이 안정적임)
  aws:
    s3:
      # S3 버킷 이름 및 지역 설정
      bucket: "your-clearml-artifacts-bucket"
      region: "ap-northeast-2"
      # 필요 시 Access Key 입력 (IRSA 미사용 시)
      key: "YOUR_AWS_ACCESS_KEY"
      secret: "YOUR_AWS_SECRET_KEY"

```


## 수집 코드 ##
```
from clearml import Task

# 프로젝트명과 실험 이름을 지정하여 초기화
task = Task.init(project_name="My_Project", task_name="EFA_Training_Run_01")

# 인스턴스 타입이나 EFA 여부를 파라미터로 저장 (나중에 검색 가능)
params = {'instance_type': 'p4d.24xlarge', 'efa': True, 'epochs': 100}
task.connect(params)

# 이후 모델 훈련 코드...
# GPU/CPU/네트워크 트래픽은 ClearML이 자동으로 수집합니다.
```
