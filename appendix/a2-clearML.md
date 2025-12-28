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
```
```
helm install clearml clearml/clearml-server -f values.yaml --namespace clearml --create-namespace

```

```
pip install clearml
clearml-init
```
* 명령어를 실행하면 URL 입력창이 뜹니다. ClearML 웹 UI의 Settings > Workspace > Create new credentials에서 복사한 API 키를 붙여넣으세요.

[코드]
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
