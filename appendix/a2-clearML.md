ClearML은 인프라 정보(GPU/EFA)부터 모델 실험 데이터까지 훈련의 모든 과정을 자동으로 추적·관리하고 실행 환경까지 제어할 수 있는 올인원(All-in-One) 오픈소스 MLOps 플랫폼입니다

```
helm repo add clearml https://clearml.github.io/clearml-helm-charts
helm repo update
```

ClearML은 실험 관리(Experiment Tracking)뿐만 아니라 MLOps 인프라 제어 기능을 포함하고 있어, 다른 도구보다 하드웨어 레벨의 정보를 더 깊게 자동 기록합니다. 
* 인프라 자동 기록: 별도의 코드 없이도 훈련이 수행된 인스턴스 정보, CPU/GPU 모델 및 개수, 메모리 사양을 자동으로 감지하여 저장합니다.
* 하드웨어 메트릭 (GPU/EFA): GPU 사용량뿐만 아니라 네트워크(Network) 사용량을 실시간으로 로깅하므로, EFA 활성화 여부와 데이터 전송 효율을 시각적으로 확인할 수 있습니다.
* 상세 훈련 이력: 에폭(Epoch), 모델 하이퍼파라미터, 소요 시간, 콘솔 출력 로그(stdout/stderr)를 모두 보관합니다.
* 완전 오픈 소스: 서버를 직접 구축(Self-hosted)할 수 있는 무료 오픈 소스 버전이 제공됩니다
