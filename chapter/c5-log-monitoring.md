## grafana loki ## 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki.webp)

Grafana Loki는 Grafana Labs에서 개발한 오픈소스 로그 집계 시스템으로, 대규모 시스템의 로그를 효율적이고 저렴하게 저장하고 검색하기 위해 설계되었다. 흔히 "로그판 프로메테우스(Prometheus)"라고도 불리며, 현대적인 클라우드 환경(Kubernetes 등)에 최적화되어 있다. 분산 학습 환경에서 Grafana Loki를 사용할 때의 핵심 장점은 다음과 같다. 
* 실시간 통합 모니터링: 여러 노드에 흩어진 파드 로그를 하나의 타임라인으로 병합하여 실시간(Live)으로 확인하고, 키워드 필터링을 통해 대량의 로그 속에서도 Loss나 에러를 즉각 추적.
* 강력한 디버깅 지원: 에러 발생 시점 앞뒤의 로그를 보여주는 'Context 조회' 기능을 통해 분산 학습 중 발생하는 복잡한 데드락이나 통신 오류의 원인을 쉽고 빠르게 파악.
* 높은 효율성과 가성비: S3와 같은 객체 스토리지 연동으로 대용량 학습 로그를 저렴하게 보관하며, 인덱싱 최적화를 통해 전문 검색 엔진 대비 리소스를 적게 사용하면서도 현대적인 분석 환경을 제공.
  

## 설치 순서 ##
* [Step 1. Loki 백엔드 설정](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring-loki.md) 

* [Step 2. Alloy 로그 수집기 설정](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring-alloy.md) 

* [Step 3. Loki 그라파나 대시보드 설정](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring-loki-grafana.md) 


## 레퍼런스 ##

* [Loki Architecture: A Log Aggregation Journey with Grafana](https://sujayks007.medium.com/loki-architecture-a-log-aggregation-journey-with-grafana-bde6d9df6a04)



