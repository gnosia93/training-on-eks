# training-on-eks

<< insert architectur diagram >>

<< insert workshop scope and spec >>

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* C5. Observability 
   * [C5-1. GPU 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    
   
   * [C5-2. 훈련 로그 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring.md)
     
* [C6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)    

* [C7. 멀티노드 분산 훈련 최적화](https://github.com/gnosia93/training-on-eks/blob/main/chapter/7-training-otimization.md) 
   - efa RDMA
   - cillium CNI - https://cilium.io/
   - topoloy aware
      - export NCCL_TOPO_FILE=/path/to/system.xml  
      - NVLINK&Switch / EFA / ENA / PCIe  /w docker
   - placement group / same AZ 
   - ultra cluster and capacity block 

* [C8. 분산 훈련 스케줄링 및 복원력]
   - [Kueue 갱 스케줄링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-kueue.md)

   - [노드/GPU 스케줄링 배제](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-schedule-evit.md)
      
   - 복원
      - 컨테이너 / 파드 / 노드 크래쉬
      - 모델 체크 포인팅
   - [연산 속도가 느린 GPU / 네트워크 카드 / 노드 식별](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-detect-perf-drop.md)
  
* [C9. 인터커넥트 성능 비교](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c9-interconnect-perf.md)
  * 테스트 아키텍처 ( 8GPU 1대 vs 1GPU 8대, 최대 통신량 방식, 트레이닝 시간 측정 )

          
* [C10. 병렬 분산 파일 시스템(Lustre)]
   
* [C11. MLOps /w Airflow]
   - MLflow
   - S3 Data Upload -> trigger LLM training
   - S3 Data Upload -> trigger Spark Curation on AWS Graviton4 -> trigger LLM training
         
* [C12. CPU 분산 훈련]
   
## 레퍼런스 ## 

* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)
