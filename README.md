# training-on-eks

<< insert architectur diagram >>

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* C5. Observability 
   * [C5-1. GPU 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    
   * [C5-2. 훈련 로그 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring.md)
   * [c5-3. EFA 네트워크 모니터링] - https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-network-monitoring.md
   * [C5-3. 연산 속도가 느린 GPU / 네트워크 카드 / 노드 식별](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-detect-perf-drop.md)
  
     
* [C6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)    

* C7. 멀티노드 분산 훈련 최적화
   - [EFA RDMA](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-otimization-efa.md)
   - topoloy aware - https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-nccl.md
      - export NCCL_TOPO_FILE=/path/to/system.xml  
      - NVLINK&Switch / EFA / ENA / PCIe  /w docker

* [C8. 분산 훈련 스케줄링 및 복원력] 
   - [Kueue 갱 스케줄링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-kueue.md)
   - [노드/GPU 스케줄링 배제](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-schedule-evit.md)      
   - [장애노드 식별 및 자동 복구](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-auto-recovery.md)
   - [트레이닝 작업 복구](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-trainjob.md) 

* [C9. 체크포인트 저장하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c10-lustre-eks.md)

* [c10. 커스텀 컨테이너 이미지]

* [C11. 인터커넥트 성능 비교](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c9-intercon-perf.md)

   
## 레퍼런스 ## 

* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)
