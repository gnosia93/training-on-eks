# training-on-eks

<< insert architectur diagram >>

<< insert workshop scope and spec >>

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* C5.Observability 
   * [GPU 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    
   * [훈련 로그 모니터링]
     
* [C6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)    
   - t5-fsdp.py 로 교체 필요

* [C7. 멀티노드 분산 트레이닝 최적화](https://github.com/gnosia93/training-on-eks/blob/main/chapter/7-training-otimization.md) 
   - efa RDMA
   - cillium CNI - https://cilium.io/
   - topoloy aware
      - export NCCL_TOPO_FILE=/path/to/system.xml  
      - NVLINK&Switch / EFA / ENA / PCIe  /w docker
         - PCIe 속도 - https://blog.naver.com/techref/223777086733    
   - placement group / same AZ 
   - ultra cluster and capacity block 

* [C8. 분산 훈련 스케줄링/배제 및 복원력](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-scheduling-resiliency.md)
   - 갱 스케줄링 - Kueue 권장
      - 잡큐 관리 / 스케줄링 우선순위 / 재시작 / 잡 파티션 관리 등등...
   - 스케줄링 배제 
      - 노드, GPU 레벨 배제 (훈련전 또는 훈련중 배제)  
      - 특정 GPU ID 또는 노드 배제 or 제거(재시작시)
   - 복원
      - 컨테이너 / 파드 / 노드 크래쉬
      - 모델 체크 포인팅
   - 연산 속도가 느린 GPU / 네트워크 카드 / 노드 식별 방법???
     
* [C9. 병렬 분산 파일 시스템(Lustre)]
  
* [C10. 훈련 방식에 따른 성능 비교]
   - fsdp
      - nvlink
      - efa + nvlink
      - pci
      - eni
   - megatron
     
* [C11. MLOps /w Airflow]
   - MLflow
   - S3 Data Upload -> trigger LLM training
   - S3 Data Upload -> trigger Spark Curation on AWS Graviton4 -> trigger LLM training
         
* [C12. CPU 분산 훈련]
   
## 레퍼런스 ## 

* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)
