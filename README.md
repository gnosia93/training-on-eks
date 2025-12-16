# training-on-eks

<< insert architectur diagram >>

<< insert workshop scope and spec >>

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* [C5. GPU 모니터링 하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    

* [C6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)       <--- 작성중

* [C7. 분산 훈련 스케줄링 및 복원력]
   - 갱 스케줄링
      - 잡큐 관리 / 스케줄링 우선순위 / 재시작 / 잡 파티션 관리 등등... ?? <----slurm 은 기본적으로 제공하는데.... 
   - 컨테이너 / 파드 / 노드 크래쉬
   - 모델 체크 포인팅
   - 특정 GPU ID 또는 노드 회피 or 제거(재시작시)
* [C8. 멀티노드 분산 트레이닝 최적화] 
   - efa RDMA
   - cillium CNI - https://cilium.io/
   - topoloy aware
      - export NCCL_TOPO_FILE=/path/to/system.xml  
      - NVLINK&Switch / EFA / ENA / PCIe  /w docker
   - placement group / same AZ 
   - ultra cluster and capacity block 
* [C9. 병렬 분산 파일 시스템(Lustre)]
* [C10. MLOps /w Airflow]
   - MLflow
   - S3 Data Upload -> trigger LLM training
   - S3 Data Upload -> trigger Spark Curation -> trigger LLM training

## 레퍼런스 ## 

* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)
