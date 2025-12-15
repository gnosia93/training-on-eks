# training-on-eks

<< insert architectur diagram >>

<< insert workshop scope and spec >>

* [1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* [5. GPU 모니터링 하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)      <-- 작성중

* [6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)

* [7. 분산 훈련 복원력]
   - 갱 스케줄링
   - 파드 크래쉬
   - 노드 크래쉬
   - 모델 체크 포인트
   - 특정 GPU ID 또는 노드 회피 (재시작시)
* [8. 멀티노드 분산 트레이닝] 
   - efa RDMA
   - cillium CNI - https://cilium.io/
   - topoloy aware   - placement / same AZ ... ??
   - ultra cluster and capacity block 
* [9. 병렬 분산 파일 시스템(Lustre)]
* [10. MLOps /w Airflow & MLflow]
   - S3 Data Upload -> trigger LLM training
   - S3 Data Upload -> trigger Spark Curation -> trigger LLM training

## 레퍼런스 ## 

* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)
