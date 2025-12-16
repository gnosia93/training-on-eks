# training-on-eks

<< insert architectur diagram >>

<< insert workshop scope and spec >>

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* [C5. GPU 모니터링 하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    

* [C6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)    
   - t5-fsdpy.py 로 교체 필요

* [C7. 멀티노드 분산 트레이닝 최적화] 
   - efa RDMA
   - cillium CNI - https://cilium.io/
   - topoloy aware
      - export NCCL_TOPO_FILE=/path/to/system.xml  
      - NVLINK&Switch / EFA / ENA / PCIe  /w docker
         - PCIe 속도 - https://blog.naver.com/techref/223777086733    
   - placement group / same AZ 
   - ultra cluster and capacity block 

* [C8. 분산 훈련 스케줄링 및 복원력]
   - 갱 스케줄링 - https://www.google.com/search?q=gt&sca_esv=0ef5ae5df6bcde53&rlz=1C5GCCM_en&sxsrf=AE3TifMG194JXcpIRk9-V8Gzsr5iirkmpA%3A1765898855627&udm=50&fbs=AIIjpHyDg0Pef0CibV20xjIa-FReso8zz6zeRF3Jry8tnzgEaKdNaiajtzDmrSsC-QKZvOhR4qwrtaWwq1sy0SouoHTcPGCg4N8XIpY7mkKLw6S90w6qE4t-zHB5EXKN5qqc7scrUta3-nwbsHtimb1CgcNzr58jrsxP5NgNIdHn6Gw0A-zP2AIhn3fHRe5-FJ22MHdXUEz2TU3tFwTMQC9jpfKisRPSeWCEcfgjHGMFvM3eFhgtZYU&aep=1&ntc=1&sa=X&ved=2ahUKEwj_0_7vtcKRAxUos1YBHaFOG1cQ2J8OegQIDxAE&biw=1349&bih=702&dpr=2.2&mtid=aXpBaf2ZE4G00-kPlIPZiAQ&mstk=AUtExfCmbSa3ziEBiS1VWiY-RfgCYp_8klYFqKu9nsZmaitoueM4F0OhhLPCQ5taYz-XJOL6RxJih-k2J-YwW6JT0thaHYvTh9nWElLR4CkqfmnoiUxpiuOg9OzFuVDPQnQdLq8P6isEona3tJytb5BeYSn91fod1s_RKSFWCuwhQMqrD7yFSwd4hSlBdbhUBbduHyuZxu4_uIOCV8pIJLgojPUs3be44-m-dGF_U9jxhhCeK6XS2YWu47Vn5iM8wd7U1tcghi7ugBZ0YnMh_LGj_1cu9D8rRj6yONYe5XIDvaqdM1G6Rka_PswDfa673DurkqRB4aNOwgOQAw&csuir=1
      - 잡큐 관리 / 스케줄링 우선순위 / 재시작 / 잡 파티션 관리 등등... ?? <----slurm 은 기본적으로 제공하는데.... 
   - 컨테이너 / 파드 / 노드 크래쉬
   - 모델 체크 포인팅
   - 특정 GPU ID 또는 노드 회피 or 제거(재시작시)

* [C9. 병렬 분산 파일 시스템(Lustre)]
* [C10. MLOps /w Airflow]
   - MLflow
   - S3 Data Upload -> trigger LLM training
   - S3 Data Upload -> trigger Spark Curation on AWS Graviton4 -> trigger LLM training

## 레퍼런스 ## 

* [AI/ML 워크로드용 Amazon EKS 클러스터 구성](https://docs.aws.amazon.com/ko_kr/eks/latest/userguide/ml-cluster-configuration.html)
