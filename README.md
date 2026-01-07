# training-on-eks

![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/terraform-all.png)

_본 워크샵은 EKS 및 Kubeflow Training Operator를 활용하여 대규모 모델의 분산 훈련과 모니터링 핵심 기술을 다룹니다. GPU 인스턴스는 Karpenter NodePool을 통해 동적으로 프로비저닝하며, 모니터링은 Prometheus/Grafana 스택을 활용합니다. 인프라 구성에는 Terraform(VPC)과 eksctl(EKS 클러스터)을 사용하며, 원활한 실습 진행을 위해 컨테이너와 EKS에 대한 기초 지식이 필요합니다. 분산 훈련 전략으로는 PyTorch DDP/FSDP 및 DeepSpeed 프레임워크를 사용하고 있으며, TP/PP를 지원하는 NVIDIA Megatron-LM의 고급 병렬화 기법는 포함되어 있지 않습니다 (향후 업데이트 예정)._  

* 사전 준비 사항 - Hugging Face Llama 3-8B 모델을 다운로드 할수 있는 토근 (HF_TOKEN) 

### _Topics_ ###

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* C5. Observability 
   * [GPU 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    
   * [EFA 네트워크 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-network-efa-mon.md) 
   * [훈련 로그 수집](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring.md)
   * [GPU / 네트워크 / 노드 스로틀링 식별](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-detect-perf-drop.md)
  
     
* [C6. FSDP로 LLM 훈련하기 (싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)    

* C7. 분산 훈련 최적화
   - [EFA 사용하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-otimization-efa.md)
   - [Placement Group 설정](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-optimization-placement-group.md) 
   - [GPU 토폴로지 최적화](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-optimization-intra-topology.md)
   - [NCCL 최적화](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-optimization-nccl.md)
      
* C8. 훈련 스케줄링 및 복원력 
   - [Kueue 갱 스케줄링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-kueue.md)
   - [노드 / GPU 스케줄링 배제](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-schedule-evit.md)      
   - [장애노드 식별 및 자동 복구 (NPD)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-auto-recovery.md)
   - [훈련 자동 복구 / 재실행](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-trainjob.md) 

* [C9. 커스텀 컨테이너 이미지 빌드](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c9-custom-container.md)
  
* [C10. 체크포인트 저장하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c10-saving-checkpoint.md)

* [C11. DeepSpeed 분산 훈련 (멀티 GPU 노드)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c11-deepspeed.md)

### _Appendix_ ###

* [A1. 트러블 슈팅](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a1-karpenter-message.md)
* [A2. EC2 분산 훈련](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a2-ec2-nvlink.md)
* [A3. 토폴리지별 훈련성능](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a3.training-perf-topology.md)
* [A4. CPU 분산훈련](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a4.training-on-cpu.md)

### _Revision History_ ###
* 2026-01-07 First Released — EKS 및 Kubeflow 기반 분산 훈련 가이드 작성
