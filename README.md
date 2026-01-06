# training-on-eks

![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/workshop-archi.png)


### _Topics_ ###

* [C1. VPC 생성하기](https://github.com/gnosia93/training-on-eks/tree/main/tf)

* [C2. EKS 클러스터 생성하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c2-provison-eks.md) 

* [C3. GPU 노드 준비하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c3-gpu-node.md)

* [C4. Pytorch Training Operator 설치](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c4-training-operator.md)

* C5. Observability 
   * [GPU 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-gpu-monitoring.md)    
   * [EFA 네트워크 모니터링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-network-efa-mon.md) 
   * [훈련 로그 수집](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c5-log-monitoring.md)
   * [연산 속도가 느린 GPU / 네트워크 카드 / 노드 식별](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-detect-perf-drop.md)
  
     
* [C6. FSDP로 LLM 훈련하기(싱글노드 멀티 GPU)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c6-fsdp.md)    

* C7. 분산 훈련 최적화
   - [EFA 사용하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-otimization-efa.md)
   - [Placement Group 설정](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-optimization-placement-group.md) 
   - [GPU 토폴로지 최적화](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-optimization-intra-topology.md)
   - [NCCL 최적화](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c7-training-optimization-nccl.md)
      
* C8. 훈련 스케줄링 및 복원력 
   - [Kueue 갱 스케줄링](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-kueue.md)
   - [노드/GPU 스케줄링 배제](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-schedule-evit.md)      
   - [장애노드 식별 및 자동 복구 (NPD)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-auto-recovery.md)
   - [훈련 자동 복구/재실행](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c8-trainjob.md) 

* [C9. 커스텀 컨테이너 이미지 빌드](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c9-custom-container.md)
  
* [C10. 체크포인트 저장하기](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c10-saving-checkpoint.md)

* [C11. DeepSpeed 분산 훈련(멀티 GPU 노드)](https://github.com/gnosia93/training-on-eks/blob/main/chapter/c11-deepspeed.md)

### _Appendix_ ###

* [A1. 트러블 슈팅](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a1-karpenter-message.md)
* [A2. EC2 분산 훈련](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a2-ec2-nvlink.md)
* [A3. 토폴리지별 훈련성능](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a3.training-perf-topology.md)
* [A4. CPU 분산훈련](https://github.com/gnosia93/training-on-eks/blob/main/appendix/a4.training-on-cpu.md)

### _Revision History_ ###
* 2026-01-07 First Released
