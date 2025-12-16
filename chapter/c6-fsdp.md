## FSDP ##
FSDP는 AI 분산 학습에서 사용되는 Fully Sharded Data Parallel 기술로, 모델의 가중치(파라미터), 그래디언트, 옵티마이저 상태를 여러 GPU에 걸쳐 '분할'하여 저장함으로써 메모리 효율성을 극대화하고 매우 큰 모델을 학습할 수 있게 해줍니다. 기존의 DistributedDataParallel (DDP)와 달리 모델 전체를 각 GPU 마다 복제하지 않고 필요한 부분만 가져와 사용하기 때문에 GPU 메모리 부족 문제를 해결하는 데 효과적이다. Meta(이전 Facebook AI Research, FAIR)의 FairScale 팀에 의해 개발되었습니다. 
DeepSpeed의 영향을 받았으며 PyTorch 프레임워크에 네이티브 기능으로 통합되었다. 텐서 및 파이트 라인 페러랠을 사용하는 NVIDIA Megatron 에 비해서 노드간의 집합 통신량이 많다. 

## 훈련하기 ##
* T5 fsdp ..



