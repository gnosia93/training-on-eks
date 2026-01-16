
### ReduceScatter ###
ReduceScatter는 여러 디바이스(Rank)에 흩어져 있는 데이터를 하나로 Reduce(요약/합산)한 뒤, 그 결과를 다시 각 Rank에 Scatter(분할 배분)하는 집합 통신(Collective Communication) 작업이다.
#### 주요 특징 및 동작 방식 ###
* 작업 흐름: 모든 Rank가 보유한 동일한 크기의 데이터 버퍼에 대해 요소별(element-wise) 연산(예: 합계, 최댓값)을 수행합니다.연산된 전체 결과는 N 개의 균등한 블록으로 나뉘어, 각 Rank i 전체 결과 중 자신의 인덱스에 해당하는 i 번째 블록만 수신하게 됩니다.
* Rank-Device 매핑의 영향: 이 작업은 결과 데이터가 어떤 순서로 어느 디바이스에 저장될지가 Rank 인덱스에 의해 결정됩니다. 따라서 물리적인 디바이스와 논리적인 Rank가 어떻게 연결(Mapping)되느냐에 따라 실제 데이터의 레이아웃과 통신 효율이 크게 달라집니다.
* 효율성: 모든 Rank가 전체 결과를 가지는 AllReduce와 달리, 각 Rank는 필요한 부분만 가지므로 통신 비용과 메모리 사용량을 줄일 수 있습니다.
#### 실제 활용 사례 ####
* 분산 학습 (ZeRO/FSDP): DeepSpeed의 ZeRO나 PyTorch FSDP 기술에서 그래디언트(Gradient)를 동기화할 때 주로 사용됩니다. 각 GPU는 전체 그래디언트 합계 중 자신이 업데이트를 담당할 샤드(Shard)만 수신하여 메모리 부하를 최소화합니다.
* 복합 연산: ReduceScatter를 수행한 후 AllGather를 연속해서 실행하면 논리적으로 AllReduce와 동일한 결과를 얻을 수 있습니다. 



## 레퍼런스 ##

* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
