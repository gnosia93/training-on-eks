### AllReduce ###
![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/AllReduce.png)
AllReduce는 분산 컴퓨팅 환경에서 여러 프로세스(Rank)가 가진 데이터를 하나로 모아 연산(Sum, Max, Min 등)한 뒤, 그 최종 결과물을 다시 모든 프로세스에게 동일하게 배포하는 통신 패턴이다. 
* Reduction: 각 노드가 가진 데이터의 i번째 요소들을 지정된 연산(예: 합계)으로 결합한다.
* Broadcast: 결합된 결과 벡터를 모든 참여 노드에 뿌려준다. 
결과적으로 모든 노드는 out[i] = rank_0[i] + rank_1[i] + ... + rank_k-1[i]와 같이 계산된 완전한 합계 데이터를 동일하게 보유하게 됩니다.

### 주요 활용 사례 및 구현 ###
* 딥러닝 학습: 분산 데이터 병렬 처리(DDP)에서 각 GPU가 계산한 Gradient(기울기)를 동기화하여 모델 가중치를 동일하게 업데이트할 때 필수적으로 사용됩니다.
* 라이브러리: NVIDIA NCCL은 GPU 간 최적화된 AllReduce를 제공하며, PyTorch Distributed 및 MPI (Message Passing Interface) 표준에서도 핵심 API로 구현되어 있다.

### 알고리즘 ###
* Ring-AllReduce: 데이터를 조각내어 옆에 있는 Rank에게 전달하는 방식을 반복합니다. 마치 릴레이 달리기처럼 데이터를 주고받으며 연산과 배포를 동시에 끝낸다. NVIDIA NCCL이 GPU 통신에서 이 방식을 활용한다.
  * 1단계 Reduce-Scatter: 링을 한 바퀴 돌면서 각 Rank가 데이터의 일부분(담당 구역)에 대한 최종 합계를 완성함.
  * 2단계 All-Gather: 완성된 자기 조각을 다시 한번 링을 타고 전달하여, 모든 Rank가 전체 데이터를 가질 수 있게 복사함. 
* Tree-based: 나무 뿌리처럼 데이터를 타고 올라가며 합치고, 다시 가지를 타고 내려보내는 방식이다.


### ReduceScatter ###
![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/ReduceScatter.png)

ReduceScatter는 여러 디바이스(Rank)에 흩어져 있는 데이터를 하나로 Reduce(요약/합산)한 뒤, 그 결과를 다시 각 Rank에 Scatter(분할 배분)하는 집합 통신(Collective Communication) 작업이다.
전체 데이터를 랭크 수만큼 파티션하고, 그 파티션(구역)마다 독립적인 Reduce 연산을 수행하여 Scatter(배분)한다. 각 랭크는 특정 구역의 첫 번째 주자가 되어 쉬프트(Shift)를 시작하고, 이들이 동시에 움직이면서 링 전체에 여러 개의 누적 연산이 동시다발적으로 진행된다

#### 주요 특징 및 동작 방식 ####
* 작업 흐름: 모든 Rank가 보유한 동일한 크기의 데이터 버퍼에 대해 요소별(element-wise) 연산(예: 합계, 최댓값)을 수행합니다.연산된 전체 결과는 N 개의 균등한 블록으로 나뉘어, 각 Rank i 전체 결과 중 자신의 인덱스에 해당하는 i 번째 블록만 수신하게 됩니다.
* Rank-Device 매핑의 영향: 이 작업은 결과 데이터가 어떤 순서로 어느 디바이스에 저장될지가 Rank 인덱스에 의해 결정됩니다. 따라서 물리적인 디바이스와 논리적인 Rank가 어떻게 연결(Mapping)되느냐에 따라 실제 데이터의 레이아웃과 통신 효율이 크게 달라집니다.
* 효율성: 모든 Rank가 전체 결과를 가지는 AllReduce와 달리, 각 Rank는 필요한 부분만 가지므로 통신 비용과 메모리 사용량을 줄일 수 있습니다.
#### 실제 활용 사례 ####
* 분산 학습 (ZeRO/FSDP): DeepSpeed의 ZeRO나 PyTorch FSDP 기술에서 그래디언트(Gradient)를 동기화할 때 주로 사용됩니다. 각 GPU는 전체 그래디언트 합계 중 자신이 업데이트를 담당할 샤드(Shard)만 수신하여 메모리 부하를 최소화합니다.
* 복합 연산: ReduceScatter를 수행한 후 AllGather를 연속해서 실행하면 논리적으로 AllReduce와 동일한 결과를 얻을 수 있습니다. 

### AlltoAll ###
![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/AlltoAll.png)
* ReduceScatter 과 동일하나 합치는게 아니라 Raw 데이터 그대로 들고 있다.
* AllToAll은 연산이 없어서 직관적이지만, 모든 GPU가 동시에 서로에게 데이터를 쏘기 때문에 네트워크 트래픽(Congestion)이 엄청나게 발생한다. (풀 메시 형태의 통신이 발생)
* 데이터의 배치(Layout)를 바꾸기 위해서 발생한다. 
  * Rank 0이 [A1, A2]를 들고 있고, Rank 1이 [B1, B2]를 들고 있을 때
  * AllToAll을 하면 Rank 0은 [A1, B1]을, Rank 1은 [A2, B2]를 갖게 됩니다.


## 레퍼런스 ##

* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
