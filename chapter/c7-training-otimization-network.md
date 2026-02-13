Aggregation(Spine) 스위치단에서 오버서브스크립션(Oversubscription)이 발생하고 있다면, 이건 단순히 노드 몇 개 사이의 문제가 아니라 랙(Rack)과 랙 사이의 거대한 데이터 병목이 터진 상황이라고 봐야한다.
이 계층에서 NCCL 버퍼 사이즈를 키우는 것은 오히려 독이 될 확률이 더 크다. Aggregation 스위치는 수많은 ToR로부터 들어오는 트래픽을 처리하는데, 여기서 NCCL 버퍼를 키워 한꺼번에 큰 덩어리(Burst)를 던지면 인캐스트(Incast) 현상으로 인해 스위치 버퍼가 넘쳐나고 패킷 드랍이 무더기로 발생하기 때문이다.

### 1. 계층적 집합 통신 (Hierarchical All-Reduce) 강제 ###
Aggregation 스위치를 통과하는 트래픽 자체를 최소화해야 합니다. 랙 내부(Intra-rack) GPU들끼리 먼저 Reduce를 끝내서 데이터를 1/N로 압축한 뒤, 그 결과값만 랙 간(Inter-rack) Aggregation 스위치로 보낸다. 
NCCL_ALGO=Tree 설정을 통해 물리적 토폴로지를 인지하게 하거나, NVIDIA SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 기능을 지원하는 멜라녹스(Mellanox) 스위치라면 이를 활성화하여 스위치 하드웨어가 직접 리듀스 연산을 하게 만들어야 한다. AWS EFA 는 불행하게도 이런 기능을 지원하지 않는다. 

### 2. 멀티레일(Multi-rail) 및 채널 최적화 ###
Aggregation 단의 대역폭을 100% 활용하기 위해 데이터를 여러 경로로 찢어야 한다. 버퍼를 키우지 말고 NCCL_MIN_NCHANNELS를 늘려 데이터를 잘게 쪼개서 여러 NIC와 스위치 경로로 분산시킨다.
큰 버퍼는 특정 경로에 과부하를 주지만, 작은 패킷 여러 개는 ECMP(Equal-Cost Multi-Path)나 Adaptive Routing이 적용된 Aggregation 단에서 훨씬 유연하게 흐르게 된다.

### 3. RoCE v2 혼잡 제어 (Congestion Control) 튜닝 ###
이더넷 기반이라면 Aggregation 스위치에서 발생하는 혼잡을 하드웨어 레벨에서 막아야 한다. ECN(Explicit Congestion Notification) 임계값을 아주 타이트하게 잡는다. 스위치 버퍼가 차기 전에 송신측(GPU 노드)에 "천천히 보내"라는 신호를 빨리 보내서 패킷 드랍을 원천 차단해야 한다. Aggregation 단에서 패킷 손실이 발생하면 재전송 오버헤드가 ToR 단보다 훨씬 크다. PFC 의  무손실(Lossless) 네트워크 설정이 완벽한지 다시 점검해야 한다.


EFA는 IB의 SHARP 같은 인-네트워크 컴퓨팅 기능이 없다. 따라서 Aggregation 단의 오버서브스크립션은 물리적 병목으로, 이를 해결하려면 랙 간 대역폭을 1:1로 증설하거나, 통신량을 줄이기 위해 Pipeline Parallelism을 도입해 랙 간 통신 횟수 자체를 설계 단계에서 줄여야 한다.
