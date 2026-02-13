## IB / ROCE ##
Aggregation(Spine) 스위치단에서 오버서브스크립션(Oversubscription)이 발생하고 있다면, 이건 단순히 노드 몇 개 사이의 문제가 아니라 랙(Rack)과 랙 사이의 거대한 데이터 병목이 터진 상황이라고 봐야한다.
이 계층에서 NCCL 버퍼 사이즈를 키우는 것은 오히려 독이 될 확률이 더 크다. Aggregation 스위치는 수많은 ToR로부터 들어오는 트래픽을 처리하는데, 여기서 NCCL 버퍼를 키워 한꺼번에 큰 덩어리(Burst)를 던지면 인캐스트(Incast) 현상으로 인해 스위치 버퍼가 넘쳐나고 패킷 드랍이 무더기로 발생하기 때문이다.

#### 1. 계층적 집합 통신 (Hierarchical All-Reduce) 강제 ####
Aggregation 스위치를 통과하는 트래픽 자체를 최소화해야 합니다. 랙 내부(Intra-rack) GPU들끼리 먼저 Reduce를 끝내서 데이터를 1/N로 압축한 뒤, 그 결과값만 랙 간(Inter-rack) Aggregation 스위치로 보낸다. 
NCCL_ALGO=Tree 설정을 통해 물리적 토폴로지를 인지하게 하거나, NVIDIA SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 기능을 지원하는 멜라녹스(Mellanox) 스위치라면 이를 활성화하여 스위치 하드웨어가 직접 리듀스 연산을 하게 만들어야 한다. AWS EFA 는 불행하게도 SHARP 기능을 지원하지 않는다. 

#### 2. 멀티레일(Multi-rail) 및 채널 최적화 ####
Aggregation 단의 대역폭을 100% 활용하기 위해 데이터를 여러 경로로 찢어야 한다. 버퍼를 키우지 말고 NCCL_MIN_NCHANNELS를 늘려 데이터를 잘게 쪼개서 여러 NIC와 스위치 경로로 분산시킨다.
큰 버퍼는 특정 경로에 과부하를 주지만, 작은 패킷 여러 개는 ECMP(Equal-Cost Multi-Path)나 Adaptive Routing이 적용된 Aggregation 단에서 훨씬 유연하게 흐르게 된다.

#### 3. RoCE v2 혼잡 제어 (Congestion Control) 튜닝 ####
이더넷 기반이라면 Aggregation 스위치에서 발생하는 혼잡을 하드웨어 레벨에서 막아야 한다. ECN(Explicit Congestion Notification) 임계값을 아주 타이트하게 잡는다. 스위치 버퍼가 차기 전에 송신측(GPU 노드)에 "천천히 보내"라는 신호를 빨리 보내서 패킷 드랍을 원천 차단해야 한다. Aggregation 단에서 패킷 손실이 발생하면 재전송 오버헤드가 ToR 단보다 훨씬 크다. PFC 의  무손실(Lossless) 네트워크 설정이 완벽한지 다시 점검해야 한다.

## EFA 환경 ##
EFA는 IB의 SHARP 같은 인-네트워크 컴퓨팅 기능이 없다. 따라서 Aggregation 단의 오버서브스크립션은 물리적 병목으로 이를 해결하려면 랙 간 대역폭을 1:1로 증설하거나, 통신량을 줄이기 위해 NCCL 버퍼 튜닝 같은 미시적인 방법보다는 통신 위상(Topology) 자체를 뒤틀어야 한다.

#### 1. Hierarchical All-Reduce (가장 현실적) ####
랙 내부(Intra-rack)는 100G/200G EFA 대역폭을 풀로 써서 공유 메모리로 리듀스하고, Aggregation 단으로는 딱 한 놈(Rank)만 대표로 데이터를 보낸다. Aggregation 스위치를 통과하는 노드 간 트래픽을 랙당 노드 수만큼(예: 1/8) 줄일 수 있다.

#### 2. NCCL_ALGO=Tree 강제 ####
EFA 환경에서도 NCCL 알고리즘을 Tree로 강제하면, 링(Ring)처럼 모든 노드를 거치지 않고 계층적으로 데이터를 모우게 된다. Aggregation 스위치를 타는 횟수를 물리적으로 줄이는 유일한 소프트웨어 설정이다.

#### 3. Chunking 최적화 (NCCL_BUFFSIZE의 역설) ####
오히려 버퍼를 줄여야 할 수도 있다. 대형 버퍼는 Aggregation 스위치에 마이크로 버스트(Micro-burst)를 유발해 인캐스트를 심화시키게 된다. 작은 청크로 쪼개서 SRD가 여러 경로로 잘게 쪼개 보내도록 유도하는 게 레이턴시 안정성에는 더 좋다.

## NCCL ##

* NCCL의 기본 채널 수(NCCL_MIN_NCHANNELS)는 고정된 하나의 숫자가 아니라, 사용 중인 GPU의 개수와 하드웨어 토폴로지(NVLink 유무 등)에 따라 시스템이 자동으로 결정된다.
* NCCL 버퍼 사이즈의 기본값은 4MB(4194304 bytes)이다. 오버스크립션이 발생하는 경우 버퍼사이즈를 과감하게 2MB 또는 1MB 로 줄여서 성능을 측정해 본다.
* NCCL_ALGO=tree 인 경우 특정 GPU가 '부모(Parent) 노드' 역할을 맡게된다. 하위 GPU들이 보낸 데이터를 자신의 메모리에서 합친(Reduce) 뒤, 결과값만 상위 노드로 전달한다. 리듀스 연산은 IB SHARP 처럼 스위치가 아니라 GPU가 하게된다. 계층 구조를 짜서 Aggregation 스위치를 통과하는 데이터의 절대량과 왕복 횟수를 줄이는 전략이다. 이 Tree 는 이진트리로 하나의 부모당 자식노드는 2개이다. 
* NCCL_TREE_THRESHOLD=1048576 (1MB 이상은 Ring 사용)
   * 기본값은 LLONG_MAX 으로 NCC_ALGO 가 Tree 인 경우 모든 패킷은 Tree 방식으로 전달된다.  
   * 1MB 미만 메시지: 트리(Tree) 알고리즘 사용 - 레이턴시에 민감한 작은 메시지를 빠르게 모으는 데 유리
   * 1MB 이상 메시지: 링(Ring) 알고리즘 사용 - 대역폭을 꽉 채워야 하는 큰 메시지를 모든 노드에 골고루 분산시켜 쏘는 데 유리 
