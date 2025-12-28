<< 작성중 ..>>

## NCCL 로그 ##
```
PyTorch: setting up devices
df: /root/.triton/autotune: No such file or directory
llama-3-8b-node-0-0:152:152 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:152:152 [0] NCCL INFO Bootstrap: Using eth0:10.0.4.51<0>
llama-3-8b-node-0-0:152:152 [0] NCCL INFO cudaDriverVersion 13000
llama-3-8b-node-0-0:152:152 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
llama-3-8b-node-0-0:152:152 [0] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Successfully loaded external plugin libnccl-net.so
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI Using Libfabric version 2.1
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI Configuring AWS-specific options
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI Setting provider_filter to efa
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI Internode latency set at 75.0 us
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI No eligible providers were found

[2025-12-28 16:04:25] llama-3-8b-node-0-0:152:386 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):259 NCCL WARN NET/OFI Unable to find a protocol that worked.  Failing initialization.

[2025-12-28 16:04:25] llama-3-8b-node-0-0:152:386 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):352 NCCL WARN NET/OFI aws-ofi-nccl initialization failed

[2025-12-28 16:04:25] llama-3-8b-node-0-0:152:386 [0] ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t):166 NCCL WARN NET/OFI Initializing plugin failed
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/IB : No device found.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/IB : Using [RO]; OOB eth0:10.0.4.51<0>
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/Socket : Using [0]eth0:10.0.4.51<0>
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Initialized NET plugin Socket
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Assigned NET plugin Socket to comm
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Using network Socket
llama-3-8b-node-0-0:152:386 [0] NCCL INFO ncclCommInitRankConfig comm 0x562b2c1973d0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 36000 commId 0xbc42413ad5ab3a45 - Init START
llama-3-8b-node-0-0:152:386 [0] NCCL INFO RAS client listening socket at ::1<28028>
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Bootstrap timings total 59.827020 (create 0.000078, send 0.000189, recv 59.819732, ring 0.005660, delay 0.000001)
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Setting affinity for GPU 0 to 0-31
llama-3-8b-node-0-0:152:386 [0] NCCL INFO comm 0x562b2c1973d0 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Channel 00/02 : 0 1 2 3
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Channel 01/02 : 0 1 2 3
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Trees [0] 2/-1/-1->0->-1 [1] -1/-1/-1->0->1
llama-3-8b-node-0-0:152:386 [0] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:152:386 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
llama-3-8b-node-0-0:152:456 [0] NCCL INFO [Proxy Service] Device 0 CPU core 12
llama-3-8b-node-0-0:152:457 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 20
llama-3-8b-node-0-0:152:386 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
llama-3-8b-node-0-0:152:386 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
llama-3-8b-node-0-0:152:386 [0] NCCL INFO CC Off, workFifoBytes 1048576
llama-3-8b-node-0-0:152:386 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
llama-3-8b-node-0-0:152:386 [0] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
llama-3-8b-node-0-0:152:386 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.8xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:152:386 [0] NCCL INFO ncclCommInitRankConfig comm 0x562b2c1973d0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 36000 commId 0xbc42413ad5ab3a45 - Init COMPLETE
llama-3-8b-node-0-0:152:386 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 59.98 (kernels 0.13, alloc 0.01, bootstrap 59.83, allgathers 0.01, topo 0.00, graphs 0.00, connections 0.00, rest 0.00)
The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
Using auto half precision backend
Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4. Using DeepSpeed's value.
Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)
[rank0]:W1228 16:05:29.287000 152 site-packages/torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
[rank0]:W1228 16:05:29.287000 152 site-packages/torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000020, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
llama-3-8b-node-0-0:152:152 [0] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Assigned NET plugin Socket to comm
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Using network Socket
llama-3-8b-node-0-0:152:507 [0] NCCL INFO ncclCommSplit comm 0x562b2ca93860 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 36000 parent 0x562b2c1973d0 splitCount 1 color 2003953581 key 0- Init START
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Setting affinity for GPU 0 to 0-31
llama-3-8b-node-0-0:152:507 [0] NCCL INFO comm 0x562b2ca93860 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Channel 00/02 : 0 1 2 3
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Channel 01/02 : 0 1 2 3
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Trees [0] 2/-1/-1->0->-1 [1] -1/-1/-1->0->1
llama-3-8b-node-0-0:152:507 [0] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
llama-3-8b-node-0-0:152:508 [0] NCCL INFO [Proxy Service] Device 0 CPU core 1
llama-3-8b-node-0-0:152:509 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 3
llama-3-8b-node-0-0:152:507 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
llama-3-8b-node-0-0:152:507 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
llama-3-8b-node-0-0:152:507 [0] NCCL INFO CC Off, workFifoBytes 1048576
llama-3-8b-node-0-0:152:507 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.8xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:152:507 [0] NCCL INFO ncclCommSplit comm 0x562b2ca93860 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 36000 parent 0x562b2c1973d0 splitCount 1 color 2003953581 key 0 - Init COMPLETE
llama-3-8b-node-0-0:152:507 [0] NCCL INFO Init timings - ncclCommSplit: rank 0 nranks 4 total 0.10 (kernels 0.00, alloc 0.00, bootstrap 0.01, allgathers 0.01, topo 0.00, graphs 0.00, connections 0.00, rest 0.08)
llama-3-8b-node-0-0:152:511 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 3
llama-3-8b-node-0-0:152:510 [0] NCCL INFO Channel 00/0 : 3[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:510 [0] NCCL INFO Channel 01/0 : 3[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:510 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:152:510 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:152:510 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
Stage 3 initialize beginning
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 7.77 GB, percent = 3.1%
DeepSpeedZeRoOffload initialize [begin]
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 7.71 GB, percent = 3.1%
llama-3-8b-node-0-0:152:513 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 7
llama-3-8b-node-0-0:152:512 [0] NCCL INFO Channel 00/0 : 3[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:512 [0] NCCL INFO Channel 01/0 : 3[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:512 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:152:512 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:152:512 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
Parameter Offload - Persistent parameters statistics: param_count = 65, numel = 266240
DeepSpeedZeRoOffload initialize [end]
MA 0.0 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 11.78 GB, percent = 4.7%
Before creating fp16 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 11.78 GB, percent = 4.7%
llama-3-8b-node-0-0:152:529 [0] NCCL INFO Channel 00/0 : 2[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:529 [0] NCCL INFO Channel 00/0 : 0[0] -> 2[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:152:529 [0] NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:529 [0] NCCL INFO Connected all trees
After creating fp16 partitions: 3
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 15.85 GB, percent = 6.4%
Before creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 15.85 GB, percent = 6.4%
After creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 24.16 GB, percent = 9.7%
Before initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 24.16 GB, percent = 9.7%
After initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 32.63 GB, percent = 13.1%
After initializing ZeRO optimizer
MA 0.03 GB         Max_MA 1.99 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 36.11 GB, percent = 14.5%
***** Running training *****
  Num examples = 36,718
  Num Epochs = 3
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 1,722
  Number of trainable parameters = 8,030,261,248
  0%|          | 0/1722 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
llama-3-8b-node-0-0:152:532 [0] NCCL INFO Channel 00/0 : 2[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:532 [0] NCCL INFO Channel 00/0 : 0[0] -> 2[0] [send] via NET/Socket/0
llama-3-8b-node-0-0:152:532 [0] NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
llama-3-8b-node-0-0:152:532 [0] NCCL INFO Connected all trees
```

## NCCL 최적화 ##

### 1. EFA 환경 최적화 (AWS 필수 설정) ###
AWS의 고속 네트워크망을 제대로 쓰려면 NCCL이 EFA를 기본 통신 계층으로 사용하도록 강제해야 합니다.
* FI_PROVIDER="efa": 통신 프로바이더를 EFA로 지정합니다.
* NCCL_PROTO=simple: EFA 환경에서는 복잡한 프로토콜보다 simple이 더 안정적이고 성능이 잘 나오는 경우가 많습니다.
* FI_EFA_USE_DEVICE_RDMA=1: GPU 간 직접 통신(RDMA)을 활성화하여 CPU 개입을 최소화합니다.

### 2. NCCL 성능 디버깅 (로깅) ###
튜닝 전, 현재 NCCL이 어떻게 작동하는지 파악하는 것이 우선입니다.
* NCCL_DEBUG=INFO: 모든 통신 로그를 출력합니다. 로그에서 "Selected Provider is EFA" 또는 "NVLink" 사용 여부를 반드시 확인하세요.
* NCCL_DEBUG_SUBSYS=GRAPH,INIT,ENV: 토폴로지 구성과 환경 변수 인식 과정을 더 자세히 들여다볼 때 사용합니다.

### 3. 주요 환경 변수 튜닝 (Performance Tuning) ###
훈련 속도(Throughput)를 높이기 위해 다음 변수들을 조정해 보며 최적값을 찾아야 합니다.
* NCCL_BUFFSIZE: 통신 버퍼 크기입니다. 기본값은 2MB(2097152)이나, 대규모 모델 훈련 시 4194304 (4MB) 또는 8388608 (8MB)로 늘리면 성능이 향상될 수 있습니다.
* NCCL_P2P_LEVEL: GPU 간 P2P(Point-to-Point) 통신 방식을 제어합니다. (예: 5는 NVLink를 통한 직접 연결 사용)
* NCCL_IB_DISABLE=1: AWS EFA 사용 시 InfiniBand(IB) 관련 에러가 발생한다면 이를 비활성화하여 EFA만 타도록 유도합니다.


## 설정 확인 ##
설정 후 파드가 실행되면 로그(kubectl logs <pod-name>)를 확인한다. NCCL_DEBUG=INFO 덕분에 다음과 같은 로그가 찍혀야 정상이다.
* NCCL INFO NET/OFI Selected Provider is efa (EFA가 정상 선택됨)
* NCCL INFO NET/OFI Using Profile efa
* NCCL INFO Using network AWS Libfabric


## 레퍼런스 ##

* https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
