## NCCL EFA 플러그인 로딩 확인 ##
NCCL 최적화를 수행하기 전에 EFA를 통한 분산 학습이 이뤄지고 있는지 확인이 필요하다. RANK 0 로그를 열어 nccl 이 efa 플러그인 성공적으로 로딩했는지 확인한다. 아래 [RANK 0 로그 예시] 에서 확인할 항목들은 아래와 같다. 이 항목들을 로그에서 관찰할 수 있다면 nccl 이 efa 를 이용하여 분산 학습을 하고 있다는 것을 의미한다.  

[efa 플러그인 관련 항목]
```
# 1. EFA 플러그인 로드 - AWS OFI(Open Fabric Interface) NCCL 플러그인(libnccl-net.so, aws-ofi-nccl)
NCCL INFO NET/Plugin: Successfully loaded external plugin libnccl-net.so
NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2

# 2. EFA 프로바이더 선택
NCCL INFO NET/OFI Setting provider_filter to efa
NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)

# 3. GPU 및 NCCL 버전 확인 - CUDA 13.0
NCCL INFO cudaDriverVersion 13000
NCCL INFO NCCL version 2.27.3+cuda12.9

# 4. 프로세스 및 GPU 매핑 - 16 랭크 구성
llama-3-8b-node-0-0:194:724 [3] ... cudaDev 3 nvmlDev 3 busId 3e000 commId ... rank 3 nranks 16
```

#### RANK 0 로그 예시 #### 
```
llama-3-8b-node-0-0:191:191 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:191:191 [0] NCCL INFO Bootstrap: Using eth0:10.0.5.28<0>
llama-3-8b-node-0-0:191:191 [0] NCCL INFO cudaDriverVersion 13000
llama-3-8b-node-0-0:191:191 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
llama-3-8b-node-0-0:194:194 [3] NCCL INFO cudaDriverVersion 13000
llama-3-8b-node-0-0:194:194 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:194:194 [3] NCCL INFO Bootstrap: Using eth0:10.0.5.28<0>
llama-3-8b-node-0-0:194:194 [3] NCCL INFO NCCL version 2.27.3+cuda12.9
llama-3-8b-node-0-0:194:194 [3] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:192:192 [1] NCCL INFO cudaDriverVersion 13000
llama-3-8b-node-0-0:192:192 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:192:192 [1] NCCL INFO Bootstrap: Using eth0:10.0.5.28<0>
llama-3-8b-node-0-0:192:192 [1] NCCL INFO NCCL version 2.27.3+cuda12.9
llama-3-8b-node-0-0:192:192 [1] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:191:191 [0] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Successfully loaded external plugin libnccl-net.so
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Using Libfabric version 2.1
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Configuring AWS-specific options
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Setting provider_filter to efa
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Internode latency set at 75.0 us
llama-3-8b-node-0-0:193:193 [2] NCCL INFO cudaDriverVersion 13000
llama-3-8b-node-0-0:193:193 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to ^docker0,lo
llama-3-8b-node-0-0:193:193 [2] NCCL INFO Bootstrap: Using eth0:10.0.5.28<0>
llama-3-8b-node-0-0:193:193 [2] NCCL INFO NCCL version 2.27.3+cuda12.9
llama-3-8b-node-0-0:193:193 [2] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI NIC group 0 device #0 0000:31:00.0
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Using transport protocol SENDRECV
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Creating one domain per process
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI GUID of rdmap49s0: 9131ba5d0000f700
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI GUID for dev[0]: 00000000000000000a00051c00000000
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Successfully loaded external plugin libnccl-net.so
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Successfully loaded external plugin libnccl-net.so
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Using Libfabric version 2.1
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Using Libfabric version 2.1
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Configuring AWS-specific options
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Configuring AWS-specific options
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Setting provider_filter to efa
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Setting provider_filter to efa
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Internode latency set at 75.0 us
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Internode latency set at 75.0 us
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Could not disable CUDA API usage for HMEM, disabling GDR
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Setting FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES not supported.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Need to force simple protocol: byte delivery ordering not supported
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Support for global registrations: false
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Support for DMA-BUF registrations: false
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Need to force simple protocol: GDR not supported
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Adding FI_EFA_FORK_SAFE=1 to environment
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Adding NCCL_PROTO=simple to environment
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI Adding NCCL_TUNER_PLUGIN=libnccl-net.so to environment
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Initialized NET plugin Libfabric
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:194:724 [3] NCCL INFO DMA-BUF is available on GPU device 3
llama-3-8b-node-0-0:194:724 [3] NCCL INFO ncclCommInitRankConfig comm 0x559d9a6ce460 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 3e000 commId 0x81dca7568efe26d7 - Init START
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Successfully loaded external plugin libnccl-net.so
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.16.2
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Using Libfabric version 2.1
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Configuring AWS-specific options
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Setting provider_filter to efa
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Internode latency set at 75.0 us
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI NIC group 0 device #0 0000:31:00.0
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Using transport protocol SENDRECV
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Creating one domain per process
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI GUID of rdmap49s0: 9131ba5d0000f700
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI GUID for dev[0]: 00000000000000000a00051c00000000
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Could not disable CUDA API usage for HMEM, disabling GDR
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Setting FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES not supported.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Need to force simple protocol: byte delivery ordering not supported
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Support for global registrations: false
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Support for DMA-BUF registrations: false
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Need to force simple protocol: GDR not supported
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Adding FI_EFA_FORK_SAFE=1 to environment
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Adding NCCL_PROTO=simple to environment
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI Adding NCCL_TUNER_PLUGIN=libnccl-net.so to environment
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Initialized NET plugin Libfabric
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:193:727 [2] NCCL INFO DMA-BUF is available on GPU device 2
llama-3-8b-node-0-0:193:727 [2] NCCL INFO ncclCommInitRankConfig comm 0x55ce46f4ce20 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 3c000 commId 0x81dca7568efe26d7 - Init START
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI NIC group 0 device #0 0000:31:00.0
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Using transport protocol SENDRECV
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Creating one domain per process
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI GUID of rdmap49s0: 9131ba5d0000f700
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI GUID for dev[0]: 00000000000000000a00051c00000000
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Could not disable CUDA API usage for HMEM, disabling GDR
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Setting FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES not supported.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Need to force simple protocol: byte delivery ordering not supported
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Support for global registrations: false
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Support for DMA-BUF registrations: false
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Need to force simple protocol: GDR not supported
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Adding FI_EFA_FORK_SAFE=1 to environment
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Adding NCCL_PROTO=simple to environment
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI Adding NCCL_TUNER_PLUGIN=libnccl-net.so to environment
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Initialized NET plugin Libfabric
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI NIC group 0 device #0 0000:31:00.0
llama-3-8b-node-0-0:192:725 [1] NCCL INFO DMA-BUF is available on GPU device 1
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)
llama-3-8b-node-0-0:192:725 [1] NCCL INFO ncclCommInitRankConfig comm 0x563eb6215af0 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 3a000 commId 0x81dca7568efe26d7 - Init START
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Using transport protocol SENDRECV
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Creating one domain per process
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI GUID of rdmap49s0: 9131ba5d0000f700
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI GUID for dev[0]: 00000000000000000a00051c00000000
llama-3-8b-node-0-0:193:727 [2] NCCL INFO RAS client listening socket at ::1<28028>
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Could not disable CUDA API usage for HMEM, disabling GDR
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Setting FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES not supported.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Need to force simple protocol: byte delivery ordering not supported
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Support for global registrations: false
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Support for DMA-BUF registrations: false
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Need to force simple protocol: GDR not supported
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Adding FI_EFA_FORK_SAFE=1 to environment
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Adding NCCL_PROTO=simple to environment
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI Adding NCCL_TUNER_PLUGIN=libnccl-net.so to environment
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Initialized NET plugin Libfabric
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:191:726 [0] NCCL INFO DMA-BUF is available on GPU device 0
llama-3-8b-node-0-0:191:726 [0] NCCL INFO ncclCommInitRankConfig comm 0x556ea5166cf0 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 38000 commId 0x81dca7568efe26d7 - Init START
llama-3-8b-node-0-0:192:725 [1] NCCL INFO RAS client listening socket at ::1<28028>
llama-3-8b-node-0-0:191:726 [0] NCCL INFO RAS client listening socket at ::1<28028>
llama-3-8b-node-0-0:194:724 [3] NCCL INFO RAS client listening socket at ::1<28028>
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Bootstrap timings total 0.829298 (create 0.000114, send 0.000270, recv 0.816621, ring 0.011037, delay 0.000001)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Bootstrap timings total 0.755313 (create 0.000069, send 0.000105, recv 0.000167, ring 0.742054, delay 0.000000)
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Bootstrap timings total 0.742893 (create 0.000057, send 0.000119, recv 0.000103, ring 0.728755, delay 0.000000)
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Bootstrap timings total 0.730364 (create 0.000079, send 0.000147, recv 0.000953, ring 0.727395, delay 0.000000)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Setting affinity for GPU 2 to 0-47
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NVLS multicast support is not available on dev 2 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Setting affinity for GPU 3 to 0-47
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NVLS multicast support is not available on dev 3 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Setting affinity for GPU 0 to 0-47
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Setting affinity for GPU 1 to 0-47
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NVLS multicast support is not available on dev 1 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:194:724 [3] NCCL INFO comm 0x559d9a6ce460 rank 3 nRanks 16 nNodes 4 localRanks 4 localRank 3 MNNVL 0
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
llama-3-8b-node-0-0:194:724 [3] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:193:727 [2] NCCL INFO comm 0x55ce46f4ce20 rank 2 nRanks 16 nNodes 4 localRanks 4 localRank 2 MNNVL 0
llama-3-8b-node-0-0:192:725 [1] NCCL INFO comm 0x563eb6215af0 rank 1 nRanks 16 nNodes 4 localRanks 4 localRank 1 MNNVL 0
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1
llama-3-8b-node-0-0:192:725 [1] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:193:727 [2] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:191:726 [0] NCCL INFO comm 0x556ea5166cf0 rank 0 nRanks 16 nNodes 4 localRanks 4 localRank 0 MNNVL 0
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Channel 00/02 :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Channel 01/02 :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Trees [0] 1/8/-1->0->-1 [1] 1/-1/-1->0->4
llama-3-8b-node-0-0:191:726 [0] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:191:726 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
llama-3-8b-node-0-0:194:724 [3] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
llama-3-8b-node-0-0:192:725 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
llama-3-8b-node-0-0:193:727 [2] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
llama-3-8b-node-0-0:191:733 [0] NCCL INFO [Proxy Service] Device 0 CPU core 45
llama-3-8b-node-0-0:193:736 [2] NCCL INFO [Proxy Service] Device 2 CPU core 0
llama-3-8b-node-0-0:193:739 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 30
llama-3-8b-node-0-0:191:737 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 16
llama-3-8b-node-0-0:192:734 [1] NCCL INFO [Proxy Service] Device 1 CPU core 5
llama-3-8b-node-0-0:192:738 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 31
llama-3-8b-node-0-0:194:735 [3] NCCL INFO [Proxy Service] Device 3 CPU core 11
llama-3-8b-node-0-0:194:740 [3] NCCL INFO [Proxy Service UDS] Device 3 CPU core 36
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:194:724 [3] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:194:724 [3] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:192:725 [1] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:192:725 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:193:727 [2] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:193:727 [2] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Enabled NCCL Func/Proto/Algo Matrix:
     Function |       LL     LL128    Simple   |          Tree           Ring  CollNetDirect   CollNetChain           NVLS       NVLSTree            PAT  
    Broadcast |        0         0         1   |             1              1              1              1              1              1              1  
       Reduce |        0         0         1   |             1              1              1              1              1              1              1  
    AllGather |        0         0         1   |             1              1              1              1              1              1              1  
ReduceScatter |        0         0         1   |             1              1              1              1              1              1              1  
    AllReduce |        0         0         1   |             1              1              1              1              1              1              1  

llama-3-8b-node-0-0:191:726 [0] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:191:726 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:191:726 [0] NCCL INFO CC Off, workFifoBytes 1048576
llama-3-8b-node-0-0:192:725 [1] NCCL INFO TUNER/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:194:724 [3] NCCL INFO TUNER/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:194:724 [3] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
llama-3-8b-node-0-0:192:725 [1] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
llama-3-8b-node-0-0:194:724 [3] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
llama-3-8b-node-0-0:192:725 [1] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
llama-3-8b-node-0-0:194:724 [3] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:192:725 [1] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:192:725 [1] NCCL INFO ncclCommInitRankConfig comm 0x563eb6215af0 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 3a000 commId 0x81dca7568efe26d7 - Init COMPLETE
llama-3-8b-node-0-0:194:724 [3] NCCL INFO ncclCommInitRankConfig comm 0x559d9a6ce460 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 3e000 commId 0x81dca7568efe26d7 - Init COMPLETE
llama-3-8b-node-0-0:192:725 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 16 total 1.02 (kernels 0.13, alloc 0.10, bootstrap 0.74, allgathers 0.01, topo 0.02, graphs 0.00, connections 0.00, rest 0.00)
llama-3-8b-node-0-0:194:724 [3] NCCL INFO Init timings - ncclCommInitRankConfig: rank 3 nranks 16 total 1.16 (kernels 0.16, alloc 0.13, bootstrap 0.83, allgathers 0.02, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
llama-3-8b-node-0-0:193:727 [2] NCCL INFO TUNER/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:193:727 [2] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
llama-3-8b-node-0-0:193:727 [2] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
llama-3-8b-node-0-0:193:727 [2] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:193:727 [2] NCCL INFO ncclCommInitRankConfig comm 0x55ce46f4ce20 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 3c000 commId 0x81dca7568efe26d7 - Init COMPLETE
llama-3-8b-node-0-0:193:727 [2] NCCL INFO Init timings - ncclCommInitRankConfig: rank 2 nranks 16 total 0.99 (kernels 0.13, alloc 0.07, bootstrap 0.76, allgathers 0.02, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
llama-3-8b-node-0-0:191:726 [0] NCCL INFO TUNER/Plugin: Plugin name set by env to libnccl-net.so
llama-3-8b-node-0-0:191:726 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.
llama-3-8b-node-0-0:191:726 [0] NCCL INFO TUNER/Plugin: Using tuner plugin nccl_ofi_tuner
llama-3-8b-node-0-0:191:726 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:191:726 [0] NCCL INFO ncclCommInitRankConfig comm 0x556ea5166cf0 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 38000 commId 0x81dca7568efe26d7 - Init COMPLETE
llama-3-8b-node-0-0:191:726 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 16 total 1.01 (kernels 0.13, alloc 0.11, bootstrap 0.73, allgathers 0.01, topo 0.02, graphs 0.00, connections 0.00, rest 0.00)
The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
max_steps is given, it will override any value given in num_train_epochs
Using auto half precision backend
Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4. Using DeepSpeed's value.
Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)
[rank1]:W1228 23:55:58.210000 192 site-packages/torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
[rank1]:W1228 23:55:58.210000 192 site-packages/torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000020, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
llama-3-8b-node-0-0:192:192 [1] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:191:191 [0] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:194:194 [3] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:193:193 [2] NCCL INFO Comm config Blocking set to 1
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO DMA-BUF is available on GPU device 3
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO DMA-BUF is available on GPU device 1
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO Assigned NET plugin Libfabric to comm
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO Using network Libfabric
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO DMA-BUF is available on GPU device 0
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO DMA-BUF is available on GPU device 2
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO ncclCommSplit comm 0x559d9ac13de0 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 3e000 parent 0x559d9a6ce460 splitCount 1 color 1734919490 key 3- Init START
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO ncclCommSplit comm 0x563eb875b740 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 3a000 parent 0x563eb6215af0 splitCount 1 color 1734919490 key 1- Init START
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO ncclCommSplit comm 0x556ea5a6db80 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 38000 parent 0x556ea5166cf0 splitCount 1 color 1734919490 key 0- Init START
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO ncclCommSplit comm 0x55ce48494fa0 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 3c000 parent 0x55ce46f4ce20 splitCount 1 color 1734919490 key 2- Init START
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO Setting affinity for GPU 1 to 0-47
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO NVLS multicast support is not available on dev 1 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO Setting affinity for GPU 3 to 0-47
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO NVLS multicast support is not available on dev 3 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO Setting affinity for GPU 2 to 0-47
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO NVLS multicast support is not available on dev 2 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Setting affinity for GPU 0 to 0-47
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO comm 0x55ce48494fa0 rank 2 nRanks 16 nNodes 4 localRanks 4 localRank 2 MNNVL 0
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO comm 0x559d9ac13de0 rank 3 nRanks 16 nNodes 4 localRanks 4 localRank 3 MNNVL 0
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO comm 0x563eb875b740 rank 1 nRanks 16 nNodes 4 localRanks 4 localRank 1 MNNVL 0
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO comm 0x556ea5a6db80 rank 0 nRanks 16 nNodes 4 localRanks 4 localRank 0 MNNVL 0
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Channel 00/02 :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Channel 01/02 :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Trees [0] 1/8/-1->0->-1 [1] 1/-1/-1->0->4
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO P2P Chunksize set to 131072
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
llama-3-8b-node-0-0:193:1167 [2] NCCL INFO [Proxy Service] Device 2 CPU core 8
llama-3-8b-node-0-0:192:1169 [1] NCCL INFO [Proxy Service] Device 1 CPU core 44
llama-3-8b-node-0-0:191:1171 [0] NCCL INFO [Proxy Service] Device 0 CPU core 30
llama-3-8b-node-0-0:194:1168 [3] NCCL INFO [Proxy Service] Device 3 CPU core 11
llama-3-8b-node-0-0:194:1173 [3] NCCL INFO [Proxy Service UDS] Device 3 CPU core 47
llama-3-8b-node-0-0:193:1170 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 26
llama-3-8b-node-0-0:192:1172 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 0
llama-3-8b-node-0-0:191:1174 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 38
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Enabled NCCL Func/Proto/Algo Matrix:
     Function |       LL     LL128    Simple   |          Tree           Ring  CollNetDirect   CollNetChain           NVLS       NVLSTree            PAT  
    Broadcast |        0         0         1   |             1              1              1              1              1              1              1  
       Reduce |        0         0         1   |             1              1              1              1              1              1              1  
    AllGather |        0         0         1   |             1              1              1              1              1              1              1  
ReduceScatter |        0         0         1   |             1              1              1              1              1              1              1  
    AllReduce |        0         0         1   |             1              1              1              1              1              1              1  

llama-3-8b-node-0-0:191:1160 [0] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO NCCL_PROTO set by environment to simple
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 1 p2p channels per peer
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO CC Off, workFifoBytes 1048576
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO ncclCommSplit comm 0x55ce48494fa0 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 3c000 parent 0x55ce46f4ce20 splitCount 1 color 1734919490 key 2 - Init COMPLETE
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO ncclCommSplit comm 0x563eb875b740 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 3a000 parent 0x563eb6215af0 splitCount 1 color 1734919490 key 1 - Init COMPLETE
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO ncclCommSplit comm 0x556ea5a6db80 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 38000 parent 0x556ea5166cf0 splitCount 1 color 1734919490 key 0 - Init COMPLETE
llama-3-8b-node-0-0:193:1166 [2] NCCL INFO Init timings - ncclCommSplit: rank 2 nranks 16 total 23.81 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.01, topo 0.01, graphs 0.00, connections 0.00, rest 23.79)
llama-3-8b-node-0-0:192:1157 [1] NCCL INFO Init timings - ncclCommSplit: rank 1 nranks 16 total 24.41 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.01, topo 0.01, graphs 0.00, connections 0.00, rest 24.38)
llama-3-8b-node-0-0:191:1160 [0] NCCL INFO Init timings - ncclCommSplit: rank 0 nranks 16 total 24.16 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.01, topo 0.01, graphs 0.00, connections 0.00, rest 24.14)
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO NET/OFI NCCL_OFI_TUNER is not available for platform : g6e.12xlarge, Fall back to NCCL's tuner
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO ncclCommSplit comm 0x559d9ac13de0 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 3e000 parent 0x559d9a6ce460 splitCount 1 color 1734919490 key 3 - Init COMPLETE
llama-3-8b-node-0-0:194:1163 [3] NCCL INFO Init timings - ncclCommSplit: rank 3 nranks 16 total 23.91 (kernels 0.00, alloc 0.00, bootstrap 0.00, allgathers 0.01, topo 0.01, graphs 0.00, connections 0.01, rest 23.88)
llama-3-8b-node-0-0:193:1176 [2] NCCL INFO Channel 00 : 2[2] -> 3[3] via SHM/direct/direct
llama-3-8b-node-0-0:192:1178 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:192:1178 [1] NCCL INFO Channel 01 : 1[1] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:193:1176 [2] NCCL INFO Channel 01 : 2[2] -> 3[3] via SHM/direct/direct
llama-3-8b-node-0-0:191:1179 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 15
llama-3-8b-node-0-0:194:1180 [3] NCCL INFO [Proxy Progress] Device 3 CPU core 29
llama-3-8b-node-0-0:194:1177 [3] NCCL INFO Channel 00/0 : 3[3] -> 4[0] [send] via NET/Libfabric/0
llama-3-8b-node-0-0:194:1177 [3] NCCL INFO Channel 01/0 : 3[3] -> 4[0] [send] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1175 [0] NCCL INFO Channel 00/0 : 15[3] -> 0[0] [receive] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1175 [0] NCCL INFO Channel 01/0 : 15[3] -> 0[0] [receive] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1175 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:191:1175 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:194:1177 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
llama-3-8b-node-0-0:193:1176 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
llama-3-8b-node-0-0:192:1178 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
llama-3-8b-node-0-0:191:1175 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
Stage 3 initialize beginning
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 27.2 GB, percent = 7.3%
llama-3-8b-node-0-0:192:1182 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:192:1182 [1] NCCL INFO Channel 01 : 1[1] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:193:1183 [2] NCCL INFO Channel 00 : 2[2] -> 3[3] via SHM/direct/direct
llama-3-8b-node-0-0:193:1183 [2] NCCL INFO Channel 01 : 2[2] -> 3[3] via SHM/direct/direct
llama-3-8b-node-0-0:194:1184 [3] NCCL INFO [Proxy Progress] Device 3 CPU core 32
llama-3-8b-node-0-0:194:1181 [3] NCCL INFO Channel 00/0 : 3[3] -> 4[0] [send] via NET/Libfabric/0
llama-3-8b-node-0-0:194:1181 [3] NCCL INFO Channel 01/0 : 3[3] -> 4[0] [send] via NET/Libfabric/0
DeepSpeedZeRoOffload initialize [begin]
MA 14.96 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 27.85 GB, percent = 7.5%
llama-3-8b-node-0-0:191:1186 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 16
llama-3-8b-node-0-0:191:1185 [0] NCCL INFO Channel 00/0 : 15[3] -> 0[0] [receive] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1185 [0] NCCL INFO Channel 01/0 : 15[3] -> 0[0] [receive] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1185 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:191:1185 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:194:1181 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
llama-3-8b-node-0-0:193:1183 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
llama-3-8b-node-0-0:192:1182 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
llama-3-8b-node-0-0:191:1185 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
Parameter Offload - Persistent parameters statistics: param_count = 65, numel = 266240
llama-3-8b-node-0-0:194:1188 [3] NCCL INFO Channel 00 : 3[3] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:194:1188 [3] NCCL INFO Channel 01 : 3[3] -> 2[2] via SHM/direct/direct
llama-3-8b-node-0-0:192:1187 [1] NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
llama-3-8b-node-0-0:193:1189 [2] NCCL INFO Channel 00 : 2[2] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:192:1187 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
llama-3-8b-node-0-0:193:1189 [2] NCCL INFO Channel 01 : 2[2] -> 1[1] via SHM/direct/direct
llama-3-8b-node-0-0:194:1188 [3] NCCL INFO Connected all trees
DeepSpeedZeRoOffload initialize [end]
MA 0.0 GB         Max_MA 14.96 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 32.26 GB, percent = 8.7%
Before creating fp16 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 32.23 GB, percent = 8.6%
llama-3-8b-node-0-0:191:1190 [0] NCCL INFO Channel 01/0 : 0[0] -> 4[0] [send] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1190 [0] NCCL INFO Channel 00/0 : 8[0] -> 0[0] [receive] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1190 [0] NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/Libfabric/0
llama-3-8b-node-0-0:191:1190 [0] NCCL INFO Channel 01/0 : 4[0] -> 0[0] [receive] via NET/Libfabric/0
llama-3-8b-node-0-0:193:1189 [2] NCCL INFO Connected all trees
llama-3-8b-node-0-0:191:1190 [0] NCCL INFO Connected all trees
llama-3-8b-node-0-0:192:1187 [1] NCCL INFO Connected all trees
After creating fp16 partitions: 2
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 37.49 GB, percent = 10.1%
Before creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 38.67 GB, percent = 10.4%
After creating fp32 partitions
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 47.39 GB, percent = 12.7%
Before initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 48.31 GB, percent = 13.0%
After initializing optimizer states
MA 0.0 GB         Max_MA 0.0 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 57.27 GB, percent = 15.4%
After initializing ZeRO optimizer
MA 0.03 GB         Max_MA 1.99 GB         CA 15.08 GB         Max_CA 15 GB 
CPU Virtual Memory:  used = 61.74 GB, percent = 16.6%
***** Running training *****
  Num examples = 36,718
  Num Epochs = 1
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 256
  Gradient Accumulation steps = 4
  Total optimization steps = 50
  Number of trainable parameters = 8,030,261,248
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
