## 실행 ##
* 단일 프로세스 
```
torchrun --nproc_per_node=1 samples/cpu-amx/cpu-llama3.py
```
* 멀티 프로세스
```
GLOO_LOG_LEVEL=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 train.py
```   
[결과]
```
W0108 08:13:23.689836 81085 torch/distributed/run.py:774] 
W0108 08:13:23.689836 81085 torch/distributed/run.py:774] *****************************************
W0108 08:13:23.689836 81085 torch/distributed/run.py:774] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0108 08:13:23.689836 81085 torch/distributed/run.py:774] *****************************************
[Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank [Gloo] Rank 3 is connected to 32 peer ranks.  is connected to Expected number of connected peer ranks is : 33 peer ranks. 
Expected number of connected peer ranks is : 3
[rank1]:[W108 08:13:30.227417871 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
[rank3]:[W108 08:13:30.230041522 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
My guessed rank = 1
My guessed rank = 3
[rank2]:[W108 08:13:30.435487868 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
My guessed rank = 2
[rank0]:[W108 08:13:30.741381204 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
My guessed rank = 0
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
[2026-01-08 08:13:32,205] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
[2026-01-08 08:13:32,216] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
[2026-01-08 08:13:32,322] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
DeepSpeed deepspeed.ops.comm.deepspeed_shm_comm_op built successfully
DeepSpeed deepspeed.ops.comm.deepspeed_shm_comm_op built successfully
DeepSpeed deepspeed.ops.comm.deepspeed_shm_comm_op built successfully
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
[2026-01-08 08:13:32,917] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
DeepSpeed deepspeed.ops.comm.deepspeed_shm_comm_op built successfully
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████| 4/4 [00:40<00:00, 10.01s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████| 4/4 [00:39<00:00,  9.87s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████| 4/4 [00:39<00:00,  9.88s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████| 4/4 [00:40<00:00, 10.03s/it]
/home/ec2-user/train/train.py:79: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
/home/ec2-user/train/train.py:79: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
--- 학습 시작 ---
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 128001}.
--- 학습 시작 ---
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 128001}.
/home/ec2-user/train/train.py:79: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
/home/ec2-user/train/train.py:79: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
--- 학습 시작 ---
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 128001}.
--- 학습 시작 ---
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 128001}.
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
2026-01-08 08:14:30,780 - accelerator.py - accelerate.accelerator - WARNING - Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4. Using DeepSpeed's value.
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank [Gloo] Rank 23 is connected to  is connected to 33 peer ranks.  peer ranks. Expected number of connected peer ranks is : Expected number of connected peer ranks is : 33

[Gloo] Rank [Gloo] Rank 30 is connected to  is connected to 33 peer ranks.  peer ranks. Expected number of connected peer ranks is : [Gloo] Rank Expected number of connected peer ranks is : 331

 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
Stage 3 initialize beginning
MA 4.81 GB         Max_MA 4.81 GB         CA 4.81 GB         Max_CA 5 GB 
CPU Virtual Memory:  used = 26.09 GB, percent = 3.5%
DeepSpeedZeRoOffload initialize [begin]
MA 4.81 GB         Max_MA 4.81 GB         CA 4.81 GB         Max_CA 5 GB 
CPU Virtual Memory:  used = 26.09 GB, percent = 3.5%
Parameter Offload - Persistent parameters statistics: param_count = 65, numel = 266240
DeepSpeedZeRoOffload initialize [end]
MA 4.81 GB         Max_MA 4.81 GB         CA 4.81 GB         Max_CA 5 GB 
CPU Virtual Memory:  used = 26.1 GB, percent = 3.5%
Before creating fp16 partitions
MA 4.81 GB         Max_MA 4.81 GB         CA 4.81 GB         Max_CA 5 GB 
CPU Virtual Memory:  used = 26.1 GB, percent = 3.5%
After creating fp16 partitions: 3
MA 5.44 GB         Max_MA 5.44 GB         CA 5.44 GB         Max_CA 5 GB 
CPU Virtual Memory:  used = 30.42 GB, percent = 4.1%
Before creating fp32 partitions
MA 5.44 GB         Max_MA 5.44 GB         CA 5.44 GB         Max_CA 5 GB 
CPU Virtual Memory:  used = 32.06 GB, percent = 4.3%
After creating fp32 partitions
MA 12.92 GB         Max_MA 12.92 GB         CA 12.92 GB         Max_CA 13 GB 
CPU Virtual Memory:  used = 58.58 GB, percent = 7.9%
Before initializing optimizer states
MA 12.92 GB         Max_MA 12.92 GB         CA 12.92 GB         Max_CA 13 GB 
CPU Virtual Memory:  used = 59.83 GB, percent = 8.0%
After initializing optimizer states
MA 20.4 GB         Max_MA 20.4 GB         CA 20.4 GB         Max_CA 20 GB 
CPU Virtual Memory:  used = 88.56 GB, percent = 11.9%
After initializing ZeRO optimizer
MA 24.47 GB         Max_MA 24.47 GB         CA 24.47 GB         Max_CA 24 GB 
CPU Virtual Memory:  used = 104.54 GB, percent = 14.1%
  0%|                                                                                                                 | 0/50 [00:00<?, ?it/s]
```


## 코드분석 ##
* AutoModelForCausalLM.from_pretrained 호출시 deepspeed 가 관여
  * [deepspeed.zero.Init(config_dict_or_path=deepspeed_config()), set_zero3_state()]

```
python3 train.py 
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. In order to use Torch DDP, launch your script with `python -m torch.distributed.launch
`torch_dtype` is deprecated! Use `dtype` instead!
[rank0]:[W108 06:27:06.141870526 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
/home/ec2-user/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:829: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
[2026-01-08 06:27:08,164] [WARNING] [real_accelerator.py:209:get_accelerator] Setting accelerator to CPU. If you have GPU or other accelerator, we were unable to detect it.
DeepSpeed deepspeed.ops.comm.deepspeed_shm_comm_op built successfully
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/ec2-user/train/train.py", line 95, in <module>
[rank0]:     main()
[rank0]:   File "/home/ec2-user/train/train.py", line 70, in main
[rank0]:     model = AutoModelForCausalLM.from_pretrained(
[rank0]:   File "/home/ec2-user/.local/lib/python3.9/site-packages/transformers/models/auto/auto_factory.py", line 604, in from_pretrained
[rank0]:     return model_class.from_pretrained(
[rank0]:   File "/home/ec2-user/.local/lib/python3.9/site-packages/transformers/modeling_utils.py", line 277, in _wrapper
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/home/ec2-user/.local/lib/python3.9/site-packages/transformers/modeling_utils.py", line 4967, in from_pretrained
[rank0]:     model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)
[rank0]:   File "/home/ec2-user/.local/lib/python3.9/site-packages/transformers/modeling_utils.py", line 4374, in get_init_context
[rank0]:     init_contexts.extend([deepspeed.zero.Init(config_dict_or_path=deepspeed_config()), set_zero3_state()])
[rank0]:   File "/home/ec2-user/.local/lib/python3.9/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1053, in __init__
[rank0]:     self.local_device = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))
[rank0]:   File "/usr/lib64/python3.9/os.py", line 679, in __getitem__
[rank0]:     raise KeyError(key) from None
[rank0]: KeyError: 'LOCAL_RANK'
```

## PyTorch Inductor ##
```
ec2-user   48920  0.0  0.0 7114092 364128 ?      Sl   07:03   0:00 /usr/bin/python3 /home/ec2-user/.local/lib/python3.9/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=48479 --read-fd=12 --write-fd=15 --torch-key=TmHW0OWOvK60ZPStAdgW7mmhY1tj9nMcqB+xYdVKN5k=
ec2-user   48922  0.0  0.0 7114092 364128 ?      Sl   07:03   0:00 /usr/bin/python3 /home/ec2-user/.local/lib/python3.9/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=48479 --read-fd=12 --write-fd=15 --torch-key=TmHW0OWOvK60ZPStAdgW7mmhY1tj9nMcqB+xYdVKN5k=
ec2-user   48924  0.0  0.0 7114092 364128 ?      Sl   07:03   0:00 /usr/bin/python3 /home/ec2-user/.local/lib/python3.9/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=48479 --read-fd=12 --write-fd=15 --torch-key=TmHW0OWOvK60ZPStAdgW7mmhY1tj9nMcqB+xYdVKN5k=
...
```
위의 프로세스들은 PyTorch Inductor가 Intel CPU(Xeon/AMX)에 딱 맞춘 전용 연산 코드를 '즉석에서' 제조하고 있는 단계를 실행하고 있다는 것이다.
더 구체적으로는 다음과 같은 작업을 수행한다.

#### 1. 그래프 융합 (Kernel Fusion) ####
모델의 수많은 연산(덧셈, 곱셈, 활성화 함수 등)을 하나하나 따로 실행하면 데이터 이동 시간이 오래 걸리는데 이 워커(Worker)들은 수천 개의 작은 연산들을 하나의 커다란 덩어리(Kernel)로 묶는 설계를 한다.
#### 2. 하드웨어 최적화 (Targeting Intel AMX) ####
현재 사용 중인 서버의 CPU가 Intel AMX를 지원한다는 것을 감지하면, 일반적인 연산 방식이 아니라 AMX 전용 명령어(TMM 등)를 사용하도록 C++ 코드를 생성한다.
컴파일 워커 들은 Llama 3 모델의 행렬 연산을 이 CPU에서 가장 빠르게 돌릴 수 있는 C++ 소스 코드를 짜라 라고 명령을 받고 병렬로 코드를 짜고 있는 것이다.

#### 3. JIT (Just-In-Time) 빌드 ####
워커들이 짠 C++ 코드를 실제로 실행 가능한 바이너리(기계어)로 만들기 위해 백그라운드에서 gcc나 g++ 같은 컴파일러를 돌린다. 프로세스 목록에 여러 개가 떠 있는 이유는 32개의 코어를 동시에 써서 이 빌드 시간을 최대한 단축하기 위해서이다
