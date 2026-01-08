## 실행 ##
```
torchrun --nproc_per_node=1 /home/ec2-user/train/train.py
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
