## 카펜터 로그 확인 ##

![](https://github.com/gnosia93/training-on-eks/blob/main/appendix/images/karpenter-message-1.png)

```
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter
```

## 오류 메시지 및 현상 ##

## ET/OFI Request 0x7f456c224810 completed with error. RC: 103. Error: 4126 (Unresponsive receiver (reachable by EFA device but handshake failed) My EFA addr: fi_addr_efa://[fe80::c55:2eff:fe44:db5]:0:142160333 My host id: i-07792e0f955a9673b Peer EFA addr ## 
```
llama-3-8b-node-0-0:194:713 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 19
llama-3-8b-node-0-0:194:712 [0] NCCL INFO Channel 00/0 : 3[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:194:712 [0] NCCL INFO Channel 01/0 : 3[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:194:712 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
llama-3-8b-node-0-0:194:712 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA

[2026-01-01 12:08:32] llama-3-8b-node-0-0:194:710 [0] int cm_req_handle_error_entry(nccl_net_ofi_context_t*, fid_cq*, fi_cq_err_entry*, uint16_t):46 NCCL WARN NET/OFI Request 0x7f456c222f60 completed with error. RC: 103. Error: 4126 (Unresponsive receiver (reachable by EFA device but handshake failed) My EFA addr: fi_addr_efa://[fe80::c55:2eff:fe44:db5]:0:142160333 My host id: i-07792e0f955a9673b Peer EFA addr: fi_addr_efa://[fe80::454:14ff:fedd:d5b3]:0:825957744 Peer host id: N/A). Completed length: 0
llama-3-8b-node-0-0:194:710 [0] NCCL INFO transport/net.cc:753 -> 6
llama-3-8b-node-0-0:194:712 [0] NCCL INFO transport.cc:198 -> 6

[2026-01-01 12:08:32] llama-3-8b-node-0-0:194:710 [0] int cm_req_handle_error_entry(nccl_net_ofi_context_t*, fid_cq*, fi_cq_err_entry*, uint16_t):46 NCCL WARN NET/OFI Request 0x7f456c224810 completed with error. RC: 103. Error: 4126 (Unresponsive receiver (reachable by EFA device but handshake failed) My EFA addr: fi_addr_efa://[fe80::c55:2eff:fe44:db5]:0:142160333 My host id: i-07792e0f955a9673b Peer EFA addr: fi_addr_efa://[fe80::454:14ff:fedd:d5b3]:0:825957744 Peer host id: N/A). Completed length: 0
llama-3-8b-node-0-0:194:710 [0] NCCL INFO transport/net.cc:753 -> 6
llama-3-8b-node-0-0:194:712 [0] NCCL INFO transport/generic.cc:19 -> 6
llama-3-8b-node-0-0:194:712 [0] NCCL INFO group.cc:146 -> 6
llama-3-8b-node-0-0:194:712 [0] NCCL INFO group.cc:73 -> 6 [Async thread]
llama-3-8b-node-0-0:194:194 [0] NCCL INFO group.cc:545 -> 6
llama-3-8b-node-0-0:194:194 [0] NCCL INFO group.cc:694 -> 6
llama-3-8b-node-0-0:194:194 [0] NCCL INFO enqueue.cc:2432 -> 6
[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/code/samples/deepspeed/llama-3-8b.py", line 163, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/code/samples/deepspeed/llama-3-8b.py", line 136, in main
[rank0]:     trainer.train()
[rank0]:   File "/usr/local/lib/python3.12/site-packages/transformers/trainer.py", line 2325, in train
[rank0]:     return inner_training_loop(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/transformers/trainer.py", line 2480, in _inner_training_loop
[rank0]:     model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
[rank0]:                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/accelerate/accelerator.py", line 1547, in prepare
[rank0]:     result = self._prepare_deepspeed(*args)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/accelerate/accelerator.py", line 2290, in _prepare_deepspeed
[rank0]:     engine, optimizer, _, lr_scheduler = ds_initialize(**kwargs)
[rank0]:                                          ^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/__init__.py", line 203, in initialize
[rank0]:     engine = DeepSpeedEngine(args=args,
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/runtime/engine.py", line 302, in __init__
[rank0]:     self._configure_distributed_model(model)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/runtime/engine.py", line 1415, in _configure_distributed_model
[rank0]:     self._broadcast_model()
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/runtime/engine.py", line 1327, in _broadcast_model
[rank0]:     dist.broadcast(p.data, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/comm/comm.py", line 118, in log_wrapper
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/comm/comm.py", line 225, in broadcast
[rank0]:     return cdb.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/comm/torch.py", line 215, in broadcast
[rank0]:     return torch.distributed.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2824, in broadcast
[rank0]:     work = group.broadcast([tensor], opts)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:3699, remote process exited or there was a network error, NCCL version 2.27.3
[rank0]: ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
[rank0]: Last error:
[rank0]: NET/OFI Request 0x7f456c224810 completed with error. RC: 103. Error: 4126 (Unresponsive receiver (reachable by EFA device but handshake failed) My EFA addr: fi_addr_efa://[fe80::c55:2eff:fe44:db5]:0:142160333 My host id: i-07792e0f955a9673b Peer EFA addr: fi_addr_efa://[fe80::454:14ff:fedd:d5b3]:0:825957744 Peer host id: N/A). Completed length: 0
[rank0]:[W101 12:08:33.945328381 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```


## torch.OutOfMemoryError: CUDA out of memory ##
```
Master Address: llama-3-8b-node-0-0.llama-3-8b
Master Port: 29500
df: /root/.triton/autotune: No such file or directory
[Rank 3] GPU Memory Flushed.
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/config.json
Model config LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "dtype": "bfloat16",
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "vocab_size": 128256
}

loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/tokenizer.json
loading file tokenizer.model from cache at None
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/special_tokens_map.json
loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/tokenizer_config.json
loading file chat_template.jinja from cache at None
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/config.json
`torch_dtype` is deprecated! Use `dtype` instead!
Model config LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "dtype": "bfloat16",
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "vocab_size": 128256
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/model.safetensors.index.json
Fetching 4 files: 100%|██████████| 4/4 [03:50<00:00, 57.69s/it]
Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.
Generate config GenerationConfig {
  "bos_token_id": 128000,
  "eos_token_id": 128001
}

Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 113.18it/s]
loading configuration file generation_config.json from cache at /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/generation_config.json
Generate config GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": 128001,
  "max_length": 4096,
  "temperature": 0.6,
  "top_p": 0.9
}

Could not locate the custom_generate/generate.py inside meta-llama/Meta-Llama-3-8B.
Generating test split: 100%|██████████| 4358/4358 [00:00<00:00, 520703.53 examples/s]
Generating train split: 100%|██████████| 36718/36718 [00:00<00:00, 988900.72 examples/s]
Generating validation split: 100%|██████████| 3760/3760 [00:00<00:00, 789160.48 examples/s]
Map: 100%|██████████| 36718/36718 [00:01<00:00, 23851.83 examples/s]
PyTorch: setting up devices
max_steps is given, it will override any value given in num_train_epochs
Using auto half precision backend
Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)
[rank3]:W0101 11:09:46.337000 192 site-packages/torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
[rank3]:W0101 11:09:46.337000 192 site-packages/torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
Adam Optimizer #0 is created with AVX512 arithmetic capability.
Config: alpha=0.000020, betas=(0.900000, 0.999000), weight_decay=0.010000, adam_w=1
[rank3]: Traceback (most recent call last):
[rank3]:   File "/workspace/code/samples/deepspeed/llama-3-8b.py", line 156, in <module>
[rank3]:     main()
[rank3]:   File "/workspace/code/samples/deepspeed/llama-3-8b.py", line 129, in main
[rank3]:     trainer.train()
[rank3]:   File "/usr/local/lib/python3.12/site-packages/transformers/trainer.py", line 2325, in train
[rank3]:     return inner_training_loop(
[rank3]:            ^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/transformers/trainer.py", line 2480, in _inner_training_loop
[rank3]:     model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
[rank3]:                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/accelerate/accelerator.py", line 1547, in prepare
[rank3]:     result = self._prepare_deepspeed(*args)
[rank3]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/accelerate/accelerator.py", line 2290, in _prepare_deepspeed
[rank3]:     engine, optimizer, _, lr_scheduler = ds_initialize(**kwargs)
[rank3]:                                          ^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/__init__.py", line 203, in initialize
[rank3]:     engine = DeepSpeedEngine(args=args,
[rank3]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/runtime/engine.py", line 302, in __init__
[rank3]:     self._configure_distributed_model(model)
[rank3]:   File "/usr/local/lib/python3.12/site-packages/deepspeed/runtime/engine.py", line 1359, in _configure_distributed_model
[rank3]:     self.module.to(self.device)
[rank3]:   File "/usr/local/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4343, in to
[rank3]:     return super().to(*args, **kwargs)
[rank3]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1369, in to
[rank3]:     return self._apply(convert)
[rank3]:            ^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 928, in _apply
[rank3]:     module._apply(fn)
[rank3]:   File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 928, in _apply
[rank3]:     module._apply(fn)
[rank3]:   File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 928, in _apply
[rank3]:     module._apply(fn)
[rank3]:   [Previous line repeated 2 more times]
[rank3]:   File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 955, in _apply
[rank3]:     param_applied = fn(param)
[rank3]:                     ^^^^^^^^^
[rank3]:   File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1355, in convert
[rank3]:     return t.to(
[rank3]:            ^^^^^
[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 119.38 MiB is free. Including non-PyTorch memory, this process has 8.38 GiB memory in use. Of the allocated memory 7.96 GiB is allocated by PyTorch, and 7.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
E0101 11:10:18.826000 7 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 0 (pid: 192) of binary: /usr/local/bin/python
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/distributed/run.py", line 901, in main
    run(args)
  File "/usr/local/lib/python3.12/site-packages/torch/distributed/run.py", line 892, in run
    elastic_launch(
  File "/usr/local/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 143, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 277, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
llama-3-8b.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
-----------------------------------------------------
```

지금 모델 로딩(4/4 shards)은 성공했으므로, 초기화 시점의 메모리 피크(Peak)만 오프로드로 깎아내면 바로 학습이 시작될 것입니다










### Allocate failed due to no healthy devices present ###
```
Events:
  Type     Reason                    Age                    From               Message
  ----     ------                    ----                   ----               -------
  Warning  FailedScheduling          11m                    default-scheduler  0/4 nodes are available: 4 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/4 nodes are available: 4 Preemption is not helpful for scheduling.
  Normal   Nominated                 11m                    karpenter          Pod should schedule on: nodeclaim/gpu-v278f
  Warning  FailedScheduling          10m (x4 over 10m)      default-scheduler  0/5 nodes are available: 1 node(s) had untolerated taint(s), 4 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/5 nodes are available: 5 Preemption is not helpful for scheduling.
  Warning  FailedScheduling          9m53s (x2 over 9m53s)  default-scheduler  0/5 nodes are available: 1 Insufficient nvidia.com/gpu, 4 node(s) didn't match Pod's node affinity/selector. no new claims to deallocate, preemption: 0/5 nodes are available: 5 Preemption is not helpful for scheduling.
  Normal   Scheduled                 9m22s                  default-scheduler  Successfully assigned default/llama-3-8b-node-0-0-w5psk to ip-10-0-5-89.ap-northeast-2.compute.internal
  Normal   Pulling                   9m20s                  kubelet            Pulling image "public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0"
  Warning  ExceededGracePeriod       6m31s                  kubelet            Container runtime did not kill the pod within specified grace period.
  Normal   Pulled                    5m29s                  kubelet            Successfully pulled image "public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0" in 3m51.333s (3m51.333s including waiting). Image size: 12916508213 bytes.
  Normal   Created                   5m28s                  kubelet            Created container: node
  Normal   Started                   5m26s                  kubelet            Started container node
  Normal   Killing                   5m26s                  kubelet            Stopping container node
  Warning  UnexpectedAdmissionError  3m2s                   kubelet            Allocate failed due to no healthy devices present; cannot allocate unhealthy devices nvidia.com/gpu, which is unexpected
  Warning  FailedMount               3m2s                   kubelet            MountVolume.SetUp failed for volume "kube-api-access-pj2nv" : object "default"/"kube-root-ca.crt" not registered
[ec2-user@ip-10-0-0-122 deepspeed]$ 
```



### #1. training memory warning ###

[2025-12-31 17:08:24,236] [WARNING] [stage3.py:2236:step] 1 pytorch allocator cache flushes since last step. this happens when there is high memory pressure and is detrimental to performance. if this is happening frequently consider adjusting settings to reduce memory consumption. If you are unable to make the cache flushes go away consider adding get_accelerator().empty_cache() calls in your training loop to ensure that all ranks flush their caches at the same time




### #1. 노드의 잦은 Not Ready 상태로의 변경 ###
훈련 도중 노드가 Not Ready 상태로 변경되면, 해당 노드에서 실행중인 파드는 쿠버네티스로 부터 종료 시그널을 받게 된다. 시그널(SIGTERM-Signal 15)을 받은 파드는 진행중인 작업을 중단하고 강제 종료 되는데, 이경우 NCCL 통신이 broken 되어 전체 작업이 비정상적으로 끝나게 된다. 이를 방지하게 위해서 카펜터를 쓰는 쿠버네티스 환경에서는 아래와 같이 두가지 설정이 필요하다.

#### 1. 파드 annotation 추가 #### 
```
  metadata:
    annotations:
      karpenter.sh/do-not-disrupt: "true"                  # Karpenter의 노드 회수 방지
```

#### 2. 카펜터 Consolidation 정책 조정 ####
```
disruption:
    consolidationPolicy: WhenEmpty                         # WhenEmptyOrUnderutilized 보다는 WhenEmpty을 사용 (보수적인 노드 콘솔리데이터 방식 채택)  
    consolidateAfter: 10m
```


### #2. "error":"no instance type which had enough resources and the required offering met the scheduling requirements ###

이 에러는 Kubernetes의 Karpenter 가 Pod 에서 요구한 리소스를 제공할 수 있는 EC2 인스턴스를 찾지 못했을 때 발생한다.
주로 다음의 설정이 충돌할 때 발생하며, 최신 가이드에 따른 해결 방법은 다음과 같다.

#### 1. 리소스 요구량과 인스턴스 사양 불일치 ####
YAML에 정의한 limits 값이 지정한 instance-type의 실제 물리적 사양보다 큰 경우이다.
* 체크: nvidia.com/gpu 나 vpc.amazonaws.com/efa 가 p4d.24xlarge 또는 p5.48xlarge 같은 실제 인스턴스가 제공하는 갯수와 일치하는지 확인.
* 예: p4d.24xlarge는 GPU가 8개인데, 실수로 nvidia.com/gpu 필드에 16개를 요청하면 위 에러가 발생.

#### 2. 가용 영역(AZ) 및 구매 옵션(Spot/On-Demand) 불일치 ####
가장 빈번한 원인으로, 지정한 AZ에 해당 인스턴스 재고가 없는 경우이다.
* 해결: topology.kubernetes.io/zone: ap-northeast-2a 설정을 제거하거나, 여러 AZ를 허용하도록 수정.
* 스팟 사용시: 스팟 인스턴스로 요청했는데 해당 리전에 스팟 물량이 없다면 capacity-type: on-demand로 변경.

#### 3. EFA 장치 요청 오류 ####
YAML에 vpc.amazonaws.com/efa: 8 과 같이 명시했다면, 해당 노드 그룹(또는 Karpenter Provisioner)에 EFA 드라이버와 관련 설정이 되어 있어야 함.
일반 GPU 인스턴스(EFA 미지원)를 쓰면서 EFA 리소스를 요청하면 스케줄링이 불가능.
* 테스트: vpc.amazonaws.com/efa 부분을 주석 처리하고 학습이 시작되는지 확인.

#### 4. Karpenter NodePool/Provisioner 제약 ####
Karpenter를 사용 중이라면, NodePool 설정에서 해당 인스턴스 타입을 허용하고 있는지 확인.
```
kubectl get nodepool -o yaml
```
spec.template.spec.requirements에 node.kubernetes.io/instance-type과 karpenter.sh/capacity-type이 Pod의 nodeSelector와 일치하는지 확인.
