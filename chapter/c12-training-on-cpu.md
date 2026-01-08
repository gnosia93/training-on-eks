![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/intel-amx.png)
Intel 제온 프로세서 기반의 7i 인스턴스에서 [IPEX(Intel Extension for PyTorch)](https://github.com/intel/intel-extension-for-pytorch)를 활용해 분산 훈련 성능을 극대화하려면, 인텔 4세대 및 5세대 제온 프로세서에 탑재된 [AMX(Advanced Matrix Extensions)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html) 가속 기술을 적극 활용해야 한다. AMX는 bf16 및 int8 데이터 타입의 행렬 연산을 비약적으로 가속하는 전용 엔진으로, IPEX 설정에서 dtype=torch.bfloat16을 활성화하면 GPU의 텐서 코어와 유사한 가속 성능을 CPU 환경에서도 구현할 수 있다. 특히 bf16은 모델 정밀도를 유지하면서도 연산 효율을 높여 분산 훈련 속도를 크게 향상시킨다.
이러한 하드웨어 가속은 인텔의 독자적인 기술이므로, AWS Graviton(m7g)이나 AMD(m7a) 인스턴스에서는 사용이 불가능하다. 

* Intel은 IPEX 대신 PyTorch 2.5 이상의 공식 버전을 사용할 것을 권장하며, 최신 PyTorch에는 Intel CPU/GPU 최적화 기능이 통합되어 있다. 이전 버전에서 필요한 import intel_extension_for_pytorch 및 ipex.optimize 관련 코드는 이제 불필요해 졌다. 

## efa 지원 인스턴스 조회 ##
```
 aws ec2 describe-instance-types \
    --filters "Name=instance-type,Values=*7i*" \
              "Name=network-info.efa-supported,Values=true" \
    --query "InstanceTypes[].[InstanceType, VCpuInfo.DefaultVCpus, VCpuInfo.DefaultCores, MemoryInfo.SizeInMiB]" \
    --output text | awk '{printf "Instance: %-18s | vCPU: %3d | Cores: %2d | Memory: %6d GiB\n", $1, $2, $3, $4/1024}'
```
[결과]
```
Instance: u7i-6tb.112xlarge  | vCPU: 448 | Cores: 224 | Memory:   6144 GiB
Instance: r7iz.32xlarge      | vCPU: 128 | Cores: 64 | Memory:   1024 GiB
Instance: r7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: i7ie.metal-48xl    | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: m7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:    768 GiB
Instance: i7ie.48xlarge      | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: c7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:    384 GiB
Instance: i7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: r7iz.metal-32xl    | vCPU: 128 | Cores: 64 | Memory:   1024 GiB
Instance: r7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: i7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: i7i.24xlarge       | vCPU:  96 | Cores: 48 | Memory:    768 GiB
Instance: c7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:    384 GiB
Instance: m7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:    768 GiB
```
EFA 가 필요한 경우  C / M / R7i.48xlarge 로 구성된 카펜터 노드풀을 생성 한다.  

## 카펜터 노드풀 생성 ##
Llama 3-8B 모델을 World Size 4로 훈련 시켜보면, 각 프로세스당 약 50GB 정도의 메모리를 점유한다.
노드풀 생성시 c7i.8xlarge(32 vCPU, 64 GB Memory) 기준으로 하나의 Pod 로 할당하게 설정한다.   
```
cat <<EOF | kubectl apply -f -  
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: cpu-amx
spec:
  template:
    metadata:
      labels:
        nodeType: "cpu-amx" 
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: "karpenter.k8s.aws/instance-family"
          operator: In
          values: ["c7i", "m7i", "r7i"]
        - key: "karpenter.k8s.aws/instance-size"
          operator: In
          values: ["8xlarge", "16xlarge", "24xlarge, 48xlarge"]
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: cpu-amx
      expireAfter: 720h # 30 * 24h = 720h
      taints:
      - key: "intel.com/amx"           
        value: "present"                  
        effect: NoSchedule               
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmpty       
    consolidateAfter: 20m
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: cpu-amx
spec:
  role: "eksctl-KarpenterNodeRole-training-on-eks"
  amiSelectorTerms:
    - name: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023)*"
  amiFamily: AL2023 
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks" 
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks" 
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
  userData: |
    #!/bin/bash
    # Intel AMX 성능 최적화를 위한 라이브러리 로드 및 설정  
    sudo yum install python3-devel -y
    dnf install -y google-perftools
    pip install intel-extension-for-pytorch
EOF
```
생성된 cpu-amx 노드 클래스와 노드 풀을 확인한다. 
```
kubectl get ec2nodeclass,nodepool
```
[결과]
```
NAME                                     READY   AGE
ec2nodeclass.karpenter.k8s.aws/cpu-amx   True    6m13s
ec2nodeclass.karpenter.k8s.aws/gpu       True    25h

NAME                            NODECLASS   NODES   READY   AGE
nodepool.karpenter.sh/cpu-amx   cpu-amx     0       True    6m13s
nodepool.karpenter.sh/gpu       gpu         0       True    25h
```


## 훈련 하기 ##
* 파드당 Memory 는 64GB, CPU 는 32 Core 로 설정.

<< 작성 필요 >>


## 레퍼런스 ##
* https://tutorials.pytorch.kr/recipes/amx.html

----
## 파라미터 분산 로딩 ##
```
from deepspeed.runtime.zero import Init # 2026년 표준 라이브러리

# ... (프로세스 그룹 초기화 및 토크나이저 설정)

# 3. 모델 로딩: 컨텍스트 매니저를 사용하여 '진짜' 조각 로딩 수행
with Init(): # 이 안에서 생성해야 처음부터 쪼개진 상태(Sharded)로 로드됨
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True, # 메모리 피크 방지를 위해 필수
        attn_implementation="sdpa",
    )

```
* 메모리 점유: with Init()을 쓰면 각 노드는 모델 전체를 담을 공간을 만들지 않고, 자기가 가질 4GB 분량의 공간만 확보합니다.
* 데이터 흐름: 1번 노드가 파일에서 가중치를 읽을 때, "이건 내 거네(유지)", "이건 2번 거네(전송 후 삭제)" 식으로 즉시 처리합니다.
* 성능: 모든 노드가 16GB를 로드했다가 버리는 헛수고를 하지 않으므로 학습 시작까지 걸리는 시간이 훨씬 단축됩니다.

### low_cpu_mem_usage=True ###
모델을 로딩할 때 메모리 점유율이 순간적으로 치솟는 '피크(Peak) 현상'을 강제로 억제하는 옵션입니다.
Hugging Face from_pretrained 함수에서 이 옵션이 있고 없고의 차이는 다음과 같습니다.

#### 1. 옵션이 없을 때 (기본 방식) ####
* 빈 껍데기 생성: 모델 구조와 똑같은 크기의 메모리 공간을 RAM에 만듭니다 (예: 16GB).
* 가중치 로딩: 디스크에서 실제 파라미터 값(State Dict)을 읽어와서 별도의 메모리에 올립니다 (또 16GB 추가).
* 복사: 읽어온 값을 빈 껍데기에 복사합니다.
* 결과: 순간적으로 모델 크기의 약 2배(32GB)에 달하는 RAM이 필요합니다. 이때 메모리가 부족하면 서버가 즉시 뻗어버립니다.

#### 2. low_cpu_mem_usage=True일 때 (최적화 방식) ####
* 이 옵션은 PyTorch의 'Meta Device' 기능을 활용합니다.
* 설계도만 로드: 실제 메모리를 차지하지 않는 '가짜 모델(Meta Model)'을 먼저 만듭니다. (0GB)
* 조각 로딩: 가중치 파일을 아주 작은 조각 단위로 읽습니다.
* 즉시 주입: 읽은 조각을 실제 메모리에 할당하면서 모델에 바로 채워 넣습니다.
* 결과: 메모리 사용량이 모델 크기(16GB) 이상으로 튀지 않고 선형적으로 일정하게 유지됩니다



주요 구성 포인트 요약
* DeepSpeed Stage 3: ds_config.json에서 설정한 대로 모델 파라미터를 4대의 서버에 4분의 1씩 나누어 올립니다.
* IPEX + bf16: CPU의 한계를 극복하기 위해 인텔 AMX 가속기를 사용하여 연산 속도를 높입니다.
* FSx 경로: output_dir을 FSx 경로로 지정하여 모든 노드가 동일한 스토리지를 공유하고, 체크포인트 저장 시 데이터 유실을 방지합니다.
* EFA 보안 그룹: 인스턴스 간 통신을 위해 보안 그룹에서 자기 참조(All Traffic)가 허용되어 있어야 gloo 백엔드가 정상 작동합니다.


----

```
import os
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, TrainingArguments, Trainer

def main():
    # 1. 랑데뷰 (GPU 간 통신 그룹 설정)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model_id = "meta-llama/Meta-Llama-3-8B"
    ds_config_path = "ds_config.json"

    # 2. 모델 설정(Config)만 먼저 읽기 (실제 가중치 로드 X)
    config = AutoConfig.from_pretrained(model_id)

    # 3. [핵심] DeepSpeed ZeRO-3 Init 컨텍스트 매니저
    # 이 안에서 모델을 로드하면 모든 GPU가 각각 읽지 않습니다.
    # Rank 0이 가중치를 한 조각씩 읽을 때마다 다른 GPU로 분산 배포하며,
    # 메모리에는 전체 모델이 아닌 1/N 조각만 남게 됩니다.
    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True # CPU RAM 사용량 최소화
        )

    # 4. Trainer 설정
    training_args = TrainingArguments(
        output_dir="./llama-8b-sharded",
        deepspeed=ds_config_path,
        per_device_train_batch_size=1,
        bf16=True,
        local_rank=local_rank,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=...
    )

    trainer.train()

if __name__ == "__main__":
    main()

```

왜 이렇게 하면 메모리가 절약되나요?
* deepspeed.zero.Init()의 역할: 이 컨텍스트 매니저는 모델의 파라미터를 생성할 때 Meta Device 기술을 사용하여 실제 메모리를 할당하지 않고 '선언'만 합니다.
* 분산 로딩 (Sharded Loading): from_pretrained가 호출되면 DeepSpeed가 로딩 과정을 가로챕니다. Rank 0이 디스크에서 가중치를 읽으면, 즉시 WORLD_SIZE(GPU 개수)로 나누어 각 GPU에 전달하고 자신은 그 조각만 남기고 나머지는 메모리에서 즉시 비웁니다.
* 중복 로드 방지: low_cpu_mem_usage=True와 DeepSpeed의 결합으로 인해, 모든 GPU가 동시에 16GB 파일을 읽어 CPU RAM이 터지는 현상을 방지합니다. Hugging Face 공식 문서에서도 ZeRO-3 사용 시 이 방식을 가장 권장합니다.








