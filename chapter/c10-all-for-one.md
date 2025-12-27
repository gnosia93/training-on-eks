
[train.py]
```
import torch
import torch.nn as nn
import deepspeed
import argparse

# 1. 아규먼트 파싱 (DeepSpeed 필수 인자 포함)
def get_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Training')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # DeepSpeed 설정 파일을 받기 위한 인자
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

# 2. 아주 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 10))
    def forward(self, x):
        return self.net(x)

args = get_args()
model = SimpleModel()
dataset = [(torch.randn(100), torch.randint(0, 10, (1,)).item()) for _ in range(100)]

# 3. DeepSpeed 엔진 초기화
# 여기서 모델, 옵티마이저, 데이터로더가 DeepSpeed용으로 래핑됩니다.
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters()
)

# 4. 학습 루프
model_engine.train()
for x, y in dataset:
    # 데이터를 현재 GPU 장치로 이동
    x = x.to(model_engine.local_rank).half() # fp16 사용 시 .half()
    y = torch.tensor([y]).to(model_engine.local_rank)

    # 순전파 (Forward)
    outputs = model_engine(x.unsqueeze(0))
    loss = nn.CrossEntropyLoss()(outputs, y)

    # 역전파 및 최적화 (Step)
    # DeepSpeed가 내부적으로 backward, step, zero_grad를 관리합니다.
    model_engine.backward(loss)
    model_engine.step()

print("학습 완료!")
```

[ds_config.json]
```
{
  "train_batch_size": 32,
  "steps_per_print": 10,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "zero_optimization": {
    "show_config": true,
    "stage": 2
  },
  "fp16": {
    "enabled": true
  }
}
```
[requirements.txt]
```
# 1. 핵심 딥러닝 프레임워크 (CUDA 12.x 호환 버전 권장)
torch>=2.4.0
torchvision
torchaudio

# 2. 대규모 학습 최적화 및 분산 처리
deepspeed>=0.15.0
accelerate>=0.34.0

# 3. 모델 가속 및 커스텀 커널 (FlashAttention 등)
triton>=3.0.0
flash-attn>=2.6.0

# 4. 모델 허브 및 트랜스포머 라이브러리
transformers>=4.44.0
diffusers
safetensors

# 5. 데이터셋 및 평가
datasets
evaluate
scikit-learn

# 6. 유틸리티 및 모니터링
tensorboard
wandb
tqdm
psutil
pyyaml
```

### 실행하기 ###
```
pip install -r requirements.txt
torchrun --nproc_per_node=2 train.py --deepspeed --deepspeed_config ds_config.json
```
