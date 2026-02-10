import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# 1. 모델 레이어 교체 함수 (제공해주신 코드 확장)
def replace_with_te_layers(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # H100 최적화를 위해 te.Linear로 교체
            te_linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                params_dtype=torch.bfloat16
            )
            te_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                te_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, te_linear)
        else:
            replace_with_te_layers(module)
    return model

# 2. FP8 지원 커스텀 Trainer 정의
class TETrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # H100 전용 FP8 훈련 레시피 (E4M3/E5M2 하이브리드 모드)
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)

        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # 역전파 및 가중치 업데이트는 기존과 동일하게 처리
        self.accelerator.backward(loss)
        return loss.detach()

# 3. 모델 및 토크나이저 준비
model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" # H100 필수 옵션
).cuda()

# 모델 레이어 교체
model = replace_with_te_layers(model)

# 4. 훈련 설정
args = TrainingArguments(
    output_dir="./llama8b-fp8-res",
    per_device_train_batch_size=4, # FP8은 메모리 여유가 있어 더 키울 수 있습니다
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True, # 기본 저장 정밀도는 bf16 유지
    logging_steps=10,
    max_steps=100 # 벤치마크용
)

# 5. 실행
trainer = TETrainer(
    model=model,
    args=args,
    train_dataset=your_dataset, # 준비하신 데이터셋
)

trainer.train()
