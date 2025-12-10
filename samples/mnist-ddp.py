import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 분산 환경 초기화 함수
def setup_ddp():
    # Kubeflow PyTorchJob Operator가 주입하는 환경 변수를 사용합니다.
    if 'MASTER_ADDR' not in os.environ:
        print("DDP 환경 변수가 설정되지 않았습니다. 단일 프로세스 모드로 실행합니다.")
        return False
    
    # 환경 변수 가져오기
    rank = int(os.environ['RANK'])
    # local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])

    print(f"Initializing DDP | Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}, Master: {master_addr}:{master_port}")
    
    # 통신 백엔드 설정 (nccl은 GPU에 최적화됨)
    dist.init_process_group(backend='nccl', 
                            init_method=f'tcp://{master_addr}:{master_port}',
                            world_size=world_size,
                            rank=rank)
    
    # 현재 프로세스에 할당된 GPU 설정
    torch.cuda.set_device(local_rank)
    print(f"Process {rank} using GPU {local_rank}")
    return True

# 2. 신경망 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 3. 학습 함수
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # 랭크 0에서만 로그 출력 (중복 방지)
        if dist.get_rank() == 0 and batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 4. 메인 실행 함수
def main():
    use_ddp = setup_ddp()
    
    # DDP 사용 시 local_rank, 아니면 0 (단일 GPU)
    device_id = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{device_id}")

    # 데이터 로드 및 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 데이터셋 다운로드는 랭크 0에서만 수행하여 충돌 방지
    if use_ddp and dist.get_rank() != 0:
        dist.barrier() # 랭크 0이 다운로드할 때까지 대기

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    if use_ddp and dist.get_rank() == 0:
        dist.barrier() # 랭크 0이 완료되면 다른 랭크들 진행

    # DDP를 위한 DistributedSampler 사용
    sampler = DistributedSampler(train_dataset) if use_ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=sampler,
        shuffle=(sampler is None), # DDP 사용 시 shuffle=False
        num_workers=2,
        pin_memory=True
    )

    # 모델 초기화
    model = Net().to(device)
    
    # DDP 모델로 감싸기
    if use_ddp:
        model = DDP(model, device_ids=[device_id], output_device=device_id)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 시작
    for epoch in range(1, 3): # 예제에서는 2 에폭만 실행
        if use_ddp:
            sampler.set_epoch(epoch) # Epoch마다 셔플링 시드 설정
        train(model, device, train_loader, optimizer, epoch)

    # 학습 완료 후 DDP 종료 (선택 사항)
    if use_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
