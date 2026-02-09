## Shared Memory 최적화 ##

### 컨테이너 /dev/shm ###
도커 및 쿠버네티스에서 shared memroy 의 기본값은 64MB 로, 별도의 설정이 없는 경우 컨테이너의 /dev/shm 크기는 64MB 로 제한된다.
일반적인 고해상도 이미지 한 장이나 텐서 데이터 몇 개만으로도 순식간에 가득 차는 용량으로 파이토치의 num_workers가 전처리한 데이터를 이 좁은 공간에 밀어넣으려다 실패하면 즉시 Bus error (core dumped)가 발생하며 학습이 멈추게 된다. 

컨테이너 안에서 df -h /dev/shm을 입력해서 64M라고 나온다면 학습 안정성을 보장하기 위해서 아래과 같이 Pod 의 shared memory 공간을 늘려줘야 한다. 
```
# K8s 설정 예시
containers:
  - name: training-node                                               
    volumeMounts:                                           
    - mountPath: /dev/shm
      name: dshm

volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: "32Gi" # 호스트 RAM 중 32GB를 공유 메모리로 할당
```
* 일반적인 LLM 학습 환경에서는 전체 RAM의 25% ~ 50% 사이를 할당하는 것이 좋다.(텐서가 텍스트이냐, 이미지 또는 영상이냐에 따라 다르긴 하다) <--- 모니터링을 통해 적정량을 산출 필요.

### Shared Memory(shm)와의 연쇄 작용 ###
* DataLoader (Worker): 디스크에서 데이터를 읽어 전처리 후 Shared Memory(/dev/shm)에 저장한다.
* Main Process: Shared Memory 에 있는 데이터를 자기 영역으로 가져와 Pin Memory에 고정한다.
* GPU DMA 엔진: 고정된 메모리 주소에서 데이터를 읽어 GPU VRAM으로 직접 이동시킨다. 

### DataLoader의 pin_memory=True ###
일반적인 메모리(Pageable Memory)는 OS가 물리적 위치를 언제든 바꿀 수 있어, GPU가 데이터를 가져가려면 반드시 CPU가 데이터를 복사해서 전달해줘야 한다. Pin Memory 사용 시 메모리 주소가 물리적으로 고정(Lock)되어, GPU 내부의 DMA(Direct Memory Access) 엔진이 CPU 도움 없이 직접 시스템 RAM의 해당 주소에 접근하여 데이터를 복사해 간다.  
Pin Memory 옵션을 사용하는 경우 CPU가 데이터를 다른 곳으로 옮기는 오버헤드가 사라지게 되고, CPU가 다른 연산을 수행하는 동안 GPU DMA 엔진은 독립적으로 데이터를 끌어올 수 있어, 연산과 통신의 오버랩(Overlap)이 가능해 진다.


### sizeLimit과 Pod Resources의 관계 (중요!) ###
emptyDir로 할당한 메모리(32Gi)는 해당 Pod의 전체 메모리 사용량(usage)에 합산된다.
만약 Pod의 resources.limits.memory가 100Gi인데 sizeLimit을 32Gi로 설정했다면, 실제 학습 프로세스가 쓸 수 있는 일반 RAM은 68Gi로 줄어들게 된다. 이를 계산하지 않으면 Cgroup에 의한 OOM이 발생할 수 있다. Kubernetes Resource Management 규칙을 참고하여 limits를 넉넉히 잡아야 한다.

### ulimit -l (Locked Memory) 설정 확인 ###
pin_memory=True를 쓰려면 OS 레벨에서 "메모리를 고정(Lock)해도 좋다"는 허용량이 충분해야 한다.
컨테이너 내부에서 ulimit -l 값이 너무 작게 설정되어 있으면 pin_memory 할당 시 RuntimeError가 발생할 수 있으므로, 쿠버네티스 노드 설정이나 Docker 실행 시 --ulimit memlock=-1:-1 옵션을 통해 unlimited로 설정되어 있는지 확인이 필요하다.

### DataLoader의 prefetch_factor와의 관계 ###
공유 메모리 점유율을 결정하는 숨은 변수로 num_workers뿐만 아니라 prefetch_factor (기본값 2)에 의해서도 공유 메모리 사용량이 결정된다. 즉 num_workers * prefetch_factor 만큼의 데이터 배치가 항상 /dev/shm에 대기하게 되므로, 대용량 데이터(이미지/비디오) 학습 시에는 이 곱셈 결과에 맞춰 sizeLimit을 설계해야 한다.

### 모니터링 ###
nvidia-smi dmon이나 df -h /dev/shm을 통해 데이터 파이프라인이 병목 없이 흐르는지 모니터링할 수 있다.

## NCCL 의 Shared Memory 통신 매커니즘 ##
동일 노드안에서 GPU 와 GPU 간의 통신 방식은 NVLINK, P2P, Shared Memory 로 나뉘어 진다. NVLINK 를 활용하지 못하게 되는 경우(하드웨어 오류 또는 명시적으로 NVLINK 를 Disable 시킴 등) 
* PCIe P2P가 빠름에도 불구하고 NCCL이 /dev/shm을 활성화(NCCL_SHM_DISABLE=0)하는 이유는 'PCIe 복잡성' 때문이다.
* 모든 GPU가 서로 직접 P2P 통신을 할 수 있는 것은 아니다. 중간에 PCIe 스위치가 복잡하게 얽혀 있거나, 서로 다른 CPU 소켓에 꽂혀 있으면 P2P 통신이 하드웨어적으로 차단되는 경우가 빈번하다. 
* P2P가 불가능한 상황에서 NCCL은 시스템 멈춤(Hang)을 방지하기 위해 즉시 Shared Memory 방식으로 전환한다(Fallback). 이때 /dev/shm이 64MB라면 통신 시도가 시작되자마자 에러가 나며 학습이 중단된다.
* 작은 크기의 데이터를 여러 GPU에 뿌릴 때는 직접 P2P를 여러 번 맺는 것보다, 공유 메모리에 한 번 쓰고 다 같이 읽어가는 게 더 효율적일 때가 있다 (데이터 복사 효율이 좋음) 

### PCIe Topology 확인 ###
노드 내 GPU 간 P2P 지원 여부는 nvidia-smi topo -p2p r 명령어로 확인할 수 있다. 만약 결과가 NS(Not Supported)라면 해당 구간은 반드시 Shared Memory를 경유하게 되므로 /dev/shm 용량 확보가 
필수 이다.
