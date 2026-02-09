### 컨테이너 /dev/shm 크기 ###
도커 및 쿠버네티스에서 shared memroy 의 기본값은 64MB 로, 별도의 설정이 없는 경우 컨테이너의 /dev/shm 크기는 64MB 로 제한된다.
일반적인 고해상도 이미지 한 장이나 텐서 데이터 몇 개만으로도 순식간에 가득 차는 용량으로 파이토치의 num_workers가 전처리한 데이터를 이 좁은 공간에 밀어넣으려다 실패하면 즉시 Bus error (core dumped)가 발생하며 학습이 멈추게 된다. 

컨테이너 안에서 df -h /dev/shm을 입력해서 64M라고 나온다면 학습 안정성을 보장하기 위해서 아래과 같이 Pod 의 shared memory 공간을 늘려줘야 한다. 
```
# K8s 설정 예시
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: "32Gi" # 호스트 RAM 중 32GB를 공유 메모리로 할당
```

### Shared Memory(shm)와의 연쇄 작용 ###
* DataLoader (Worker): 디스크에서 데이터를 읽어 전처리 후 Shared Memory(/dev/shm)에 저장한다.
* Main Process: Shared Memory 에 있는 데이터를 자기 영역으로 가져와 Pin Memory에 고정한다.
* GPU DMA 엔진: 고정된 메모리 주소에서 데이터를 읽어 GPU VRAM으로 직접 이동시킨다. 


### pin_memory=True ###
일반적인 메모리(Pageable Memory)는 OS가 물리적 위치를 언제든 바꿀 수 있어, GPU가 데이터를 가져가려면 반드시 CPU가 데이터를 복사해서 전달해줘야 한다. Pin Memory 사용 시 메모리 주소가 물리적으로 고정(Lock)되어, GPU 내부의 DMA(Direct Memory Access) 엔진이 CPU 도움 없이 직접 시스템 RAM의 해당 주소에 접근하여 데이터를 복사해 간다.  
Pin Memory 옵션을 사용하는 경우 CPU가 데이터를 다른 곳으로 옮기는 오버헤드가 사라지게 되고, CPU가 다른 연산을 수행하는 동안 GPU DMA 엔진은 독립적으로 데이터를 끌어올 수 있어, 연산과 통신의 오버랩(Overlap)이 가능해 진다.
