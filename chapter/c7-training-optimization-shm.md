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
