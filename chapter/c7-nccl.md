② NCCL 백엔드 성능 튜닝
PyTorch 분산 학습에서 사용하는 NCCL(NVIDIA Collective Communications Library)은 노드 간 데이터를 주고받을 때 가장 빠른 경로를 찾습니다. 이때 네트워크 토폴로지를 확인하거나 소켓 버퍼 크기 등을 조정하여 통신 속도를 극대화하기 위해 NET_ADMIN 권한이 활용됩니다.
