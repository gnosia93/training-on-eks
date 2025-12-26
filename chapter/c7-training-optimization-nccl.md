분산 훈련 환경(특히 EKS와 EFA가 활성화된 환경)에서 GPU 간 및 노드 간 통신 경로를 확인하기 위해 NCCL Topology를 확인하는 방법은 크게 두가지 이다. 


### 1. NCCL 디버그 로그 활성화 ###
#### 환경변수 ####
Pod를 실행할 때 환경 변수를 설정하면, NCCL이 초기화될 때 감지된 모든 하드웨어 토폴로지(EFA 인터페이스, PCIe 스위치 구조 등)를 표준 출력(stdout)으로 보여준다.
```
env:
- name: NCCL_DEBUG
  value: "INFO"
- name: NCCL_DEBUG_SUBSYS
  value: "GRAPH,INIT,ENV"
```
#### Pod 로그 확인 ####
로그를 통해 EFA 인터페이스가 정상적으로 감지되었는지, GPU들이 NVLink로 연결되었는지 확인할 수 있다.
```
kubectl logs <pod-name> | grep -i "topology"
```
### 2. nvidia-smi topo -m 명령 ###
Pod 내부 또는 EKS 노드에 직접 접속하여 현재 호스트의 GPU 간 연결 상태를 매트릭스 형태로 확인할 수 있다.
```
kubectl exec -it <pod-name> -- nvidia-smi topo -m
```
* NV#: GPU 간 NVLink로 연결됨 (가장 빠름)
* PHB/PIX: PCIe 스위치 또는 호스트 브리지를 통해 연결됨
* NODE: 다른 CPU 소켓 또는 원격 노드 (네트워크 통신 필요)

## NCCL Tests ##
네트워크를 포함한 전체 토폴로지의 성능을 측정하려면 NVIDIA에서 제공하는 nccl-tests를 사용해야 한다.
다음 명령어를 EKS 노드 내 혹은 전용 컨테이너에서 실행한다. 
```
./all_reduce_perf -b 8 -e 128M -f 2 -g 8
```
EFA가 활성화된 경우 노드 간 통신에서 높은 대역폭과 낮은 지연 시간이 나타난다.

## 로그 확인 ##
* NCCL 로그에서 NCCL INFO NET/OFI라는 문구가 보여야 EFA(Libfabric)를 통한 최적의 노드 간 토폴로지가 구성된 것이다.
* Topology Aware Scheduler: EKS에서 대규모 훈련 시 성능 극대화를 위해 노드 내 GPU 위치를 고려하는 스케줄러 설정이 필요하다.
  
