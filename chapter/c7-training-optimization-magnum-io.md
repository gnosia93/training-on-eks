## Magnum IO ##
Magnum IO는 NVIDIA가 만든 "데이터 이동의 최적화 도구 모음(Software Stack)" 이다.
GPU 연산이 아무리 빨라도 데이터를 가져오는 속도가 느리면 전체 연산 효율이 떨어지게 된다. 그래서 NVIDIA는 데이터가 저장소나 네트워크에서 GPU로 들어오는 모든 경로를 광속으로 만들기 위해 이 기술 들을 개발하였다.

#### 1. GPUDirect Storage (GDS) ####
* 역할: 데이터가 SSD → CPU 메모리 → GPU를 거치지 않고, SSD → GPU로 직접 쏴버리는 기술이다.
* 효과: CPU 부하를 줄이고 데이터 로딩 속도를 최대 2~3배 높인다.

#### 2. GPUDirect RDMA & P2P #### 
* 역할: 서버와 서버 사이(RDMA), 혹은 한 서버 내 GPU 사이(P2P)에서 데이터를 주고받을 때 CPU를 거치지 않게 한다..
* 연결: 우리가 얘기한 ATS/PASID가 바로 이 RDMA 성능을 극대화하기 위한 하드웨어적 밑바탕이 된다.

#### 3. NCCL (NVIDIA Collective Communications Library) ####
* 역할: 여러 개의 GPU가 동시에 계산할 때(분산 학습), 데이터가 꼬이지 않고 가장 빠른 길로 이동하도록 최적화하는 라이브러리이다.
* 연결: NVLink와 NVSwitch를 가장 잘 활용하도록 설계된 소프트웨어이다

#### 4. NVSHMEM ####
역할: 여러 GPU의 메모리를 마치 하나의 거대한 메모리처럼 보이게 하여, 개발자가 복잡한 복사 명령 없이 데이터를 다룰 수 있게 한다.


## GPUDirect Storage(GDS)를 활성화 ##

#### 1. 전제 조건 확인 ####
GDS는 데이터가 NVMe SSD → PCIe 스위치 → GPU로 직접 흐르게 합니다. 이를 위해선 다음이 필요합니다.
* GPU: H100 (SXM5)
* OS: Ubuntu 20.04/22.04 등 최신 리눅스 커널
* FS: GDS를 지원하는 파일 시스템 (가이드) - ext4, xfs, Lustre, Weka 등.

#### 2. nvidia-fs 커널 모듈 설치 ####
GDS의 핵심은 nvidia-fs라는 커널 드라이버입니다. 이게 있어야 GPU가 파일 시스템에 직접 명령을 내립니다.
* NVIDIA 가속 드라이버 설치: H100 드라이버가 깔려 있어야 합니다.
* nvidia-gds 패키지 설치:
```
sudo apt-get install nvidia-gds
```

#### 3. 커널 모듈 로드 및 확인 ####
```
sudo modprobe nvidia-fs
lsmod | grep nvidia_fs
```
#### 4. 서비스 활성화 (nvidia-gds-check) ####
NVIDIA에서 제공하는 도구로 현재 시스템이 GDS를 돌릴 준비가 되었는지 검사합니다.
```
/usr/local/gds/tools/gdscheck -p
```
* 결과 확인: 모든 항목이 PASS이거나 Supported여야 합니다. 특히 IOMMU(ATS/PASID) 설정이 안 되어 있으면 여기서 에러가 납니다.

#### 5. 애플리케이션에서 사용 (cuFile API) ####
GDS는 일반적인 read() 함수를 쓰지 않습니다. 개발자가 코드에서 cuFile API를 호출해야 작동합니다.
* C++: cuFileHandleRegister, cuFileRead 등의 함수 사용.
* PyTorch: 기본적으로 지원 준비 중이며, nv-kvs 같은 라이브러리를 통해 연동합니다.


#### AWS P5 인스턴스 사용 시 팁 ####
* AWS DLAMI를 사용하면 드라이버는 대부분 잡혀 있지만, GDS는 사용하는 스토리지(EBS, 로컬 NVMe, FSx for Lustre)에 따라 추가 설정이 필요할 수 있습니다.
* 로컬 NVMe(인스턴스 스토어): 가장 빠른 속도를 냅니다. /etc/default/nvidia-gds/의 설정 파일에서 해당 마운트 지점을 허용해야 합니다.
* FSx for Lustre: 대규모 학습 데이터를 쓸 때 GDS와 찰떡궁합입니다. AWS FSx GDS 가이드를 참고하여 클라이언트 설정을 맞추세요.
