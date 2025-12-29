## 어디에 무엇을 설치해야 할까? ##
호스트에는 드라이버만, 컨테이너에는 CUDA를 설치하는 것이 표준이다.

### 1. 호스트(Host) 설치 ###
호스트에는 하드웨어를 직접 제어하는데 필요한 소프트웨어를 설치해야 한다.  
* NVIDIA 커널 드라이버 (GPU Driver): 하드웨어와 OS를 연결하는 가장 기초적인 소프트웨어이다. (예: nvidia-smi를 실행했을 때 나오는 드라이버 버전)
* NVIDIA Container Toolkit (구 nvidia-docker2): 컨테이너가 호스트의 GPU 드라이버를 인식하고 사용할 수 있도록 다리 역할을 해주는 도구이다.
* CUDA 커널 (선택적): 사실 호스트에는 CUDA 전체를 깔 필요가 없다. 드라이버에 포함된 CUDA 드라이버(libcuda.so)만 있으면 컨테이너를 돌리는 데 충분하다.

### 2. 컨테이너(Container)에 설치 ###
애플리케이션 실행에 필요한 환경은 모두 컨테이너 안에 넣는다.
* CUDA 툴킷 (CUDA Toolkit): nvcc 컴파일러, 라이브러리(cuBLAS, cuDNN 등)가 여기에 포함되는데 컨테이너 내부의 앱은 이 라이브러리를 사용한다.
* NCCL (NVIDIA Collective Communications Library): GPU P2P 통신을 담당하는 핵심 라이브러리로 보통 PyTorch나 TensorFlow 이미지 안에 내장되어 있다. P2P 통신 관점에서는 호스트의 드라이버가 NVLink/PCIe P2P를 지원하는 상태여야 하고, 컨테이너 내부의 NCCL이 이를 활용하도록 설정(NCCL_P2P_DISABLE=0)되어야 한다.
* 딥러닝 프레임워크: PyTorch, TensorFlow, JAX 등

### 'CUDA 드라이버' vs 'CUDA 툴킷'의 차이 ###
* CUDA 드라이버 (libcuda.so 등): 호스트에 설치된 NVIDIA 드라이버에 포함되어 있다. GPU와 직접 대화하는 역할을 하며, 호스트에만 존재해야 한다.
* CUDA 툴킷 (nvcc, libcudart.so 등): 개발 도구와 라이브러리 모음으로, 컨테이너 내부의 앱이 이 툴킷을 통해 명령을 내리면 호스트의 드라이버가 이를 전달받아 GPU를 구동한다.   
  * nvcc: CUDA C/C++ 컴파일러 (툴킷의 핵심)
  * nvprof / nsys: 성능 분석(Profiling) 도구
  * cuBLAS, cuDNN: 딥러닝 연산 가속 라이브러리

## NVIDIA Container Toolkit ##
NVIDIA Container Toolkit은 호스트 OS에 설치된 NVIDIA GPU 드라이버와 컨테이너(Docker, K8s) 사이를 연결해 주는 '다리' 역할을 하는 소프트웨어 패키지이다.
일반적인 컨테이너(Docker 등)는 호스트의 하드웨어와 격리되어 있다. 특히 GPU는 복잡한 커널 드라이버가 필요한데, 컨테이너 안에 무거운 드라이버를 통째로 넣을 수는 없다.
이때 컨테이너 툴킷이 있으면, 컨테이너가 실행될 때 호스트의 GPU 드라이버 자원(라이브러리, 실행 파일 등)을 컨테이너 내부로 자동으로 마운트(연결)해 준다.
컨테이너 내 프로세스가 호스트의 GPU를 내 것처럼 인식하게 하고, 호스트의 libcuda.so 같은 핵심 라이브러리와 nvidia-smi 같은 도구를 컨테이너 안으로 노출시켜 준다.
GPU 간 P2P 통신에 필요한 장치 노드(/dev/nvidiactl 등)를 컨테이너가 사용할 수 있도록 권한을 부여해 준다.

* nvidia-container-cli: 실제 컨테이너 안에 GPU 장치를 넣어주는 하위 레벨 도구
* nvidia-container-runtime: Docker 같은 런타임이 GPU를 사용할 수 있게 중간에서 가로채서 처리해 주는 엔진
* libnvidia-container: 하드웨어 리소스 처리를 위한 라이브러리

#### 설치 순서 ####
* 호스트에 NVIDIA 드라이버 설치
* 호스트에 Docker 또는 Containerd 설치
* 호스트에 NVIDIA Container Toolkit 설치
* 컨테이너 실행 (예: docker run --gpus all ...) 


## 커스텀 컨테이너 이미지 빌드 ##


### 1. 기반 이미지 선택 ###

1. 먼저 호스트 서버에서 nvidia-smi 를 이용하여 CUDA 버전을 확인한다.   

2. 특정 CUDA 버전에 맞는 태그 찾기. 
  * NGC 페이지의 "Release Notes" 또는 "Tags" 탭에서 상세 스펙을 확인한다.
  * NGC PyTorch 컨테이너 페이지에 접속한다.
  * 각 태그 설명에 포함된 Framework 및 CUDA 버전을 확인한다. 
    * 예: 24.10 태그는 보통 CUDA 12.6, Python 3.12, PyTorch 2.5 정도를 포함하고 있다.
    * 2025년형 최신 GPU(Blackwell 등)를 쓰신다면 반드시 25.xx 시리즈를 선택해야 최적화된 버전을 사용할 수 있다. 

3. NGC 이미지는 이미 최적화된 NCCL, cuDNN, TransformerEngine 등을 포함하고 있으므로, pip install torch를 실행할 필요는 없다.

### 2. 이미지 만들기 ###
[dockerfile]
```
# 1. 최적화된 NVIDIA 공식 이미지 사용 (CUDA, cuDNN, NCCL 포함)
FROM nvcr.io/nvidia/pytorch:24.12-py3
LABEL maintainer="gnosia93@naver.com"

# 파이썬 출력 속도 향상 및 NCCL 디버깅 활성화
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NCCL_DEBUG=INFO \
    NCCL_P2P_DISABLE=0 

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 5. 파이썬 라이브러리 설치 (캐시 활용을 위해 requirements.txt 먼저 복사)
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 작업 소스 코드 복사
# COPY . .
# 7. 실행 권한 부여 및 진입점 설정 (선택 사항)
# CMD ["python", "train.py"]

```

[requirements.txt]
```
# 핵심 프레임워크 (이미지에 포함되어 있으나 버전 고정을 위해 명시)
torch>=2.4.0
torchvision
torchaudio

# Hugging Face 생태계 (Llama-3 학습 필수)
transformers>=4.40.0
datasets>=2.19.0
accelerate>=0.30.0
evaluate
tokenizers>=0.19.0

# 분산 학습 및 최적화
deepspeed>=0.14.0
mpi4py                      # Multi-node 통신 보조

# 데이터 처리 및 유틸리티
sentencepiece               # Llama 토크나이저 대응
protobuf
numpy<2.0.0                 # PyTorch 호환성을 위해 1.x 유지 권장
pandas
tqdm
pyyaml

# AWS 및 클라우드 환경
boto3                       # S3/FSx 연동용
fsspec>=2024.3.1
huggingface_hub             # Llama-3 게이트 모델 인증용

# 성능 모니터링 (필요 시 선택)
psutil
py-cpuinfo

# 플래시 어탠서2 커널
flash-attn
```

도커 이미지를 만들어서 ecr 에 푸시한다.
```
aws ecr create-repository --repository-name my-dl-repo --region ${AWS_REGION}
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS \
    --password-stdin 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com
docker build -t my-dl-image .
docker tag my-dl-image:latest 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com
```   

