
```
# 1. 최적화된 NVIDIA 공식 이미지 사용 (CUDA, cuDNN, NCCL 포함)
FROM nvcr.io/nvidia/pytorch:24.12-py3
LABEL maintainer="soonbeom@amazon.com"

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

---

2025년 현재, 사용 중인 호스트 시스템의 NVIDIA 드라이버 버전과 GPU 아키텍처(예: H100, A100 등)에 따라 가장 적합한 태그가 달라집니다. 확인하는 구체적인 방법은 다음과 같습니다.

### 1. 호스트의 CUDA 지원 버전 확인 ###
먼저 호스트 서버에서 아래 명령어를 입력하여 지원 가능한 최대 CUDA 버전을 확인합니다.
```
nvidia-smi
```
### 2. NGC Catalog에서 태그 읽는 법 ###
* NVIDIA는 YY.MM (연도.월) 형식의 릴리즈 태그를 사용합니다.
* PyTorch 이미지 주소: nvcr.io/nvidia/pytorch
   * 25.01-py3: 2025년 1월 릴리즈 (가장 최신 기능을 포함하지만, 매우 높은 드라이버 버전 요구 가능성 있음)
   * 24.12-py3: 2024년 12월 릴리즈 (안정 버전)

### 3. 특정 CUDA 버전에 맞는 태그 찾기 ####
* NGC 페이지의 "Release Notes" 또는 "Tags" 탭에서 상세 스펙을 확인해야 합니다.
* NGC PyTorch 컨테이너 페이지에 접속합니다.
* 각 태그 설명에 포함된 Framework 및 CUDA 버전을 확인합니다.
   * 예: 24.10 태그는 보통 CUDA 12.6, Python 3.12, PyTorch 2.5 정도를 포함하고 있습니다.
   * 2025년형 최신 GPU(Blackwell 등)를 쓰신다면 반드시 25.xx 시리즈를 선택해야 최적화 혜택을 받습니다.

### 4. 선택기준 ###
* 최신 성능 지향: nvcr.io/nvidia/pytorch:25.10-py3 (CUDA 12.x 기반 최신 릴리즈)
* 범용 안정성 지향: nvcr.io/nvidia/pytorch:24.12-py3
* 구형 드라이버 환경: 호스트 드라이버가 낮다면 23.xx 시리즈까지 내려가야 할 수도 있습니다.

NGC 이미지는 이미 최적화된 NCCL, cuDNN, TransformerEngine 등을 포함하고 있으므로, pip install torch를 다시 실행하지 마세요. 설치된 최적화 버전이 덮어씌워져 성능이 저하될 수 있습니다.
