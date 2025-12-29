
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
