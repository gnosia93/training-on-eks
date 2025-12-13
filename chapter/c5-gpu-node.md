
metadata:
  name: efa-gpu-dl-pod
  labels:
    app: dl-training
spec:
  # EKS Auto Mode 사용을 위한 필수 셀렉터
  nodeSelector:
    eks.amazonaws.com/compute-type: auto
    # EFA 지원 인스턴스 선택을 위한 레이블 (AWS가 자동으로 붙여줌)
    # P4, P5 인스턴스 등에서 EFA가 활성화됩니다.
    # 클러스터 구성에 따라 레이블 키가 달라질 수 있습니다.
    # 예시 레이블:
    # networking.eks.amazonaws.com: "true" 

  # GPU 노드에 스케줄링될 수 있도록 톨러레이션 추가
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "gpu-workload"
    operator: "Exists"
    effect: "NoSchedule"

  containers:
  - name: dl-container-efa
    # 위에서 추천한 AWS DLC 이미지 사용 (리전과 태그를 실제 값으로 변경하세요)
    image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
    command: ["/bin/bash", "-c"]
    args:
        - |
          echo "Checking for EFA fabric interface using fi_info..."
          # EFA 활성화 확인 명령어
          fi_info -p efa
          
          # 추가적인 연결 테스트는 여기 아래에 명령어를 추가할 수 있습니다.
          # 예시: /opt/amazon/efa/bin/efa_test.sh
          
          if [ $? -eq 0 ]; then
            echo "EFA interface found successfully."
          else
            echo "Failed to find EFA interface."
          fi
    resources:
      limits:
        # 8개의 NVIDIA GPU 할당 요청
        nvidia.com/gpu: 1 
        # EFA 리소스 할당 요청 (이 리소스 타입은 EFA Device Plugin이 설치되어야 사용 가능)
        # Auto Mode에서는 AWS가 EFA 플러그인 설치를 관리합니다.
        # aws.amazon.com: "1" # 필요한 경우 주석 해제하여 사용
      requests:
        nvidia.com/gpu: 1

    # EFA 사용을 위한 환경 변수 설정 (컨테이너 내 라이브러리 설정)
    env:
    - name: NCCL_DEBUG
      value: "INFO"
    - name: NCCL_ALGO
      value: "RING"
    # AWS Libfabric을 NCCL 네트워크 제공자로 지정
    - name: NCCL_NET
      value: "AWS Libfabric"
    - name: FI_EFA_USE_DEVICE_RDMA
      value: "1"
    - name: FI_PROVIDER
      value: "efa"
    - name: FI_LOG_LEVEL
      value: "INFO"    
```



## 레퍼런스 ##
