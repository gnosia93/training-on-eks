## EFA 테스트 하기 ##

현재 EKS 오토모드는 EFA(Elastic Fabric Adapter)를 지원하지 않는다 ?. 클러스터 생성시 EKS 관리형 노드 그룹이나 자체 관리형 노드를 선택해야 한다. 
이번 워크샵에서는 ENA 를 이용하여 분산 훈련을 테스트해 볼 예정이다.

### 필수 전제 조건 요약 (Pod 배포 전 완료되어야 함): ###
* EKS 모드: EKS 관리형 노드 그룹 또는 자체 관리형 노드를 사용해야 합니다 (Fargate 불가).
#### 2. 인스턴스 유형: EFA를 지원하는 GPU 인스턴스 유형 ####
```
aws ec2 describe-instance-types \
    --filters Name=network-info.efa-supported,Values=true \
    --query "InstanceTypes[?GpuInfo.Gpus!=null].InstanceType" \
    --output text | sort
```
[결과]
```
g5.16xlarge     p4d.24xlarge
g5.8xlarge      g4dn.8xlarge    g6e.24xlarge    gr6.8xlarge     g5.48xlarge
g6.24xlarge     g6e.48xlarge    g6e.8xlarge
g6.48xlarge     g4dn.metal      g6e.16xlarge    g4dn.12xlarge   g6.8xlarge      g5.24xlarge
g6e.12xlarge    g5.12xlarge
p5en.48xlarge   g6.16xlarge     g6.12xlarge     g4dn.16xlarge
```
* 디바이스 플러그인 배포: 클러스터에 aws-efa-k8s-device-plugin이 DaemonSet으로 배포되어 실행 중이어야 합니다. 이 플러그인이 aws.amazon.com 리소스를 노출시킵니다.
#### 4. [DLC 이미지](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/appendix-dlc-release-notes-pytorch.html) 에서 헤딩 이미지를 찾는다. ####
  

```
apiVersion: v1
kind: Pod
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

* [Getting Started with Karpenter](https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/)
* [EKS Auto Mode에 대해서](https://devops-james.tistory.com/m/514#:~:text=AWS%EC%97%90%EC%84%9C%20%EA%B4%80%EB%A6%AC%20%2D%20SSH%EC%99%80%20SSM%20%EC%97%91%EC%84%B8%EC%8A%A4%EA%B9%8C%EC%A7%80%20%EA%B8%88%EC%A7%80%ED%95%98%EA%B3%A0,pod%EB%93%A4%EC%9D%B4%20%EC%97%86%EC%8A%B5%EB%8B%88%EB%8B%A4.%20%2D%20GPU%EB%8F%84%20%EB%82%B4%EC%9E%A5%20%ED%94%8C%EB%9F%AC%EA%B7%B8%EC%9D%B8%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%A9%EB%8B%88%EB%8B%A4.)
