Slinky 프로젝트는 Slurm의 개발사인 SchedMD가 직접 주도하여 만든 오픈소스 툴킷으로, 2025년 기준 EKS에서 Slurm을 운영하는 가장 발전된 방식입니다. 
이 프로젝트의 핵심은 Slurm의 강력한 스케줄링 능력(HPC용)과 Kubernetes의 유연한 인프라 관리 능력을 하나로 합치는 데 있습니다. 

### 1. 주요 구성 요소 ###
Slinky는 단순히 데몬을 띄우는 것을 넘어, Kubernetes 네이티브하게 작동하기 위해 여러 프로젝트로 나뉩니다: 
* Slurm-operator: Slurm 클러스터의 전체 라이프사이클을 관리합니다. SlurmCluster와 같은 커스텀 리소스(CRD)를 사용하여 EKS 위에 Slurm 인프라를 자동으로 배포하고 관리합니다.
* Slurm-bridge: Slurm을 Kubernetes의 스케줄러처럼 작동하게 만듭니다. 이를 통해 sbatch로 제출된 잡뿐만 아니라 일반 Kubernetes 파드도 Slurm의 우선순위 정책에 따라 스케줄링할 수 있습니다.
* Slurm-client: Slurm REST API와 통신하기 위한 라이브러리로, 다른 구성 요소들이 Slurm 상태를 실시간으로 확인하고 제어할 수 있게 합니다. 

### 2. Slinky만의 차별점 ###
* 동적 노드 세트(NodeSets): Slurm 노드들을 NodeSet이라는 단위로 관리하며, 필요에 따라 개수를 동적으로 조절할 수 있습니다.
* 오토스케일링 연동: 대기 중인 잡(Pending Jobs)이 생기면 Slinky가 이를 감지하고, EKS의 Karpenter나 HPA와 연동하여 실제 GPU 인스턴스를 추가로 생성합니다.
* 하이브리드 환경 지원: 모든 계산 리소스를 EKS에 둘 필요가 없습니다. 일부는 EKS 내 파드로, 일부는 외부 물리 서버(Bare-metal)로 구성하여 하나의 Slurm 클러스터로 묶어 관리할 수 있습니다. 

### 3. 설치 요구 사항 ###
* Kubernetes: v1.29 이상
* Slurm: 25.11 이상
* Cgroup: v2 지원 환경

### 4. 실제 도입 시나리오 ###
Slinky를 도입하면 연구자는 기존과 똑같이 sbatch 명령어로 AI 모델 학습을 던지지만, 인프라 관리자는 별도의 Slurm 전용 서버 없이 EKS 콘솔 하나로 모든 서비스와 학습 자원을 통합 관리하게 됩니다. 
더 자세한 아키텍처 다이어그램이나 기술 문서는 Slinky 공식 GitHub에서 확인할 수 있습니다

## 레퍼런스 ##
* [Running Slurm on Amazon EKS with Slinky](https://aws.amazon.com/ko/blogs/containers/running-slurm-on-amazon-eks-with-slinky/#:~:text=The%20Slinky%20Project%20is%20an%20open%20source,NodeSet%20resources%20deployed%20within%20a%20Kubernetes%20environment.)
