### 1. cert-manager 설치 ###
Slurm 컴포넌트 간 보안 통신(TLS)을 위해 필수입니다.
```
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace \
  --set crds.enabled=true
```

### 2. Slinky Slurm CRD / Operator 설치 ###
```
helm install slurm-operator-crds oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
  --namespace slinky --create-namespace
kubectl get crd | grep slinky

helm install slurm-operator oci://ghcr.io/slinkyproject/charts/slurm-operator \
  --namespace slinky --create-namespace

kubectl api-resources | grep slurm
```

### 3. Slurm 클러스터(Pod) 배포 ###
```
# JWT 인증용 키 생성
kubectl create secret generic slurm-jwt-key -n slinky --from-literal=jwt-key=$(openssl rand -base64 32)

# Slurm 내부 통신용 인증 키 (munge.key 역할) 생성
kubectl create secret generic slurm-key -n slinky --from-literal=slurm-key=$(openssl rand -base64 32)

```


EKS 에 슬럼 클러스터 데몬인 slurmctld, slurmd, login 등을 Pod 형태로 배포한다.
```
export SLURM_VERSION="25.11"
export SLURM_CTRL_NODE_NUM=1
export SLURM_WORKER_NODE_NUM=2
export SLURM_LOGIN_NODE_NUM=1
export GPU_PER_NODE=1
export EFA_PER_NODE=1

cat <<EOF > slurm-cluster.yaml
# 1. Slurm Controller (slurmctld)
apiVersion: slinky.slurm.net/v1beta1
kind: Controller
metadata:
  name: slurm-controller
  namespace: slinky
spec:
  # 모든 필드명은 소문자로 시작해야 합니다 (key, name)
  jwtHs256KeyRef:
    name: slurm-jwt-key
    key: jwt-key
  slurmKeyRef:
    name: slurm-key
    key: slurm-key
  template:
    spec:
      containers:
        - name: slurmctld
          image: "ghcr.io/slinkyproject/slurmctld:23.11.10"
---
# 2. Slurm Worker Nodes (slurmd)
apiVersion: slinky.slurm.net/v1beta1
kind: NodeSet
metadata:
  name: gpu-partition
  namespace: slinky
spec:
  replicas: ${SLURM_WORKER_NODE_NUM}
  controllerRef:
    name: slurm-controller
  updateStrategy:
    type: RollingUpdate
  template:
    spec:
      containers:
        - name: slurmd
          image: "ghcr.io/slinkyproject/slurmd:23.11.10"
          resources:
            limits:
              nvidia.com/gpu: "${GPU_PER_NODE}"
              vpc.amazonaws.com/efa: "${EFA_PER_NODE}"
---
# 3. Slurm Login Node
apiVersion: slinky.slurm.net/v1beta1
kind: LoginSet
metadata:
  name: login-node
  namespace: slinky
spec:
  replicas: ${SLURM_LOGIN_NODE_NUM}
  controllerRef:
    name: slurm-controller
  sssdConfRef:
    name: sssd-config
    key: sssd.conf
  template:
    spec:
      containers:
        - name: login
          image: "ghcr.io/slinkyproject/slurmrestd:23.11.10"
EOF
```
```
kubectl apply -f slurm-cluster.yaml
```

### 4. 설치 확인 및 사용 ###
```
kubectl get pods -n slinky --show-labels
```
[결과]
```
NAME                                      READY   STATUS    RESTARTS   AGE   LABELS
slurm-operator-56d865fbc7-g4qvx           1/1     Running   0          17m   app.kubernetes.io/instance=slurm-operator,app.kubernetes.io/managed-by=Helm,app.kubernetes.io/name=slurm-operator,app.kubernetes.io/version=25.11,helm.sh/chart=slurm-operator-1.0.1,pod-template-hash=56d865fbc7
slurm-operator-webhook-57cf4d6d85-28pf4   1/1     Running   0          17m   app.kubernetes.io/instance=slurm-operator,app.kubernetes.io/managed-by=Helm,app.kubernetes.io/name=slurm-operator-webhook,app.kubernetes.io/version=25.11,helm.sh/chart=slurm-operator-1.0.1,pod-template-hash=57cf4d6d85
```





Login 파드 접속:
bash
kubectl exec -it <login-pod-name> -n slinky-system -- /bin/bash
```

```
# 노드 상태 확인
sinfo

# 간단한 작업 제출
srun -N 2 hostname
```

💡 실무 운영을 위한 핵심 팁 (2025년 가이드)

* 공유 스토리지 (필수): Slurm은 모든 파드가 동일한 /home이나 /data를 공유해야 합니다. Amazon FSx for Lustre를 EKS의 PVC로 연결하여 각 파드에 마운트하는 설정을 slurm-cluster.yaml의 volumes 섹션에 반드시 추가해야 합니다.
* 자동 확장 (Karpenter): 워커 노드가 모자랄 때 AWS 인스턴스를 자동으로 띄우고 싶다면, EKS에 Karpenter를 설치하고 Slinky의 NodeSet과 연동하십시오.
* 고속 네트워크: GPU 간 통신(Multi-node training)이 중요하다면, EKS 노드 그룹 생성 시 EFA(Elastic Fabric Adapter) 옵션을 활성화해야 Slurm 환경에서도 최대 성능이 나옵니다








---

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
* https://slinky.schedmd.com/en/latest/
* https://www.schedmd.com/introducing-slinky-slurm-kubernetes/
* [Running Slurm on Amazon EKS with Slinky](https://aws.amazon.com/ko/blogs/containers/running-slurm-on-amazon-eks-with-slinky/#:~:text=The%20Slinky%20Project%20is%20an%20open%20source,NodeSet%20resources%20deployed%20within%20a%20Kubernetes%20environment.)
