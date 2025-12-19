## FSDP ##
FSDP는 AI 분산 학습에서 사용되는 Fully Sharded Data Parallel 기술로, 모델의 가중치(파라미터), 그래디언트, 옵티마이저 상태를 여러 GPU에 걸쳐 '분할'하여 저장함으로써 메모리 효율성을 극대화하고 매우 큰 모델을 학습할 수 있게 해줍니다. 기존의 DistributedDataParallel (DDP)와 달리 모델 전체를 각 GPU 마다 복제하지 않고 필요한 부분만 가져와 사용하기 때문에 GPU 메모리 부족 문제를 해결하는 데 효과적이다. Meta(이전 Facebook AI Research, FAIR)의 FairScale 팀에 의해 개발되었습니다. 
DeepSpeed의 영향을 받았으며 PyTorch 프레임워크에 네이티브 기능으로 통합되었다. 텐서 및 파이트 라인 페러랠을 사용하는 NVIDIA Megatron 에 비해 노드간의 집합 통신량이 많다. 

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/fsdp-arch.webp)

## 훈련하기 ##

#### [t5-small 모델 훈련](https://github.com/gnosia93/training-on-eks/blob/main/samples/fsdp/t5-fsdp.py) ####
  
training-on-eks 으로 디렉토리로 이동한 후 pytorch fsdp 작업을 실행한다 (4 pods, 데이터 건수 1000건, epoch 1)
```
git clone https://github.com/gnosia93/training-on-eks.git
cd /home/ec2-user/training-on-eks/kustomize/overlays/fsdp

kubectl kustomize .
kubectl kustomize . | kubectl apply -f -
```

pytorchjob 을 조회한다.  
```
kubectl get pytorchjob
kubectl get all 
```
[결과]
```
NAME               STATE     AGE
pytorch-dist-job   Created   3m59s

NAME                            READY   STATUS              RESTARTS   AGE
pod/pytorch-dist-job-master-0   0/1     ContainerCreating   0          2m56s
pod/pytorch-dist-job-worker-0   0/1     Init:0/1            0          2m56s
pod/pytorch-dist-job-worker-1   0/1     Init:0/1            0          2m56s
pod/pytorch-dist-job-worker-2   0/1     Init:0/1            0          2m56s

NAME                                TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)     AGE
service/pytorch-dist-job-master-0   ClusterIP   None         <none>        23456/TCP   2m57s
service/pytorch-dist-job-worker-0   ClusterIP   None         <none>        23456/TCP   2m57s
service/pytorch-dist-job-worker-1   ClusterIP   None         <none>        23456/TCP   2m57s
service/pytorch-dist-job-worker-2   ClusterIP   None         <none>        23456/TCP   2m57s
```

실시간 로그를 확인한다. (-f) 
```
kubectl logs -f pod/pytorch-dist-job-master-0 -n pytorch
```

#### 카펜터 노드 프로비저닝 확인 ####
```
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter
```
```
{"level":"INFO","time":"2025-12-19T08:46:17.537Z","logger":"controller","message":"created nodeclaim","commit":"1ad0d78","controller":"provisioner","namespace":"","name":"","reconcileID":"5f3acae3-d315-4da0-9164-a622db9b5e17","NodePool":{"name":"gpu"},"NodeClaim":{"name":"gpu-845rb"},"requests":{"cpu":"465m","memory":"590Mi","nvidia.com/gpu":"4","pods":"11"},"instance-types":"g4dn.12xlarge, g4dn.metal, g5.12xlarge, g5.24xlarge, g5.48xlarge and 10 other(s)"}
{"level":"INFO","time":"2025-12-19T08:46:20.354Z","logger":"controller","message":"launched nodeclaim","commit":"1ad0d78","controller":"nodeclaim.lifecycle","controllerGroup":"karpenter.sh","controllerKind":"NodeClaim","NodeClaim":{"name":"gpu-845rb"},"namespace":"","name":"gpu-845rb","reconcileID":"689003c4-f700-4b71-aa42-24591fbebe4e","provider-id":"aws:///ap-northeast-2d/i-0379b9b05def0f920","instance-type":"g6.12xlarge","zone":"ap-northeast-2d","capacity-type":"spot","allocatable":{"cpu":"47810m","ephemeral-storage":"269Gi","memory":"187596052Ki","nvidia.com/gpu":"4","pods":"234","vpc.amazonaws.com/pod-eni":"114"}}
{"level":"INFO","time":"2025-12-19T08:47:36.854Z","logger":"controller","message":"registered nodeclaim","commit":"1ad0d78","controller":"nodeclaim.lifecycle","controllerGroup":"karpenter.sh","controllerKind":"NodeClaim","NodeClaim":{"name":"gpu-845rb"},"namespace":"","name":"gpu-845rb","reconcileID":"01bf0192-3ee3-42bc-a79b-cb4d87e5f206","provider-id":"aws:///ap-northeast-2d/i-0379b9b05def0f920","Node":{"name":"ip-10-0-7-27.ap-northeast-2.compute.internal"}}
{"level":"INFO","time":"2025-12-19T08:48:37.858Z","logger":"controller","message":"initialized nodeclaim","commit":"1ad0d78","controller":"nodeclaim.lifecycle","controllerGroup":"karpenter.sh","controllerKind":"NodeClaim","NodeClaim":{"name":"gpu-845rb"},"namespace":"","name":"gpu-845rb","reconcileID":"00f80e60-fda7-4699-bc0a-7ce8b2ba11f6","provider-id":"aws:///ap-northeast-2d/i-0379b9b05def0f920","Node":{"name":"ip-10-0-7-27.ap-northeast-2.compute.internal"},"allocatable":{"cpu":"47810m","ephemeral-storage":"288764809146","hugepages-1Gi":"0","hugepages-2Mi":"0","memory":"187596060Ki","nvidia.com/gpu":"4","pods":"234"}}
```

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-gpu-dashboard-1.png)

#### Job 삭제 #### 
```
kubectl delete pytorchjob pytorch-dist-job -n pytorch
```

## kustmize yaml 구성 설명 ##
멀티 GPU를 가진 한대의 서버에 모든 파드를 스케줄링 시키기 위해서, 마스터에는 노드 어피니티만 워커에는 노드와 파드 어피니티 모두를 사용하고 있다. 
하지만 특정 인스턴스 타입이 사전에 정의된 경우는 어피니티를 사용하는 대신에 NodeSelector 사용하는 것이 훨씬 쉽고 간결하다. 
```
# overlays/custom-url/kustomization.yaml

# 베이스 파일 지정
resources:
- ../../base

# 패치 적용
patches:
# Master 스펙의 args 필드를 덮어씁니다.
# << 주의 >>
# 마스터파드 및 워커파드간의 Node 안착 교착 상태를 방지하기 위해서 마스터에는 노드 어피니티만을 워커에는 노드와 파드 어피니티 모두를 설정한다.
# 이렇게 하지 않는 경우 노드풀에 GPU 4장 짜리 서버가 없으면, 노드 프로비저닝시 파드들이 교착상태에 빠져서 무기한 pending 상태로 빠진다. 
- patch: |-
    - op: add
      path: /spec/pytorchReplicaSpecs/Master/template/spec/affinity
      value:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: node.kubernetes.io/instance-type
                    operator: In
                    values:
                      # P 시리즈 (최고 성능, 인트라 노드 통신 최적화)
                      - p4d.24xlarge   # A100 8장
                      - p4de.24xlarge  # A100 8장 (메모리 확장형)
                      - p5.48xlarge    # H100 8장
                
                      # G 시리즈 (가성비 추론/학습용)
                      - g4dn.12xlarge  # T4 4장
                      - g4dn.metal     # T4 8장
                      - g5.12xlarge    # A10G 4장
                      - g5.48xlarge    # A10G 8장
                      - g6.12xlarge    # L4 4장 (2025년 최신 가성비)
                      - g6.48xlarge    # L4 8장
                
                      # 최신 G6e 시리즈 (H100 기반 가성비 모델)
                      - g6e.12xlarge   # L40S 4장
                      - g6e.48xlarge   # L40S 8장     
    - op: replace
      path: /spec/pytorchReplicaSpecs/Master/template/spec/containers/0/args/0
      value: |
        git clone https://github.com/gnosia93/training-on-eks /workspace/code
        cd /workspace/code/samples/fsdp
        echo "working directory: "$(pwd)
        pip install -r requirements.txt
        torchrun --nnodes 4 --nproc_per_node 1 t5-fsdp.py
  target:
    kind: PyTorchJob
    name: pytorch-dist-job

# Worker 스펙의 args 필드를 덮어씁니다.
- patch: |-
    - op: add
      path: /spec/pytorchReplicaSpecs/Worker/template/spec/affinity
      value:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: node.kubernetes.io/instance-type
                    operator: In
                    values:
                      # P 시리즈 (최고 성능, 인트라 노드 통신 최적화)
                      - p4d.24xlarge   # A100 8장
                      - p4de.24xlarge  # A100 8장 (메모리 확장형)
                      - p5.48xlarge    # H100 8장
                
                      # G 시리즈 (가성비 추론/학습용)
                      - g4dn.12xlarge  # T4 4장
                      - g4dn.metal     # T4 8장
                      - g5.12xlarge    # A10G 4장
                      - g5.48xlarge    # A10G 8장
                      - g6.12xlarge    # L4 4장 (2025년 최신 가성비)
                      - g6.48xlarge    # L4 8장
                
                      # 최신 G6e 시리즈 (H100 기반 가성비 모델)
                      - g6e.12xlarge   # L40S 4장
                      - g6e.48xlarge   # L40S 8장
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: training.kubeflow.org/job-name
                    operator: In
                    values:
                      - pytorch-dist-job
              topologyKey: kubernetes.io/hostname 
    - op: replace
      path: /spec/pytorchReplicaSpecs/Worker/template/spec/containers/0/args/0
      value: |
        git clone https://github.com/gnosia93/training-on-eks /workspace/code
        cd /workspace/code/samples/fsdp
        echo "working directory: "$(pwd)
        pip install -r requirements.txt
        torchrun --nnodes 4 --nproc_per_node 1 t5-fsdp.py
    - op: replace
      path: /spec/pytorchReplicaSpecs/Worker/replicas
      value: 3        
  target:
    kind: PyTorchJob
    name: pytorch-dist-job
```

```
$ kubectl get nodes -L nvidia.com/gpu.machine
NAME                                            STATUS   ROLES    AGE    VERSION               GPU.MACHINE
ip-10-0-3-244.ap-northeast-2.compute.internal   Ready    <none>   4d1h   v1.34.2-eks-ecaa3a6   
ip-10-0-6-88.ap-northeast-2.compute.internal    Ready    <none>   4d1h   v1.34.2-eks-ecaa3a6   
ip-10-0-7-177.ap-northeast-2.compute.internal   Ready    <none>   10m    v1.34.2-eks-ecaa3a6   g6.12xlarge
```

## 레퍼런스 ##

* [DP / DDP / FSDP 간단 비교](https://velog.io/@kaiba0514/DP-DDP-FSDP-%EA%B0%84%EB%8B%A8-%EB%B9%84%EA%B5%90)
* [다중 GPU를 효율적으로 사용하는 방법: DP부터 FSDP까지](https://medium.com/tesser-team/%EB%8B%A4%EC%A4%91-gpu%EB%A5%BC-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-dp%EB%B6%80%ED%84%B0-fsdp%EA%B9%8C%EC%A7%80-3057d31150b6)
