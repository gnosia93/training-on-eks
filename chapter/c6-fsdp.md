## FSDP ##
FSDP는 AI 분산 학습에서 사용되는 Fully Sharded Data Parallel 기술로, 모델의 가중치(파라미터), 그래디언트, 옵티마이저 상태를 여러 GPU에 걸쳐 '분할'하여 저장함으로써 메모리 효율성을 극대화하고 매우 큰 모델을 학습할 수 있게 해줍니다. 기존의 DistributedDataParallel (DDP)와 달리 모델 전체를 각 GPU 마다 복제하지 않고 필요한 부분만 가져와 사용하기 때문에 GPU 메모리 부족 문제를 해결하는 데 효과적이다. Meta(이전 Facebook AI Research, FAIR)의 FairScale 팀에 의해 개발되었습니다. 
DeepSpeed의 영향을 받았으며 PyTorch 프레임워크에 네이티브 기능으로 통합되었다. 텐서 및 파이트 라인 페러랠을 사용하는 NVIDIA Megatron 에 비해 노드간의 집합 통신량이 많다. 

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/fsdp-arch.webp)

## 훈련하기 ##

#### t5-small 모델 훈련 ####
  
training-on-eks 으로 디렉토리로 이동한 후 pytorch fsdp 작업을 실행한다 (p4d.24xlarge, 4 pods, 데이터 건수 1000건, epoch 1)
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

## 레퍼런스 ##

* [DP / DDP / FSDP 간단 비교](https://velog.io/@kaiba0514/DP-DDP-FSDP-%EA%B0%84%EB%8B%A8-%EB%B9%84%EA%B5%90)
* [다중 GPU를 효율적으로 사용하는 방법: DP부터 FSDP까지](https://medium.com/tesser-team/%EB%8B%A4%EC%A4%91-gpu%EB%A5%BC-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-dp%EB%B6%80%ED%84%B0-fsdp%EA%B9%8C%EC%A7%80-3057d31150b6)
