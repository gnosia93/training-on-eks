## FSDP ##
FSDP는 AI 분산 학습에서 사용되는 Fully Sharded Data Parallel 기술로, 모델의 가중치(파라미터), 그래디언트, 옵티마이저 상태를 여러 GPU에 걸쳐 '분할'하여 저장함으로써 메모리 효율성을 극대화하고 매우 큰 모델을 학습할 수 있게 해줍니다. 기존의 DistributedDataParallel (DDP)와 달리 모델 전체를 각 GPU 마다 복제하지 않고 필요한 부분만 가져와 사용하기 때문에 GPU 메모리 부족 문제를 해결하는 데 효과적이다. Meta(이전 Facebook AI Research, FAIR)의 FairScale 팀에 의해 개발되었습니다. 
DeepSpeed의 영향을 받았으며 PyTorch 프레임워크에 네이티브 기능으로 통합되었다. 텐서 및 파이트 라인 페러랠을 사용하는 NVIDIA Megatron 에 비해 노드간의 집합 통신량이 많다. 

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/fsdp-arch.webp)

## 훈련하기 ##

#### [t5-small 60M 모델](https://huggingface.co/google-t5/t5-small) 훈련 ####
  
training-on-eks 으로 디렉토리로 이동한 후 pytorch fsdp 작업을 실행한다
```
git clone https://github.com/gnosia93/training-on-eks.git
cd training-on-eks

kubectl create ns pytorch
kubectl apply -k kustomize/overlays/fsdp/
```

pytorchjob 을 조회한다.  
```
kubectl get pytorchjob -n pytorch
kubectl get all -n pytorch
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

![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-gpu-dashboard.png)


#### pytorchjob 자동 재시작 ####

JOB은 backoffLimit 값까지 재시작된다. 파트는 동일하나 컨테이너가 재시작하는 것으로 생각하면 되는데 기본값은 6회 이다. 6번 재시작했는데도 작업이 오류가 발생되면, CrashLoopBackOff 상태로 빠지고 더이상 컨테이너를 재생성하지 않는다. 
```
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "t5-fsdp-training"
  namespace: "pytorch"
spec:
  # 이 부분이 재시도 횟수 설정입니다.
  backoffLimit: 6  
  
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure # 실패 시 재시작 정책
      template:
        spec:
          containers:
```

#### Job 삭제 ####
실패한 JOB은 명시적으로 삭제 시켜줘야한다. 
```
kubectl delete pytorchjob pytorch-dist-job -n pytorch
```

## 레퍼런스 ##

* [DP / DDP / FSDP 간단 비교](https://velog.io/@kaiba0514/DP-DDP-FSDP-%EA%B0%84%EB%8B%A8-%EB%B9%84%EA%B5%90)
* [다중 GPU를 효율적으로 사용하는 방법: DP부터 FSDP까지](https://medium.com/tesser-team/%EB%8B%A4%EC%A4%91-gpu%EB%A5%BC-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-dp%EB%B6%80%ED%84%B0-fsdp%EA%B9%8C%EC%A7%80-3057d31150b6)
