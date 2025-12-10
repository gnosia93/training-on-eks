kubeflow 의 트레이닝만 오퍼레이터만 단독으로 설치한다. 분산 트레이닝을 실행하기 위해서 다른 모듈은 필요하지 않다.  
```
sudo dnf install git -y
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
kubectl get crd | grep pytorchjobs
```

[결과]
```
pytorchjobs.kubeflow.org                        2025-12-10T11:29:59Z
```

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/
