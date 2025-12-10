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
