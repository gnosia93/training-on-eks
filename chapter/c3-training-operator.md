```
sudo dnf install git -y

kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```  

## 레퍼런스 ##
* https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/
