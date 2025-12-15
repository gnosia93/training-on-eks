## [Prometheus Stack 설치](https://github.com/prometheus-operator/kube-prometheus) ##
```
helm repo add prometheus https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prometheus-stack prometheus/kube-prometheus-stack --namespace monitoring --create-namespace

kubectl get pods -l "release=kube-prometheus-stack" --namespace monitoring
```

#### Get Grafana 'admin' user password by running: ####
```
kubectl --namespace monitoring get secrets kube-prometheus-stack-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo
```

#### grafana ####
```
kubectl --namespace monitoring get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=kube-prometheus-stack" -oname

kubectl get secret --namespace monitoring -l app.kubernetes.io/component=admin-secret -o jsonpath="{.items[0].data.admin-password}" | base64 --decode ; echo
```


## NVIDIA DCGM(Data Center GPU Manager) 설치 ##
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/dcgm-exporter.png)
```
helm repo add nvidia https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

# custom-values.yaml 파일 내용
cat <<EOF > custom-values.yaml
nodeSelector:
  karpenter.k8s.aws/instance-gpu-manufacturer: "nvidia"
EOF

helm install --generate-name nvidia/dcgm-exporter -n dcgm \
  --create-namespace \
  -f custom-values.yaml

kubectl get all -n dcgm
```

```
helm show values nvidia/dcgm-exporter
```

* 설치가 완료되면, DCGM Exporter는 쿠버네티스 노드의 GPU 메트릭을 **metrics**라는 이름의 Prometheus 엔드포인트로 노출하기 시작합니다 (기본 포트: 9400).
* 이제 Prometheus 서버가 이 엔드포인트를 **스크랩(scrape)**하도록 설정해야 합니다.
* Prometheus Operator 사용 시:
  * Prometheus Operator가 자동으로 ServiceMonitor 리소스나 PodMonitor 리소스를 감지하여 DCGM Exporter 서비스를 스크랩 대상에 추가하도록 구성할 수 있습니다.

* 마지막으로 Grafana에서 위에서 언급한 NVIDIA DCGM Exporter Dashboard (ID: 12239)를 가져오면 시각화가 완료됩니다. 

## DCGM 트러블 슈팅 ##

* Verify GPU Node and Drivers: Ensure that the Kubernetes node where the dcgm-exporter pod is scheduled actually has an NVIDIA GPU and that the appropriate NVIDIA drivers are installed and functional. You can log into the host node and run nvidia-smi to confirm this.

* Check NVIDIA Device Plugin: The NVIDIA device plugin is essential for making GPUs and their libraries visible to Kubernetes pods. Ensure it is deployed and running correctly as a DaemonSet across all your GPU nodes. The recommended way to deploy it is via Helm:

* Confirm Pod Scheduling: Make sure the dcgm-exporter pod is only scheduled on nodes that have GPUs. The daemonset configuration usually handles this automatically with appropriate node selectors, but you can verify with kubectl describe pod dcgm-exporter-xxxxx -n monitoring. The pod will go into a CrashLoopBackOff state if scheduled on a non-GPU node with no drivers.

* Check Driver/Library Versions: In some cases, a version mismatch between the host's NVIDIA driver and the DCGM library version in the container can cause issues. Ensure compatibility or consider deploying the entire NVIDIA GPU Operator which manages the installation and compatibility of all necessary components (drivers, device plugin, DCGM-exporter).

* Verify Container Runtime Access: The configuration of the container runtime (containerd, Docker, etc.) must allow access to the NVIDIA libraries on the host (like libnvidia-ml.so.1). The NVIDIA Container Toolkit usually handles this configuration. 



## 레퍼런스 ##
* [Tracking GPU Usage in K8s with Prometheus and DCGM: A Complete Guide](https://medium.com/@penkow/tracking-gpu-usage-in-k8s-with-prometheus-and-dcgm-a-complete-guide-7c8590809d7c)
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
* https://github.com/NVIDIA/dcgm-exporter
