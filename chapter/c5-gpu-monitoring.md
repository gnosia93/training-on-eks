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
helm install --generate-name nvidia/dcgm-exporter -n monitoring

kubectl get pods -n monitoring
kubectl get services -n monitoring
```

* 설치가 완료되면, DCGM Exporter는 쿠버네티스 노드의 GPU 메트릭을 **metrics**라는 이름의 Prometheus 엔드포인트로 노출하기 시작합니다 (기본 포트: 9400).
* 이제 Prometheus 서버가 이 엔드포인트를 **스크랩(scrape)**하도록 설정해야 합니다.
#### Prometheus Operator 사용 시: ####
Prometheus Operator가 자동으로 ServiceMonitor 리소스나 PodMonitor 리소스를 감지하여 DCGM Exporter 서비스를 스크랩 대상에 추가하도록 구성할 수 있습니다.
#### 수동 설정 시: ####
Prometheus 설정 파일에 scrape_configs 섹션을 추가하여 dcgm-exporter 서비스의 엔드포인트를 명시적으로 지정해야 합니다.

* 마지막으로 Grafana에서 위에서 언급한 NVIDIA DCGM Exporter Dashboard (ID: 12239)를 가져오면 시각화가 완료됩니다. 




## 레퍼런스 ##
* [Tracking GPU Usage in K8s with Prometheus and DCGM: A Complete Guide](https://medium.com/@penkow/tracking-gpu-usage-in-k8s-with-prometheus-and-dcgm-a-complete-guide-7c8590809d7c)
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
* https://github.com/NVIDIA/dcgm-exporter
