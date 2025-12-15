## EBS CSI 설치 ##
```
helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver/
helm repo update

helm install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver \
    --namespace kube-system \
    --set enableVolumeScheduling=true
```
enableVolumeScheduling=true를 설정하면, 쿠버네티스는 볼륨이 생성되는 즉시 해당 볼륨이 속한 가용 영역을 파악하고, 동일한 가용 영역에 있는 노드에만 파드를 배포하도록 지시한다.


## [Prometheus Stack 설치](https://github.com/prometheus-operator/kube-prometheus) ##
```
helm repo add prometheus https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus/kube-prometheus-stack --namespace monitoring --create-namespace

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

DCGM 은 GPU가 탑재된 모든 노드에 데몬 형태로 설치된다. 노드가 신규로 생성되면 DCGM은 이를 감지하고 해당 노드에 파드를 스케줄링 한다. 
아래 custome-values.yaml 에서 보는 바와 같이 DCGM이 제대로 설치되기 위해서는 노드 라벨 설렉터와 tolerations이 필요하다.
```
helm repo add nvidia https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

# custom-values.yaml 파일 내용
cat <<EOF > custom-values.yaml
nodeSelector:
  karpenter.k8s.aws/instance-gpu-manufacturer: "nvidia"

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
EOF

helm install dcgm-exporter nvidia/dcgm-exporter -n dcgm \
  --create-namespace \
  -f custom-values.yaml

kubectl get all -n dcgm
```
설치가 완료되면, DCGM Exporter는 쿠버네티스 노드의 GPU 메트릭을 **metrics** 라는 이름의 Prometheus 엔드포인트로 노출하기 시작합니다 (기본 포트: 9400).
이제 Prometheus 서버가 이 엔드포인트를 **스크랩(scrape)** 하도록 설정해야 합니다.
Prometheus Operator 사용 시 자동으로 ServiceMonitor 리소스나 PodMonitor 리소스를 감지하여 DCGM Exporter 서비스를 스크랩 대상에 추가하도록 구성할 수 있습니다.

* 마지막으로 Grafana에서 위에서 언급한 NVIDIA DCGM Exporter Dashboard (ID: 12239)를 가져오면 시각화가 완료됩니다. 

## todo ##
* 프로메테우스 및 그라파나 설정 - EBS (CSI 설치 필요)
* DCGM exporter 엔드포인트 스크랩 by 프로메테우스
* 그라파나 대시보드 번호 설정.


## 참고 - Helm 차트 명령어 ##
* 차트 옵션 보기
```
helm show values nvidia/dcgm-exporter
```
* 설치된 차트 조회
``` 
helm list -A
```
[결과]
```
NAME                            NAMESPACE       REVISION        UPDATED                                 STATUS          CHART                           APP VERSION
dcgm-exporter-1765815921        dcgm            2               2025-12-15 16:35:32.536365618 +0000 UTC deployed        dcgm-exporter-4.7.1             4.7.1      
karpenter                       karpenter       1               2025-12-15 08:20:56.942361288 +0000 UTC deployed        karpenter-1.8.1                 1.8.1      
kube-prometheus-stack           monitoring      1               2025-12-15 16:04:23.410703124 +0000 UTC deployed        kube-prometheus-stack-80.4.1    v0.87.1    
nvdp                            nvidia          1               2025-12-15 11:07:53.172145499 +0000 UTC deployed        nvidia-device-plugin-0.18.0     0.18.0     
```

## 레퍼런스 ##
* [Tracking GPU Usage in K8s with Prometheus and DCGM: A Complete Guide](https://medium.com/@penkow/tracking-gpu-usage-in-k8s-with-prometheus-and-dcgm-a-complete-guide-7c8590809d7c)
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
* https://github.com/NVIDIA/dcgm-exporter
