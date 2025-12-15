## EBS CSI 설치 ##
```
eksctl create iamserviceaccount \
    --name ebs-csi-controller-sa \
    --namespace kube-system \
    --cluster ${CLUSTER_NAME} \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
    --override-existing-serviceaccounts \
    --approve

helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver/
helm repo update

helm install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver \
    --namespace kube-system 

kubectl get pod -n kube-system -l "app.kubernetes.io/name=aws-ebs-csi-driver,app.kubernetes.io/instance=aws-ebs-csi-driver"

kubectl get storageclass
```
* 이 StorageClass의 volumeBindingMode가 WaitForFirstConsumer로 설정되어 있어야 합니다. 파드가 생성될 노드의 AZ를 고려하여 볼륨을 생성한다.
만약 volumeBindingMode가 Immediate로 되어 있다면, 아래와 같이 패치 명령어로 수정한다.
```
kubectl patch storageclass [YOUR_STORAGE_CLASS_NAME] -p '{"volumeBindingMode": "WaitForFirstConsumer"}'
```

## [Prometheus Stack 설치](https://github.com/prometheus-operator/kube-prometheus) ##
```
helm repo add prometheus https://prometheus-community.github.io/helm-charts
helm repo update

cat <<EOF > prometheus-values.yaml
# prometheus-values.yaml
prometheus:
  prometheusSpec:
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp2             # kubectl get storageclass 의 NAME 칼럼값
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 300Gi 
EOF

helm install prometheus prometheus/kube-prometheus-stack --create-namespace
    --namespace monitoring \
    -f prometheus-values.yaml
```

```
kubectl --namespace default get pods -l "release=prometheus"

# 그라파나 어드민
kubectl --namespace default get secrets prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo

# 그라파나 파드
kubectl --namespace default get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=prometheus" -oname
kubectl get service | grep grafana
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

* 릴리즈 삭제
```
helm uninstall aws-ebs-csi-driver --namespace kube-system
``` 


## 레퍼런스 ##
* [Tracking GPU Usage in K8s with Prometheus and DCGM: A Complete Guide](https://medium.com/@penkow/tracking-gpu-usage-in-k8s-with-prometheus-and-dcgm-a-complete-guide-7c8590809d7c)
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
* https://github.com/NVIDIA/dcgm-exporter
