본 워크샵에서는 EBS CSI 드라이버를 설치하지 않는다. 실제 프로덕션 환경에서는 EBS CSI 드라이버를 먼저 설치한 후 PVC 위에 프로메테우스 스택을 설치해야 한다.  
## [Prometheus Stack 설치](https://github.com/prometheus-operator/kube-prometheus) ##
```
helm repo add prometheus https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus/kube-prometheus-stack \
    --create-namespace \
    --namespace monitoring 
```
생성된 파드들을 조회한다. 
```
kubectl get pods -l "release=prometheus" -n monitoring 
```
[결과]
```
NAME                                                  READY   STATUS    RESTARTS   AGE
prometheus-kube-prometheus-operator-95f6bb89f-8957b   1/1     Running   0          10m
prometheus-kube-state-metrics-66f9f5bf55-zg5bx        1/1     Running   0          10m
prometheus-prometheus-node-exporter-hp42x             1/1     Running   0          10m
prometheus-prometheus-node-exporter-hs79c             1/1     Running   0          10m
```

#### 0. 프로메테우스 외부 노출 ####
```
kubectl patch svc prometheus-kube-prometheus-prometheus -n monitoring -p '{
  "spec": {
    "type": "LoadBalancer",
    "loadBalancerSourceRanges": ["122.36.213.114/32"]        
  }
}'

kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring | awk '{print $4,$5}'
```


#### 1. 그라파나 서비스 외부 노출 #### 
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/prometheus-grafana.png)

그라파나 서비스를 외부로 노출 시키고, admin 패스워드를 확인후 로그인한다. 서비스의 loadBalancerSourceRanges 필드를 이용하면 출발지 주소를 제한할 수 있다.  
```
kubectl patch svc prometheus-grafana -n monitoring -p '{
  "spec": {
    "type": "LoadBalancer",
    "loadBalancerSourceRanges": ["122.36.213.114/32"]        
  }
}'

kubectl --namespace monitoring get secrets prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo
kubectl get svc -n monitoring | grep prometheus-grafana | awk '{print $4}'
```
[결과]
```
ae286c7ef5ccc461a9565b5cb7863132-369961314.ap-northeast-2.elb.amazonaws.com  
```

#### 2. NVIDIA DCGM Exporter Dashboard (ID: 12239) 설치 ####
좌측 메뉴의 Dashboards로 이동 후 New 버튼을 누르고 팝업창에서 Import를 선택한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-1.png)
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-2.png)
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-3.png)
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-4.png)


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

serviceMonitor:
  enabled: true
  # 중요: 프로메테우스가 찾을 수 있도록 레이블을 추가합니다.
  additionalLabels:
    release: prometheus  # 'kubectl get prometheus -n [네임스페이스]'로 확인한 이름 입력
  interval: 30s
  scrapeTimeout: 10s

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"

dcgmExporter:
  env:
    - name: DCGM_REMOTE_HOSTENGINE_INFO
      value: "unix:///run/nvidia/dcgm.sock"
  extraHostVolumeMounts:
    - name: host-run-nvidia
      hostPath: /run/nvidia
      mountPath: /run/nvidia

# 호스트 소켓 공유를 위한 추가 설정
extraVolumeMounts:
  - name: dcgm-socket
    mountPath: /run/nvidia-dcgm

# 호스트 볼륨 설정 (이름 주의: extraHostVolumes)
extraHostVolumes:
  - name: dcgm-socket
    hostPath: /run/nvidia-dcgm

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 100m
    memory: 256Mi
EOF

helm install dcgm-exporter nvidia/dcgm-exporter -n dcgm \
  --create-namespace \
  -f custom-values.yaml

kubectl get all -n dcgm
```
설치가 완료되면, DCGM Exporter는 쿠버네티스 노드의 GPU 메트릭을 metrics 라는 이름의 Prometheus 엔드포인트로 노출하기 시작한다. (기본 포트: 9400).
이후 Prometheus 서버가 이 엔드포인트를 스크랩(scrape) 하게 된다.

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
