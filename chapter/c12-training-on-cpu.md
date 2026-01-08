![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/intel-amx.png)
Intel 제온 프로세서 기반의 7i 인스턴스에서 [IPEX(Intel Extension for PyTorch)](https://github.com/intel/intel-extension-for-pytorch)를 활용해 분산 훈련 성능을 극대화하려면, 인텔 4세대 및 5세대 제온 프로세서에 탑재된 [AMX(Advanced Matrix Extensions)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html) 가속 기술을 적극 활용해야 한다. AMX는 bf16 및 int8 데이터 타입의 행렬 연산을 비약적으로 가속하는 전용 엔진으로, IPEX 설정에서 dtype=torch.bfloat16을 활성화하면 GPU의 텐서 코어와 유사한 가속 성능을 CPU 환경에서도 구현할 수 있다. 특히 bf16은 모델 정밀도를 유지하면서도 연산 효율을 높여 분산 훈련 속도를 크게 향상시킨다.
이러한 하드웨어 가속은 인텔의 독자적인 기술이므로, AWS Graviton(m7g)이나 AMD(m7a) 인스턴스에서는 사용이 불가능하다. 

* Intel은 IPEX 대신 PyTorch 2.5 이상의 공식 버전을 사용할 것을 권장하며, 최신 PyTorch에는 Intel CPU/GPU 최적화 기능이 통합되어 있다. 이전 버전에서 필요한 import intel_extension_for_pytorch 및 ipex.optimize 관련 코드는 이제 불필요해 졌다. 

## efa 지원 인스턴스 조회 ##
```
 aws ec2 describe-instance-types \
    --filters "Name=instance-type,Values=*7i*" \
              "Name=network-info.efa-supported,Values=true" \
    --query "InstanceTypes[].[InstanceType, VCpuInfo.DefaultVCpus, VCpuInfo.DefaultCores, MemoryInfo.SizeInMiB]" \
    --output text | awk '{printf "Instance: %-18s | vCPU: %3d | Cores: %2d | Memory: %6d GiB\n", $1, $2, $3, $4/1024}'
```
[결과]
```
Instance: u7i-6tb.112xlarge  | vCPU: 448 | Cores: 224 | Memory:   6144 GiB
Instance: r7iz.32xlarge      | vCPU: 128 | Cores: 64 | Memory:   1024 GiB
Instance: r7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: i7ie.metal-48xl    | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: m7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:    768 GiB
Instance: i7ie.48xlarge      | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: c7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:    384 GiB
Instance: i7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: r7iz.metal-32xl    | vCPU: 128 | Cores: 64 | Memory:   1024 GiB
Instance: r7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: i7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:   1536 GiB
Instance: i7i.24xlarge       | vCPU:  96 | Cores: 48 | Memory:    768 GiB
Instance: c7i.48xlarge       | vCPU: 192 | Cores: 96 | Memory:    384 GiB
Instance: m7i.metal-48xl     | vCPU: 192 | Cores: 96 | Memory:    768 GiB
```
EFA 가 필요한 경우  C / M / R7i.48xlarge 로 구성된 카펜터 노드풀을 생성 한다.  

## 카펜터 노드풀 생성 ##
Llama 3-8B 모델을 World Size 4로 훈련 시켜보면, 각 프로세스당 약 60GB 정도의 버추얼 메모리를 사용한다. Pod 생성시 32 vCPU, 64 GiB Memory 기준으로 할당받기 위해 instance-size 아래와 같이 설정한다.  
```
cat <<EOF | kubectl apply -f -  
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: cpu-amx
spec:
  template:
    metadata:
      labels:
        nodeType: "cpu-amx" 
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: "karpenter.k8s.aws/instance-family"
          operator: In
          values: ["c7i", "m7i", "r7i"]
        - key: "karpenter.k8s.aws/instance-size"
          operator: In
          values: ["8xlarge", "16xlarge", "24xlarge", "48xlarge"]
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: cpu-amx
      expireAfter: 720h # 30 * 24h = 720h
      taints:
      - key: "intel.com/amx"           
        value: "present"                  
        effect: NoSchedule               
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmpty       
    consolidateAfter: 20m
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: cpu-amx
spec:
  role: "eksctl-KarpenterNodeRole-training-on-eks"
  amiSelectorTerms:
    # EKS GPU Optimized AMI: NVIDIA 드라이버와 CUDA 런타임만 포함된 가벼운 이미지 (Karpenter가 자동으로 선택 가능) 가 설치됨.
    - alias: al2023@latest

  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks" 
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "training-on-eks" 
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
  userData: |
    #!/bin/bash
    # Intel AMX 성능 최적화를 위한 라이브러리 로드 및 설정  
    sudo yum install python3-devel -y
    dnf install -y google-perftools
    pip install intel-extension-for-pytorch
EOF
```
생성된 cpu-amx 노드 클래스와 노드 풀을 확인한다. 
```
kubectl get ec2nodeclass,nodepool
```
[결과]
```
NAME                                     READY   AGE
ec2nodeclass.karpenter.k8s.aws/cpu-amx   True    6m13s
ec2nodeclass.karpenter.k8s.aws/gpu       True    25h

NAME                            NODECLASS   NODES   READY   AGE
nodepool.karpenter.sh/cpu-amx   cpu-amx     0       True    6m13s
nodepool.karpenter.sh/gpu       gpu         0       True    25h
```

## 큐브플로우 Trainer 설치 ##
```
sudo dnf install git -y

export VERSION=v2.1.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"

kubectl get clustertrainingruntimes
```

## 훈련 시작 ## 
* 훈련 리소스 - https://github.com/gnosia93/training-on-eks/tree/main/samples/cpu-amx
  
발급 받은 허깅페이스 토근을 HF_TOKEN 환경변수에 설정한 후 아래 스크립트를 실행해 준다.  
```
export NODEPOOL_NAME=cpu-amx                   # 카펜터 노드풀 지정
export NODE_NUM=4                              # 4대 
export HF_TOKEN="<your huggingface token>"     # Llama-3 모델은 HF 인증이 필요.

echo "HF_TOKEN is" ${HF_TOKEN}

git clone https://github.com/gnosia93/training-on-eks.git
cd ~/training-on-eks/samples/cpu-amx

kubectl get trainjob 
kubectl delete trainjob cpu-llama3
envsubst '$NODEPOOL_NAME $NODE_NUM $HF_TOKEN' < cpu-trainjob.yaml | kubectl apply -f - 
```

#### 훈련 Pod 리스트 조회 ####
```
kubectl get pods
```
[결과]
```
kubectl get pods
NAME                        READY   STATUS    RESTARTS   AGE
al2023-debug                1/1     Running   0          31h
cpu-llama3-node-0-0-s28t9   1/1     Running   0          15m
cpu-llama3-node-0-1-g4q8v   1/1     Running   0          15m
cpu-llama3-node-0-2-j2ff7   1/1     Running   0          15m
cpu-llama3-node-0-3-9rf5n   1/1     Running   0          15m
nginx-55bbbf955c-d559t      1/1     Running   0          33h
nginx-55bbbf955c-rpn2n      1/1     Running   0          33h
```

#### rank 0 로그 조회 ####
```
kubectl logs -f cpu-llama3-node-0-0-s28t9
```
[결과]
```
Cloning into '/workspace/code'...
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.
The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `hf`CLI if you want to set the git credential as well.
Token is valid (permission: read).
The token `training-on-eks` has been saved to /root/.cache/huggingface/stored_tokens
Your token has been saved to /root/.cache/huggingface/token
Login successful.
The current active token is: `training-on-eks`
=== Launching Distributed Training ===
Requirement already satisfied: torch>=2.4.0 in /usr/local/lib/python3.12/site-packages (from -r requirements.txt (line 2)) (2.8.0+cu129)
Requirement already satisfied: torchvision in /usr/local/lib/python3.12/site-packages (from -r requirements.txt (line 3)) (0.23.0+cu129)
Requirement already satisfied: torchaudio in /usr/local/lib/python3.12/site-packages (from -r requirements.txt (line 4)) (2.8.0+cu129)
Collecting transformers>=4.40.0 (from -r requirements.txt (line 7))
  Downloading transformers-4.57.3-py3-none-any.whl.metadata (43 kB)
Collecting datasets>=2.19.0 (from -r requirements.txt (line 8))
  Downloading datasets-4.4.2-py3-none-any.whl.metadata (19 kB)
Requirement already satisfied: accelerate>=0.30.0 in /usr/local/lib/python3.12/site-packages (from -r requirements.txt (line 9)) (1.10.1)
Collecting evaluate (from -r requirements.txt (line 10))
  Downloading evaluate-0.4.6-py3-none-any.whl.metadata (9.5 kB)
Collecting tokenizers>=0.19.0 (from -r requirements.txt (line 11))
  Downloading tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting deepspeed>=0.14.0 (from -r requirements.txt (line 14))
  Downloading deepspeed-0.18.4.tar.gz (1.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 2.1 MB/s  0:00:00
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
```

#### eks-node-viewer 실행 ####
```
eks-node-viewer
```
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/cpu-node-viewer.png)

#### k9s 로 Pod 리스트 조회 ####
```
k9s
```
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/cpu-k9s.png)


## 그라파나 대시보드 관찰 ##
```
kubectl get svc prometheus-grafana  -n monitoring
```
[결과]
```
NAME                 TYPE           CLUSTER-IP      EXTERNAL-IP                                                              PORT(S)        AGE
prometheus-grafana   LoadBalancer   172.20.193.46   a083e09003c374b72b446d26c36c67aa-199507387.us-west-2.elb.amazonaws.com   80:30426/TCP   26h
```

그라파나에 접속해서 여러가지 대시보드의 메트릭을 관찰한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/node-exporter-full.png)


## 레퍼런스 ##
* https://tutorials.pytorch.kr/recipes/amx.html
