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
Llama 3-8B 모델을 World Size 4로 훈련 시켜보면, 각 프로세스당 약 60GB 정도의 메모리를 점유한다.
노드풀 생성시 c7i.8xlarge(32 vCPU, 64 GB Memory) 기준으로 하나의 Pod 로 할당하게 설정한다.   
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
    - name: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023)*"
  amiFamily: AL2023 
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


## 훈련 하기 ##
* 파드당 Memory 는 64GB, CPU 는 32 Core 로 설정.

<< 작성 필요 >>


## 레퍼런스 ##
* https://tutorials.pytorch.kr/recipes/amx.html
