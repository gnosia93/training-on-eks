$ kubectl describe node ip-10-0-5-152.ap-northeast-2.compute.internal
```
Name:               ip-10-0-5-152.ap-northeast-2.compute.internal
Roles:              <none>
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/instance-type=p4d.24xlarge
                    beta.kubernetes.io/os=linux
                    failure-domain.beta.kubernetes.io/region=ap-northeast-2
                    failure-domain.beta.kubernetes.io/zone=ap-northeast-2b
                    feature.node.kubernetes.io/cpu-cpuid.ADX=true
                    feature.node.kubernetes.io/cpu-cpuid.AESNI=true
                    feature.node.kubernetes.io/cpu-cpuid.AMXFP8=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX2=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX512BW=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX512CD=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX512DQ=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX512F=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX512VL=true
                    feature.node.kubernetes.io/cpu-cpuid.CMPXCHG8=true
                    feature.node.kubernetes.io/cpu-cpuid.FMA3=true
                    feature.node.kubernetes.io/cpu-cpuid.FXSR=true
                    feature.node.kubernetes.io/cpu-cpuid.FXSROPT=true
                    feature.node.kubernetes.io/cpu-cpuid.HYPERVISOR=true
                    feature.node.kubernetes.io/cpu-cpuid.LAHF=true
                    feature.node.kubernetes.io/cpu-cpuid.MOVBE=true
                    feature.node.kubernetes.io/cpu-cpuid.MPX=true
                    feature.node.kubernetes.io/cpu-cpuid.OSXSAVE=true
                    feature.node.kubernetes.io/cpu-cpuid.SYSCALL=true
                    feature.node.kubernetes.io/cpu-cpuid.SYSEE=true
                    feature.node.kubernetes.io/cpu-cpuid.X87=true
                    feature.node.kubernetes.io/cpu-cpuid.XGETBV1=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVE=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVEC=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVEOPT=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVES=true
                    feature.node.kubernetes.io/cpu-cstate.enabled=true
                    feature.node.kubernetes.io/cpu-hardware_multithreading=true
                    feature.node.kubernetes.io/cpu-model.family=6
                    feature.node.kubernetes.io/cpu-model.id=85
                    feature.node.kubernetes.io/cpu-model.vendor_id=Intel
                    feature.node.kubernetes.io/kernel-config.NO_HZ=true
                    feature.node.kubernetes.io/kernel-config.NO_HZ_FULL=true
                    feature.node.kubernetes.io/kernel-version.full=6.12.58-82.121.amzn2023.x86_64
                    feature.node.kubernetes.io/kernel-version.major=6
                    feature.node.kubernetes.io/kernel-version.minor=12
                    feature.node.kubernetes.io/kernel-version.revision=58
                    feature.node.kubernetes.io/memory-numa=true
                    feature.node.kubernetes.io/pci-10de.present=true
                    feature.node.kubernetes.io/pci-1d0f.present=true
                    feature.node.kubernetes.io/rdma.available=true
                    feature.node.kubernetes.io/storage-nonrotationaldisk=true
                    feature.node.kubernetes.io/system-os_release.ID=amzn
                    feature.node.kubernetes.io/system-os_release.VERSION_ID=2023
                    feature.node.kubernetes.io/system-os_release.VERSION_ID.major=2023
                    k8s.io/cloud-provider-aws=66888f96c01cf55f768e3ce2e887228e
                    karpenter.k8s.aws/ec2nodeclass=gpu
                    karpenter.k8s.aws/instance-capability-flex=false
                    karpenter.k8s.aws/instance-category=p
                    karpenter.k8s.aws/instance-cpu=96
                    karpenter.k8s.aws/instance-cpu-manufacturer=intel
                    karpenter.k8s.aws/instance-cpu-sustained-clock-speed-mhz=3000
                    karpenter.k8s.aws/instance-ebs-bandwidth=19000
                    karpenter.k8s.aws/instance-encryption-in-transit-supported=true
                    karpenter.k8s.aws/instance-family=p4d
                    karpenter.k8s.aws/instance-generation=4
                    karpenter.k8s.aws/instance-gpu-count=8
                    karpenter.k8s.aws/instance-gpu-manufacturer=nvidia
                    karpenter.k8s.aws/instance-gpu-memory=40960
                    karpenter.k8s.aws/instance-gpu-name=a100
                    karpenter.k8s.aws/instance-hypervisor=nitro
                    karpenter.k8s.aws/instance-local-nvme=8000
                    karpenter.k8s.aws/instance-memory=1179648
                    karpenter.k8s.aws/instance-network-bandwidth=400000
                    karpenter.k8s.aws/instance-size=24xlarge
                    karpenter.sh/capacity-type=spot
                    karpenter.sh/do-not-sync-taints=true
                    karpenter.sh/initialized=true
                    karpenter.sh/nodepool=gpu
                    karpenter.sh/registered=true
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=ip-10-0-5-152.ap-northeast-2.compute.internal
                    kubernetes.io/os=linux
                    node.kubernetes.io/instance-type=p4d.24xlarge
                    topology.ebs.csi.aws.com/zone=ap-northeast-2b
                    topology.k8s.aws/network-node-layer-1=nn-c79422f4e61deb9ca
                    topology.k8s.aws/network-node-layer-2=nn-6ff5053c3a4d86db7
                    topology.k8s.aws/network-node-layer-3=nn-dcbcca51aebf4c95b
                    topology.k8s.aws/zone-id=apne2-az2
                    topology.kubernetes.io/region=ap-northeast-2
                    topology.kubernetes.io/zone=ap-northeast-2b
Annotations:        alpha.kubernetes.io/provided-node-ip: 10.0.5.152
                    csi.volume.kubernetes.io/nodeid: {"ebs.csi.aws.com":"i-0b9b9fb02def3eed3"}
                    karpenter.k8s.aws/ec2nodeclass-hash: 9766261846345886753
                    karpenter.k8s.aws/ec2nodeclass-hash-version: v4
                    karpenter.k8s.aws/instance-profile-name: training-on-eks_18371070903040539160
                    karpenter.sh/nodeclaim-min-values-relaxed: false
                    karpenter.sh/nodepool-hash: 3665531442370852411
                    karpenter.sh/nodepool-hash-version: v3
                    nfd.node.kubernetes.io/feature-labels:
                      cpu-cpuid.ADX,cpu-cpuid.AESNI,cpu-cpuid.AMXFP8,cpu-cpuid.AVX,cpu-cpuid.AVX2,cpu-cpuid.AVX512BW,cpu-cpuid.AVX512CD,cpu-cpuid.AVX512DQ,cpu-c...
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Thu, 18 Dec 2025 01:28:47 +0000
Taints:             nvidia.com/gpu=present:NoSchedule
Unschedulable:      false
Lease:
  HolderIdentity:  ip-10-0-5-152.ap-northeast-2.compute.internal
  AcquireTime:     <unset>
  RenewTime:       Thu, 18 Dec 2025 01:29:59 +0000
Conditions:
  Type             Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----             ------  -----------------                 ------------------                ------                       -------
  MemoryPressure   False   Thu, 18 Dec 2025 01:29:49 +0000   Thu, 18 Dec 2025 01:28:46 +0000   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure     False   Thu, 18 Dec 2025 01:29:49 +0000   Thu, 18 Dec 2025 01:28:46 +0000   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure      False   Thu, 18 Dec 2025 01:29:49 +0000   Thu, 18 Dec 2025 01:28:46 +0000   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready            True    Thu, 18 Dec 2025 01:29:49 +0000   Thu, 18 Dec 2025 01:29:17 +0000   KubeletReady                 kubelet is posting ready status
Addresses:
  InternalIP:   10.0.5.152
  InternalDNS:  ip-10-0-5-152.ap-northeast-2.compute.internal
  Hostname:     ip-10-0-5-152.ap-northeast-2.compute.internal
Capacity:
  cpu:                96
  ephemeral-storage:  314494956Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             1176300008Ki
  nvidia.com/gpu:     8
  pods:               737
Allocatable:
  cpu:                95690m
  ephemeral-storage:  288764809146
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             1167634920Ki
  nvidia.com/gpu:     8
  pods:               737
System Info:
  Machine ID:                 ec2f7360c2f7b4c41b8304f57e0dee91
  System UUID:                ec2f7360-c2f7-b4c4-1b83-04f57e0dee91
  Boot ID:                    ab8e96c3-b4b3-445a-b258-ab9833e179eb
  Kernel Version:             6.12.58-82.121.amzn2023.x86_64
  OS Image:                   Amazon Linux 2023.9.20251208
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  containerd://2.1.5
  Kubelet Version:            v1.34.2-eks-ecaa3a6
  Kube-Proxy Version:         
ProviderID:                   aws:///ap-northeast-2b/i-0b9b9fb02def3eed3
Non-terminated Pods:          (12 in total)
  Namespace                   Name                                                     CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------                   ----                                                     ------------  ----------  ---------------  -------------  ---
  dcgm                        dcgm-exporter-vqgwc                                      100m (0%)     500m (0%)   256Mi (0%)       1Gi (0%)       49s
  kube-system                 aws-node-tp76n                                           50m (0%)      0 (0%)      0 (0%)           0 (0%)         78s
  kube-system                 ebs-csi-node-h8t7d                                       30m (0%)      0 (0%)      120Mi (0%)       768Mi (0%)     78s
  kube-system                 kube-proxy-h7c29                                         100m (0%)     0 (0%)      0 (0%)           0 (0%)         78s
  monitoring                  prometheus-prometheus-node-exporter-2cgzb                0 (0%)        0 (0%)      0 (0%)           0 (0%)         78s
  nvidia                      nvdp-node-feature-discovery-worker-btng2                 5m (0%)       0 (0%)      64Mi (0%)        512Mi (0%)     49s
  nvidia                      nvdp-nvidia-device-plugin-9kz8r                          0 (0%)        0 (0%)      0 (0%)           0 (0%)         42s
  nvidia                      nvdp-nvidia-device-plugin-gpu-feature-discovery-8wr8z    0 (0%)        0 (0%)      0 (0%)           0 (0%)         42s
  pytorch                     pytorch-dist-job-master-0                                0 (0%)        0 (0%)      0 (0%)           0 (0%)         117s
  pytorch                     pytorch-dist-job-worker-0                                50m (0%)      100m (0%)   10Mi (0%)        20Mi (0%)      117s
  pytorch                     pytorch-dist-job-worker-1                                50m (0%)      100m (0%)   10Mi (0%)        20Mi (0%)      117s
  pytorch                     pytorch-dist-job-worker-2                                50m (0%)      100m (0%)   10Mi (0%)        20Mi (0%)      117s
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests    Limits
  --------           --------    ------
  cpu                435m (0%)   800m (0%)
  memory             470Mi (0%)  2364Mi (0%)
  ephemeral-storage  0 (0%)      0 (0%)
  hugepages-1Gi      0 (0%)      0 (0%)
  hugepages-2Mi      0 (0%)      0 (0%)
  nvidia.com/gpu     4           4
Events:
  Type     Reason                   Age                From                   Message
  ----     ------                   ----               ----                   -------
  Normal   Starting                 58s                kube-proxy             
  Normal   Starting                 81s                kubelet                Starting kubelet.
  Warning  InvalidDiskCapacity      81s                kubelet                invalid capacity 0 on image filesystem
  Normal   NodeAllocatableEnforced  81s                kubelet                Updated Node Allocatable limit across pods
  Normal   NodeHasSufficientMemory  80s (x3 over 81s)  kubelet                Node ip-10-0-5-152.ap-northeast-2.compute.internal status is now: NodeHasSufficientMemory
  Normal   NodeHasNoDiskPressure    80s (x3 over 81s)  kubelet                Node ip-10-0-5-152.ap-northeast-2.compute.internal status is now: NodeHasNoDiskPressure
  Normal   NodeHasSufficientPID     80s (x3 over 81s)  kubelet                Node ip-10-0-5-152.ap-northeast-2.compute.internal status is now: NodeHasSufficientPID
  Normal   RegisteredNode           78s                node-controller        Node ip-10-0-5-152.ap-northeast-2.compute.internal event: Registered Node ip-10-0-5-152.ap-northeast-2.compute.internal in Controller
  Normal   Synced                   78s                cloud-node-controller  Node synced successfully
```
  Normal   DisruptionBlocked        74s                karpenter              Node isn't initialized
  Normal   Ready                    49s                karpenter              Status condition transitioned, Type: Ready, Status: False -> True, Reason: KubeletReady, Message: kubelet is posting ready status
  Normal   NodeReady                49s                kubelet                Node 
