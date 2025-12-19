#
# 다음은 overlays 의 kustomization.yaml 에서 실행시 patch 해야할 필드 값들이 대한 설명이다.  
# 파라미터 
#   ${NODEPOOL} = gpu 
#   ${INSTANCE_TYPE} = g6.2xlarge
#   ${TOPOLOGY_KEY} = kubernetes.io/zone, kubernetes.io/hostname, kubernetes.io/region
#   ${ZONE} = ap-northeast-2a
#   ${REPLICAS} = 3
# 설명
#   ${NODEPOOL} : 카펜터 노드풀 명칭으로 기본값은 gpu로 설정, 다른 명칭으로 생성한 경우 해당 명칭사용
#   ${INSTANCE_TYPE} : 1 GPU를 가진 g6.2xlarge로 설정, 다른 타입이 필요한 경우 변경
#   ${TOPOLOGY_KEY} : kubernetes.io/zone 으로 기본 설정하고, p4d.24xlarge 와 같이 8대의 GPU 를 사용하는 경우 kubernetes.io/hostname 으로 변경
#   ${REPLICAS} : pytorchJob 의 워커노드 수 설정 
#
# 파드는 NodeSelector + PodAffinity + Toleration 이렇게 3가지의 AND 조건으로 실행될 노드를 할당 받는다. 이 세가지 조건을 동시에 충족시키지 못하면 스케줄링은 Pending 된다.  
#
# runPolicy.cleanPodPolicy: 
#   Running / ALL / None 이렇게 3가지 파라미터가 있다. 자세한 설명은 메뉴얼 참조 
#      
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-dist-job
  namespace: pytorch 
spec:
  runPolicy:
    cleanPodPolicy: Running
  
  pytorchReplicaSpecs:
    Master:                       
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          # --- [수정] NodeSelector 추가 (gpu 강제 지정) ---
          nodeSelector:
            karpenter.sh/nodepool: ${NODEPOOL}
            node.kubernetes.io/instance-type: ${INSTANCE_TYPE} 
          # -----------------------------------------------------
          affinity:
            podAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                - labelSelector:
                     matchExpressions:
                     - key: training.kubeflow.org/job-name
                       operator: In
                       values:
                       - pytorch-dist-job
                  topologyKey: kubernetes.io/hostname          
          tolerations:            # GPU Toleration 설정 
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: pytorch
            image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
            command: ["/bin/bash", "-c"] 
            args: 
              - |
                git clone github.com /workspace/code    
                python /workspace/code/training.py
            resources:
              limits:
                nvidia.com/gpu: "1"
              requests:
                nvidia.com/gpu: "1"    
    Worker:
      replicas: ${REPLICAS}
      restartPolicy: OnFailure
      template:
        spec:
          # --- [수정] NodeSelector 추가 (gpu 강제 지정) ---
          nodeSelector:
            karpenter.sh/nodepool: ${NODEPOOL}
            node.kubernetes.io/instance-type: ${INSTANCE_TYPE} 
          # -----------------------------------------------------
          affinity:
            podAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                - labelSelector:
                     matchExpressions:
                     - key: training.kubeflow.org/job-name
                       operator: In
                       values:
                       - pytorch-dist-job
                  topologyKey: kubernetes.io/hostname      
          tolerations:            # GPU Toleration 설정 
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
          containers:
          - name: pytorch
            image: public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.0
            command: ["/bin/bash", "-c"] 
            args: 
              - |
                git clone github.com /workspace/code    
                python /workspace/code/training.py
            resources:
              limits:
                nvidia.com/gpu: "1"
              requests:
                nvidia.com/gpu: "1"
