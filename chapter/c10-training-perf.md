```
# overlays/custom-url/kustomization.yaml

resources:
- ../../base

# 1. 여기에 파라미터 역할을 할 임시 값을 정의합니다.
configMapGenerator:
- name: job-params
  literals:
  - GIT_URL=https://github.com/gnosia93/training-on-eks
  - NUM_WORKERS=3
  - INSTANCE_TYPE=p4d.24xlarge

# 2. 이 값을 필요한 곳(Master args, Worker args 등)에 뿌려줍니다.
replacements:
- source:
    kind: ConfigMap
    name: job-params
    fieldPath: data.GIT_URL
  targets:
  - select: {kind: PyTorchJob, name: pytorch-dist-job}
    fieldPaths: 
    - spec.pytorchReplicaSpecs.Master.template.spec.containers.0.args.0
    - spec.pytorchReplicaSpecs.Worker.template.spec.containers.0.args.0
    # 주의: 이 경우 args 전체가 GIT_URL로 바뀌므로, args가 단순 문자열일 때 유용합니다.

- source:
    kind: ConfigMap
    name: job-params
    fieldPath: data.NUM_WORKERS
  targets:
  - select: {kind: PyTorchJob, name: pytorch-dist-job}
    fieldPaths: ["spec.pytorchReplicaSpecs.Worker.replicas"]
```

### "파라미터 전달"을 위한 실행 명령어 ###
위와 같이 설정해두면, 이제 외부에서 kustomize edit 명령어로 값을 주입할 수 있습니다.

#### 1. 레플리카 수 파라미터 변경 ####
kustomize edit add configmap-item job-params --from-literal=NUM_WORKERS=5 --append-hash

#### 2. 인스턴스 타입 파라미터 변경 ####
kustomize edit add configmap-item job-params --from-literal=INSTANCE_TYPE=g5.48xlarge --append-hash

#### 3. 빌드 및 배포 ####
kustomize build . | kubectl apply -f -


### 만약 스크립트 내부의 "일부 문자열"만 바꾸고 싶다면? ###
작성하신 args는 긴 셸 스크립트 덩어리입니다. 이 안의 특정 단어만 바꾸는 것은 Kustomize가 매우 서툽니다. 이럴 때는 앞서 말씀드린 envsubst 조합이 정신 건강에 가장 좋습니다.

#### 추천 워크플로우: ####
kustomization.yaml의 패치 문구 안에 ${GIT_URL} 같은 변수를 적어둡니다.
```
export GIT_URL="github.com"
export WORKERS=4

kustomize build . | envsubst | kubectl apply -f -
```


