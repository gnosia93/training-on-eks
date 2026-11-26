## S3 ##
```
iter_0001000/
  __0_0.distcp      ← rank 0이 쓴 데이터
  __1_0.distcp      ← rank 1이 쓴 데이터
  ...
  __63_0.distcp     ← rank 63이 쓴 데이터
  .metadata         ← 어떤 텐서의 어느 조각이 어느 파일 어디에 있는지 적은 목록 (rank 0이 씀)
  common.pt, metadata.json   ← Megatron 공통 정보 (rank 0이 씀)
```
* rank마다 각자 파일을 쓴다. rank 0으로 모아서 한 파일로 쓰지 않습니다.
* 병렬 구성이 바뀌어도 다시 나눠 읽을 수 있다는 말은 .metadata 덕분입니다. .metadata에는 텐서마다 전체 크기와 각 조각의 위치가 적혀 있습니다. 그래서 TP=8, PP=2, DP=4로 저장한 체크포인트를 노드 수나 병렬화를 바꾼 뒤에도 읽을 수 있습니다. 각 rank는 새 구성에서 자기가 맡을 부분을 .metadata에서 찾아, 해당 파일의 해당 위치만 읽습니다.





## 분산 체크포인트 학습 재개 스크립트 (S3 + Nvme) ##
* EFS 나 Lustre 와 같은 공유 스토리지에 체크 포인트를 저장하지 않음.
* S3 와 각 노드의 NVME 에만 체크포인트를 저장하는 구조에서 slurm 작업을 재 실행하는 방법이다.
* step 번호를 찾는다. srun 이 붙어 있으면 모든 노드, srun 이 없으면 첫번째 노드에서만 스크립트가 실행된다. 
```
#!/bin/bash
#SBATCH --job-name=train-70b
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --requeue                  # 노드 장애 시 Slurm이 자동으로 다시 큐에 넣음
#SBATCH --dependency=singleton     # 같은 이름의 잡이 동시에 두 개 돌지 않게 함
#SBATCH --open-mode=append         # 다시 시작해도 이전 로그를 덮어쓰지 않음
#SBATCH --output=train-%j.out      # 다시 시작해도 잡 ID가 같아서 같은 파일에 이어 씀

export RUN=${RUN:-run1}            # 실험 이름. 실험마다 바꿔서 다른 실험의 체크포인트와 섞이지 않게 함
export S3=s3://my-bucket/$RUN      # 체크포인트를 올리는 S3 경로
export LOCAL=/nvme/ckpt/$RUN       # 노드마다 있는 로컬 NVMe 경로
MAX_RETRY=3                        # 다시 시작할 최대 횟수 (노드 장애로 다시 시작한 횟수 포함)

echo "시작: 실험 $RUN, 잡 $SLURM_JOB_ID, 재시작 횟수 ${SLURM_RESTART_COUNT:-0}"

# ===== 1. 준비 단계 =====
# 저장 쪽 규칙: 노드마다 $LOCAL/step_<번호>/에 자기 파일을 쓰고, 다 쓴 뒤 DONE 파일에 자기 노드 번호를 적음.
#              S3에는 $S3/step_<번호>/node_<노드번호>/에 같은 파일을 올리고 DONE을 맨 마지막에 올림.

# 1-1) 노드마다 로컬에 다 쓴 체크포인트를 "노드번호 step" 형식으로 출력
#      DONE에 적힌 번호가 지금 노드 번호와 다르면 다른 노드의 파일이므로 세지 않음
LOCAL_LIST=$(srun --ntasks-per-node=1 bash -c '
  for f in "$LOCAL"/step_*/DONE; do
    [ "$(cat "$f" 2>/dev/null)" = "$SLURM_NODEID" ] &&
      echo "$SLURM_NODEID $(basename "$(dirname "$f")")"
  done
  true') || { echo "로컬 목록 조회 실패" >&2; exit 1; }

# 1-2) S3에 업로드가 끝난(노드별 DONE이 있는) 체크포인트를 같은 형식으로 정리
S3_KEYS=$(aws s3 ls --recursive "$S3/"); rc=$?
if [ "$rc" -gt 1 ]; then           # 1은 "목록이 비어 있음", 그보다 크면 조회 실패
  echo "S3 목록 조회 실패 (rc=$rc)" >&2; exit 1
fi
S3_LIST=$(echo "$S3_KEYS" | awk '{print $4}' \
  | sed -nE 's#.*/(step_[0-9]+)/node_([0-9]+)/DONE$#\2 \1#p')

# 1-3) 8대 모두가 로컬이나 S3에서 구할 수 있는 step 중 가장 최신 것을 고름
STEP=$(printf '%s\n%s\n' "$LOCAL_LIST" "$S3_LIST" \
  | awk -v n="$SLURM_JOB_NUM_NODES" 'NF == 2 && !seen[$0]++ { cnt[$2]++ }
        END { for (s in cnt) if (cnt[s] == n) print s }' \
  | sort -t_ -k2,2n | tail -n 1)

# 체크포인트가 있는데 온전한 것이 하나도 없으면, 지우거나 처음부터 시작하지 말고 멈춤
if [ -z "$STEP" ] && { [ -n "$LOCAL_LIST" ] || [ -n "$S3_KEYS" ]; }; then
  echo "체크포인트는 있지만 8대 모두 구할 수 있는 것이 없음. 사람이 확인해야 함" >&2; exit 1
fi

# 1-4) 노드마다 그 step이 로컬에 없으면 자기 몫만 S3에서 받음
export RESUME_DIR=""
if [ -n "$STEP" ]; then
  export STEP
  srun --ntasks-per-node=1 bash -c '
    DIR=${LOCAL:?}/${STEP:?}
    [ "$(cat "$DIR/DONE" 2>/dev/null)" = "$SLURM_NODEID" ] && exit 0   # 로컬에 있으면 그대로 사용
    TMP=$LOCAL/.tmp-$STEP                                               # 임시 폴더에 다 받은 뒤 옮김
    rm -rf "$TMP" && mkdir -p "$TMP" &&
      s5cmd cp "$S3/$STEP/node_$SLURM_NODEID/*" "$TMP/" &&
      [ "$(cat "$TMP/DONE" 2>/dev/null)" = "$SLURM_NODEID" ] &&
      rm -rf "$DIR" && mv "$TMP" "$DIR"
  ' || { echo "체크포인트 준비 실패: $STEP" >&2; exit 1; }
  RESUME_DIR=$LOCAL/$STEP

  # 1-4b) rank 0만 쓰는 공통 파일을 1번 노드(배치 스크립트가 도는 노드)에서 8대 모두에 복사
  for f in $COMMON_FILES; do
    cp "$LOCAL/$STEP/$f" "/tmp/$f.$SLURM_JOB_ID" &&            # 원본을 직접 보내지 않고 사본에서 보냄
      sbcast -f "/tmp/$f.$SLURM_JOB_ID" "$LOCAL/$STEP/$f" ||
      { echo "공통 파일 복사 실패: $f" >&2; exit 1; }
  done
fi


# 1-5) 재개 지점보다 새 체크포인트는 로컬과 S3에서 모두 지움
#      직전 체크포인트로 돌아가면 같은 step 이름을 다시 쓰게 되므로, 버린 쪽의 옛 파일과 섞이지 않게 함
if [ -n "$STEP" ]; then KEEP=$((10#${STEP#step_})); else KEEP=-1; fi
export KEEP
srun --ntasks-per-node=1 bash -c '
  for d in "$LOCAL"/step_*; do
    [ -d "$d" ] && [ "$((10#${d##*/step_}))" -gt "$KEEP" ] && rm -rf "$d"
  done
  true' || { echo "로컬 정리 실패" >&2; exit 1; }
for s in $(echo "$S3_KEYS" | awk '{print $4}' | sed -nE 's#.*/(step_[0-9]+)/.*#\1#p' | sort -u); do
  if [ "$((10#${s#step_}))" -gt "$KEEP" ]; then
    aws s3 rm --recursive --quiet "$S3/$s/" || { echo "S3 정리 실패: $s" >&2; exit 1; }
  fi
done
echo "재개 지점: ${STEP:-없음, 처음부터 학습}"

# ===== 2. 학습 =====
# node_rank를 SLURM_NODEID로 고정해야 각 노드가 로컬에 가진 파일과 rank가 맞음
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun --ntasks-per-node=1 bash -c '
  torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=8 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train.py ${RESUME_DIR:+--resume-from "$RESUME_DIR"}'
RC=$?

# ===== 3. 재시도: 학습 프로세스만 죽은 경우 스스로 다시 큐에 넣음 =====
# (노드 장애는 --requeue 덕분에 Slurm이 알아서 다시 넣으므로 여기까지 오지 않음)
if [ "$RC" -ne 0 ] && [ "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_RETRY" ]; then
  echo "학습 실패 (rc=$RC), 다시 큐에 넣음"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit "$RC"
```



### S3에서 전체를 받아 복구했다면 ###

복구는 S3에서 체크포인트 전체를 받아서 재개했습니다. 노드마다 약 1.2 TB라 몇 분이 걸리지만, 장애가 월 1회 안팎이라 복구 시간보다 구조가 단순한 쪽을 택했습니다. 이 방식은 노드마다 파일이 전부 있으므로 프레임워크와 상관없이 로드됩니다. "복구 몇 분 대신 단순함"이라는 트레이드오프로 설명할 수 있다는 점도 좋습니다.
