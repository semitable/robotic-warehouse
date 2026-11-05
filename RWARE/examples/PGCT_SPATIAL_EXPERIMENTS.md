# 공간 확대 PGCT / IPPO 학습 실험

이 실행기는 **실제 PPO 학습**을 수행합니다. 기본 3개 맵 × PGCT/IPPO × seed1 = 6개 실험을 순차 실행합니다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\examples\run_pgct_spatial_experiments.ps1
```

| 설정 | 값 |
|---|---|
| 공통 초기 모델 | 기존 physical curriculum stage03 `final.pt` |
| 공통 환경 | 4 agents / 2 teams / home ratio0.5 / request4(팀당2) |
| 맵 e1 | 11×10: rows1 / columns3 / height8 |
| 맵 e2 | 16×16: rows2 / columns5 / height6 |
| 맵 e3 | 20×22: rows2 / columns7 / height8 |
| 예산 | 실험당4M, 실제4,000,768 steps; 기본6개 총24,004,608 steps |
| 학습 | LR1e-4, entropy.02, rollout1024, sequence64, PPO epochs4 |
| PGCT | physical range5, probe48/shared/fixed, interval5, threshold.12, warmup100, peer loss.01 |
| IPPO | communication/graph/probe/peer transfer/consensus off |
| 평가 | 공통 seeds5,000,001–5,000,020, 20 episodes×500 steps |

모든 실험은 **동일한 초기 actor·critic 가중치**에서 독립적으로 시작합니다. 앞 실험의 결과를 다음 맵이나 IPPO에 넘기지 않습니다. optimizer/graph는 새로 시작합니다. 기존 학습기·gate·보상 코드는 수정하지 않았습니다.

이 비교는 PGCT 사전학습 모델에서 **추가 학습의 transfer 효과**를 보는 실험입니다. 처음부터 pure IPPO로 학습한 비교군은 아닙니다. 기본 맵 생성기는 shelf 밀도도 e1 29.09% → e2 42.19% → e3 47.27%로 바꾸므로 공간 확대 pilot이며 밀도 통제 실험이라고 부르면 안 됩니다.

## 선택 실행

```powershell
# 설정 생성만: 학습 없음.
.\examples\run_pgct_spatial_experiments.ps1 -DryRun

# 각 조합64스텝 시험 학습: CPU, W&B/영상 off.
.\examples\run_pgct_spatial_experiments.ps1 -SmokeTest

# 먼저 기존 맵에서 matched PGCT/IPPO 두 개만 학습.
.\examples\run_pgct_spatial_experiments.ps1 -Maps e1

# 확대 맵 두 개만 학습.
.\examples\run_pgct_spatial_experiments.ps1 -Maps e2,e3

# 세 학습 seed 반복: 18 runs, 예산은 각각 동일.
.\examples\run_pgct_spatial_experiments.ps1 -Seeds 1,2,3

# W&B off여도 로컬 지표·평가 영상은 저장.
.\examples\run_pgct_spatial_experiments.ps1 -NoTrack

# 현재 최종stage04에서 추가 적응시키려면 초기 모델을 별도로 지정.
# stage03에서 분기하는 기본 비교와 연구 질문이 다릅니다.
.\examples\run_pgct_spatial_experiments.ps1 `
  -InitCheckpoint "runs/pgct09700_physical_mixing_20260929_155338_079437/stage04_4ag_physical_home0.500/final.pt"

# 후속 workload 비교: Q4 맵 실험과 별도로 실행.
.\examples\run_pgct_spatial_experiments.ps1 -Maps e3 -RequestCount 8
```

`-Modes pgct` 또는 `-Modes ippo`, `-TotalTimesteps`, `-ShelfHomeRatio`, `-EvalSeed`, `-OutputDir`, `-NoVideo`도 지원합니다. 새 출력 폴더만 허용하며 기존 폴더를 덮어쓰지 않습니다. 중단된 실험의 자동 optimizer resume 기능은 없습니다. 남은 조합을 `-Maps/-Modes/-Seeds`로 골라 새 폴더에서 실행할 수 있습니다.

## 결과와 접촉 기록

기본 출력은 `runs/sp_<timestamp>/`이고 실험 이름은 `e1_pgct_s1`처럼 짧습니다. Windows에서는 W&B Git 경로 추론 충돌을 피하도록 `WANDB_DISABLE_GIT=true`를 미지정 시 설정합니다. Python 실행기로 직접 실행하면 W&B는 `--track`을 지정할 때만 켜지며 PowerShell 기본 실행은 켭니다.

- `experiment.json`: 초기 모델, 전체 설정·예산, 각 실험 상태와 최종 결과.
- `eX_<mode>_sN/final.pt`, `checkpoint_*.pt`: 기존 학습기의 모델 저장.
- `metrics.jsonl`: W&B 여부와 무관한 로컬 train/update/probe/eval scalar 기록.
- `spatial.jsonl`: update마다 전체 rollout의 agent별 점유 histogram·phase·좌우 체류율·물리 후보 step 수, update 끝의 geometric 후보.
- `communication.jsonl`: 기존 학습기가 기록한 실제 optimizer 시점 전이.
- `summary.json`: same/cross geometric 후보와 승인 횟수·acceptance, 최종 평가.
- `final_eval_metrics.json`, `videos/`: 기존 평가 지표·영상.

IPPO의 `communication.jsonl.available`은 알고리즘 off로0이지만, `spatial.jsonl`의 geometric 후보는 좌표로 별도 계산합니다. 따라서 통신 off와 서로 만나지 않는 것을 구별할 수 있습니다. step별 후보 수와 update 시점 승인을 혼합하지 않습니다. 동일한 evaluation seed 목록을 매 체크포인트에서 사용하지만 policy 경로가 달라지면 이후 요청 RNG 소비도 달라질 수 있습니다.
