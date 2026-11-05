# Stage03 이후 critic transfer 검증 학습

`run_pgct_transfer_experiments.ps1`은 실제 PPO 학습을 수행합니다. 기본은 기존 11×10 맵에서 PGCT-on/IPPO-off × training seeds 1·2·3입니다. 모든 조건은 같은 Stage03 actor·critic에서 시작하고 optimizer와 graph를 초기화합니다. 원래 학습기, 환경, JS gate와 기존 공간 실행기는 수정하지 않았습니다. Ablation 조건에만 이 실행기 프로세스 안에서 실제 auxiliary loss의 donor allocation을 교체합니다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\examples\run_pgct_transfer_experiments.ps1
```

기본 조건: Stage03 `stage03_4ag_physical_home0.750/final.pt`, 4 agents / 2 teams, home ratio .5, Q4(팀당2), LR1e-4, entropy .02, rollout1024, 공통 평가 seeds5,000,001–5,000,020 / 20×500. 각 branch는 4M 요청 예산을 rollout 단위로 올림하여 **4,000,768 환경 step** 학습합니다. 기본6개 총24,004,608 step입니다. W&B는 기본 on, 영상은 `-SaveVideo` 지정 시 저장합니다. 출력은 짧은 `runs/tr_<timestamp>/`입니다.

## 세 단계 실행

먼저 on/off를 실행하고 성능 차이와 전이 노출량을 확인합니다.

```powershell
# 1단계: on/off. 기본과 동일.
.\examples\run_pgct_transfer_experiments.ps1 -Modes pgct,ippo -Seeds 1,2,3

# 2단계: 완료한 1단계 결과를 donor 전이 일정의 기준으로 사용.
# 이 경로는 실제 완료된 1단계 출력 폴더로 바꿉니다.
.\examples\run_pgct_transfer_experiments.ps1 -Modes random -Seeds 1,2,3 `
  -ReferenceExperiment "runs/tr_첫실험출력폴더"

# 3단계: 3–4 양방향 전이 제거. 나머지 allocation은 유지하며 재정규화하지 않음.
.\examples\run_pgct_transfer_experiments.ps1 -Modes drop34 -Seeds 1,2,3

# 필요할 때만 별도 비교: 제거한 양을 다른 물리 이웃에게 보충.
.\examples\run_pgct_transfer_experiments.ps1 -Modes drop34_refill -Seeds 1,2,3

# 전부 순차 실행할 수도 있음: 4조건×3seeds=12runs, 약48M 추가 step.
.\examples\run_pgct_transfer_experiments.ps1 -Modes pgct,ippo,random,drop34 -Seeds 1,2,3

# 설정만 생성 / 각 조건64스텝 실제 실행 검증.
.\examples\run_pgct_transfer_experiments.ps1 -DryRun
.\examples\run_pgct_transfer_experiments.ps1 -Seeds 1 -Modes pgct,ippo,random,drop34,drop34_refill -SmokeTest
```

Random은 같은 seed PGCT-on의 receiver별 전이 시점·donor 수·allocation weight를 replay합니다. Donor는 **해당 random branch의 현재 물리 이웃 중 균등하게** 선택합니다. 이웃이 부족하면 가능한 donor에 receiver 총 weight를 분배하고 donor 수 부족을 기록합니다. 이웃이 전혀 없으면 전이를 생략하고 weight 부족을 기록합니다. 따라서 완벽한 전달량 일치를 가정하지 않고 결과에서 실제 부족량을 확인해야 합니다. 보조 손실 계수는 동일한 .01입니다. Donor 예측 차이에 따른 실제 peer loss와 gradient 크기까지 같다는 뜻은 아닙니다. PGCT reference의 환경·초기 모델·학습 seed·예산·주요 학습 조건이 다르면 실행을 거부합니다.

`drop34`는 현재 branch의 gate가 여는 3→4와 4→3만 제거합니다. 다른 연결을 새로 열거나 남은 weight를 재정규화하지 않으므로 전체 전이량이 줄 수 있습니다. `drop34_refill`은 제거량을 다른 물리 이웃 한 곳에 더하며 보충할 이웃이 없으면 부족을 기록합니다. 두 조건은 각각 해당 연결 제거의 전체 효과와 전달량 보충 시의 효과를 묻습니다. 모두 agent 표시번호는 1-based, 행렬은 receiver×donor입니다. Ablation 전후 궤적이 달라지므로 다른 연결의 미래 gate 발생도 달라질 수 있습니다.

## Probe 색상 차트와 JS 진단

에이전트마다 별도 probe를 비교하지 않고 **동일한 공통 입력에 대한 각 에이전트의 응답**을 비교합니다. 진단 bank는 E1 환경과 seed1로 고정하며 모든 method·training seed에서 동일합니다. 이 bank는 진단 전용이고 실제 학습 gate의 probe bank와 criterion을 교체하지 않습니다.

- `probes.jsonl`: 최초·매5update·학습 종료의 agent별 전체 action probability vector, probe별 JS, 평균 JS, 중복 제거 평균 JS, entropy, 입력 중복 ID. 종료 스냅샷은 최종 update 이후입니다.
- `charts/response.png`: 4개 agent의 probe×action 확률과 6개 pair의 probe별 JS, 4×4 평균 JS. 확률은 노란색일수록 높고, JS는 밝을수록 유사합니다.
- `charts/js_time.png`, `unique_js_time.png`: x=추가 환경 step, y=6개 pair. Natural-log JS 범위를 [0,ln2]로 고정합니다.
- `charts/ema_time.png`: 실제 학습 gate의 neighbor-only EMA; 진단 평균 JS와 다릅니다.
- `charts/transfer_time.png`: 실제 auxiliary loss에 사용한 pair-update 전이.
- W&B `diagnostic/probe_color_chart`: 기본 매50update 확률/JS 차트. `diagnostic/js_*`, `unique_js_*` scalar에도 기록합니다.

기존48개 probe의 중복은 동일하게 유지한 평균과 고유 입력 균등 평균을 별도로 비교합니다. Dedup 결과를 기존 .12 gate에 적용하여 새로운 승인으로 해석하면 안 됩니다.

Critic 평가의 첫 stochastic episode에서는 4개 agent의 경험 observation·action-mask 이력을 모두 사용합니다. 각 이력에 모든 actor를 넣고 각 actor가 자기 hidden state를 전개하게 하여 공통 이력에서의 masked JS와 같은 observation에 대한 zero-hidden masked JS를 함께 기록합니다. 다른 actor에게는 off-policy 이력이며 rollout 자체의 행동이 일치한다는 증거는 아닙니다.

## 성능·critic·실제 전달 기록

- `metrics.jsonl`: 기존 train/update/probe/eval + diagnostic scalar. 최초 step0와 종료 예산에서도 task 평가.
- `final_eval_metrics.json`: 기존 deterministic task 성능, 추가 stochastic critic 오차, 팀별·에이전트별 이벤트/episode.
- `critic_eval.jsonl`: 평가 때마다 기본5×500 stochastic policy rollout. Local NumPy RNG로 행동을 샘플링하여 training RNG를 소비하지 않습니다. 개인별 보상 전체를 discount한 **유한 horizon Monte Carlo return**의 MSE/MAE와 one-step TD 절대오차. TD는 time limit에서는 bootstrap하고 실제 termination에서는 bootstrap하지 않습니다.
- MC는 horizon 이후 보상을 포함하지 않아 continuing critic과 tail bias가 있습니다. 동일 horizon에서도 정책별 방문 분포가 다르므로 MC MSE 감소 하나로 전이 유익성을 결론내리지 않습니다. Deterministic task 평가와 stochastic critic 평가를 구별합니다. `-CriticEpisodes 0`은 critic 진단을 끕니다.
- `transfer.jsonl`: 실제 receiver×donor allocation, 원래 local gate allocation, 물리 후보, 전달량·donor 수 부족, 전이 전 pair별 critic MSE, peer loss 계수.
- **Ablation의 실제 전이는 `transfer.jsonl`을 사용합니다.** 기존 `communication.jsonl`에는 원래 PGCT gate가 남으므로 random/drop 조건의 실제 전달과 다를 수 있습니다. 학습 영상의 선도 원래 gate를 표시할 수 있습니다.
- `transfer_summary.json`: directed 전달 update 수, weight 합, 전달량 부족, 평가 곡선 trapezoidal AUC와 AUC/step, 목표 delivery에 최초 도달한 평가 step. 기본 목표45 deliveries/episode는 결과를 보기 전에 정하는 설정값이며 `-PerformanceTarget`으로 바꿉니다. 평가 간 최초 도달 시점은 알 수 없습니다.
- 실험 전체 `charts/learning.png`, `agents.png`, `critic.png`: 조건·seed별 곡선. 현재는 개별 seed를 표시하며 통계적 유의성을 주장하지 않습니다.

Wall-clock은 평가·진단까지 포함한 elapsed seconds와 train SPS를 기록합니다. 공간 실행기의 예전 결과와 새 실행기의 wall-clock을 직접 공정 비교하면 안 됩니다. 이전 train의 `value_uncertainty`/GAE residual은 새 one-step TD/MC 오차와 다른 지표입니다.

## 현재 공간 실험 재사용

현재 `runs/sp_20261008_120638_983201`의 E1 PGCT/IPPO는 Stage03·환경·예산이 맞으므로 성능 비교에 사용 가능합니다. E2/E3는 공간 확대 후속 실험으로 보존합니다. **실행 중인 프로세스를 이 코드가 중단하거나 그 설정을 바꾸지 않습니다.** E1의 두 조건 완료 후 다음 명령은 seed1 학습을 재실행하지 않고 완료 결과를 읽으며 seeds2·3만 새로 학습합니다.

```powershell
.\examples\run_pgct_transfer_experiments.ps1 -Seeds 1,2,3 `
  -ReuseExperiment "runs/sp_20261008_120638_983201"
```

재사용하려는 조건이 아직 완료되지 않았으면 실행을 거부합니다. 새로 생성한 폴더의 기록에는 `reused_run_dir`가 표시됩니다. 이전 결과에는 새 critic 진단이 없으므로 **성능 비교용 재사용**입니다. Critic와 wall-clock까지 동일 진단 조건으로 비교하려면 `-ReuseExperiment` 없이 새 on/off를 실행합니다.

완료된 기존 E1 PGCT seed1은 random schedule 기준으로도 사용할 수 있습니다.

```powershell
.\examples\run_pgct_transfer_experiments.ps1 -Modes random -Seeds 1 `
  -ReferenceExperiment "runs/sp_20261008_120638_983201"
```

기존/live 로그와 체크포인트를 읽어서 차트를 생성하는 명령:

```powershell
& "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe" `
  examples/plot_pgct_probe_charts.py `
  --experiment-dir runs/sp_20261008_120638_983201 `
  --output-dir output/pgct_transfer_review/live_spatial
```

이 명령은 원본 모델·로그를 읽기만 합니다. 기존 IPPO에는 training probe가 없으므로 같은 맵·seed의 PGCT bank로 저장 checkpoint의 forward-only JS를 재구성합니다. 저장 간격 사이의 policy 응답은 복원할 수 없습니다.

여러 training seed의 결과는 **이 한 Stage03 모델을 조건으로 한 효과**입니다. 독립적인 Stage03 사전학습 seed에도 재현되는지는 각 Stage03 checkpoint로 이 실험을 반복해야 합니다.

## 저장 모델의 공통 rollout 이력 진단

`diagnose_pgct_rollout_compatibility.py`는 학습을 하지 않는 CPU 추론 도구입니다. 한 고정 reference policy로 평가 이력을 생성한 뒤, 같은 observation·action mask 이력을 모든 비교 모델의 actor/critic에 넣습니다. 각 network는 그 동일한 이력으로 자기 hidden state를 형성하며 hidden vector를 서로 복사하지 않습니다.

```powershell
& "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe" `
  examples/diagnose_pgct_rollout_compatibility.py `
  --checkpoints `
    runs/pgct09700_physical_mixing_20260929_155338_079437/stage04_4ag_physical_home0.500/final.pt `
    runs/sp_20261008_120638_983201/e1_pgct_s1/checkpoint_00600.pt `
  --episodes 20 --horizon 500 `
  --output-dir output/pgct_transfer_review/common_rollout_stage04_e1_u600
```

출력:

- `histories.npz`: 재사용할 observation/mask/reward/phase 이력, episode 경계, 시간, 실제 종료·truncation 사유. 기본은 첫 checkpoint의 stochastic policy와 seeds5,000,001–5,000,020. `--reference-checkpoint`으로 이력 생성 정책을 명시할 수 있습니다.
- `summary.json`: pair별 recurrent/zero-hidden × masked/unmasked policy JS, 공통 이력에서의 critic prediction MSE, phase별 결과, 가중치 hash와 추론 전후 동일 여부.
- `model_N_samples.npz`: 관측 시점별·source stream별 pair JS와 critic 예측 차이. 기본은 매10step 표본이며 `--sample-interval 1`은 모든 시점을 사용합니다. 각 step에서 hidden state는 항상 전개합니다.
- `compatibility.png`: policy JS와 critic 예측 차이, 6개 pair 평균의 관계. 6개 pair는 독립 표본이 아니며 상관계수는 설명용입니다.

후속 PGCT/IPPO checkpoint도 **동일한 저장 이력**에 평가할 수 있습니다.

```powershell
& "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe" `
  examples/diagnose_pgct_rollout_compatibility.py `
  --checkpoints "완료된PGCT모델.pt" "완료된IPPO모델.pt" `
  --cache output/pgct_transfer_review/common_rollout_stage04_e1_u600/histories.npz `
  --output-dir output/pgct_transfer_review/common_rollout_on_off
```

환경·보상·맵 설정은 같아야 합니다. 같은 E1 기본 맵의 명시/암묵 generator 인자는 동일하게 취급하며, actor 입력에 영향을 주지 않는 IPPO의 communication off는 허용합니다. Reference 이력은 다른 정책에게 off-policy이므로 저장된 reference return을 다른 정책의 실제 on-policy value 정답으로 사용하지 않습니다. 이 도구는 **critic 예측 유사성**을 측정하며 **critic 정확도나 전이 유익성**을 입증하지 않습니다. 별도 stochastic on-policy MC 평가와 구분합니다. 현재 Stage04/E1 PGCT 두 모델 비교도 PGCT-on/IPPO-off 실험 결과는 아닙니다.
