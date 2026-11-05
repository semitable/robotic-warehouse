# 잠재 팀 추론 + 팀 내부 critic consensus 검증 (latent_team)

2026-10-23 Codex 추가: 개인 보상·좌표 마스크·경험 기반 A+C reward gate를
`train_from_scratch.py --reward-probe-ac`로 선택할 수 있다.
자세한 정보 접근 조건, 예측기(경험 조건부 평균이며 CNN 아님), 보류 규칙과 검증 범위는
[A+C 구현 문서](../docs/REWARD_PROBE_AC.md)를 참조한다. 아래 초기 설명의
`relevant_goal`은 정정이 필요하다: 운반 중이면 agent 팀이 아니라 운반 선반 소유 팀 goal이다.

2026-10-13 Claude Code 작성. 기존 PGCT 코드(`examples/`, `rware/`)는 수정하지 않고, 이 폴더의 새 실행기만 기존 모듈을 import해서 사용한다.

## 연구 질문

원래 목표: **팀 정보를 모르는 완전분산 agent가 국소적으로 관측한 행동으로 같은 목표의 agent(잠재 팀)를 찾고, 그 팀 안에서 critic consensus를 하여 학습을 개선한다.**

다음 두 명제를 따로 검증한다.

1. **실험 A**: 국소 행동 정보로 잠재 팀을 식별할 수 있는가? 또 같은 팀의 critic이 다른 팀보다 실제로 더 잘 호환되는가?
2. **실험 B**: 정답 팀을 알려줘도 팀 내부 critic 공유가 학습을 개선하는가?

## 이 설계에 반영한 코드상의 사실

- 관측은 **자기 팀 기준(team-relative)**이다. `requested_shelf` 채널, `relevant_goal` 채널, `self_carrying_requested_shelf`는 모두 관측하는 agent 자신의 팀 기준으로 표시된다(`rware/utils/semantic_observation.py`의 `_requested_shelf_ids_for_observation`·`_relevant_goal_cells`, `rware/multi_team_warehouse.py`의 `_requested_shelves_for_observation`). 다른 agent의 팀은 관측에 없다.
- E1 지도에서 두 팀의 목표 칸은 (4,10)과 (5,10)으로 인접해 있다. home ratio .5에서는 팀 선반이 좌우 구역에 절반씩 섞인다.
- 따라서 같은 canonical 입력을 두 actor에 넣고 비교하는 현재 PGCT JS는 "이 장면에서 **내 팀의** 요청 선반/목표가 여기 있다면 어떻게 하느냐"를 비교한다. 두 팀 agent가 모두 과제를 잘 배웠다면 팀이 달라도 응답이 같아진다. 즉 이 신호는 **구조적으로 팀을 거의 구분하지 못한다**는 가설이 생긴다. 실험 A가 이를 확인한다.
- 같은 이유로, 관측 기준에서는 다른 팀의 critic도 상당 부분 호환될 수 있다. 이 경우 "같은 팀끼리만 공유"하는 이득은 작을 수 있다. 실험 A의 critic 항목과 실험 B가 이를 확인한다.
- 트레이너에는 이미 `graph_mode=oracle/unrestricted/wrong` 비교군이 구현되어 있었지만, 기록상 smoke만 실행했고 장기 학습은 없었다(`examples/EX5_PGCT.md`).

## 실험 A — 고정 모델 평가 (학습 없음)

`exp_a_team_identifiability.py`. 각 checkpoint가 자기 stochastic masked policy로 rollout한다(기본 20회×500 step, reset seed 5,200,001–5,200,020). 평가 전후 weight hash를 확인한다. 팀 라벨은 채점에만 쓰고 점수 계산에는 쓰지 않는다.

| 항목 | 의미 | 정보원 |
|---|---|---|
| `canonical_js` | 현재 PGCT 신호. 공통 canonical 48개 입력, zero hidden | actor 응답 교환 |
| `history_js` | 두 정책을 두 agent 각자의 실제 관측 이력에서 비교. 각 network는 자기 hidden 사용 | actor 응답 교환 |
| `occupancy_js` | 방문 칸 분포 차이(공간 역할) | 위치 |
| `goal_evidence` | 관측자가 **자기 센서 범위 안에서** 본 상대의 pickup·delivery를 자기 관점으로 분류한다. 그 선반이 내 팀 요청인가, 그 목표가 내 목표인가? | 국소 관측만 |
| `critic_gap` | 각 agent 이력에서 `E[(V_i - V_j)^2]` (음수로 저장) | critic |
| `blend_gain` | `MSE(V_i, G_i) - MSE((V_i+V_j)/2, G_i)`, 초기 200 step의 유한 MC return | critic + 실제 보상 |

채점 기준:

- same-vs-cross AUROC: episode×pair 표본을 쓴다. 표본이 서로 종속이므로 기술 통계로만 본다.
- teammate recovery: 각 agent의 최고 점수 상대가 실제 팀원인 비율이다. 우연 수준은 1/3이다.
- episode bootstrap CI: 같은 팀 점수에서 다른 팀 점수를 뺀 차이의 신뢰구간이다.
- `goal_evidence`의 episode 누적 곡선: 관측이 쌓일수록 팀을 얼마나 맞히는지 본다.

기본 모델은 Stage03(E1 환경에서 평가), E1/E2 IPPO·PGCT final, Stage04 PGCT다. IPPO final이 가장 깨끗한 조건이다. 전이로 인한 유사성이 섞이지 않기 때문이다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe latent_team\exp_a_team_identifiability.py
```

출력: `output/latent_team/exp_a_<시각>/` (`summary.json`, `REPORT.md`, `identifiability.png`, 모델별 `*_scores.npz`).

결과 해석:

- `goal_evidence`만 팀을 맞히고 `canonical_js`·`history_js`는 우연 수준이라면, 행동에는 팀 정보가 있다. 다만 현재 PGCT가 보는 "공통 입력 응답 유사성"이 그 정보를 담지 못한다는 뜻이 된다. 이 경우 팀 추론 모듈을 "누구의 요청/목표를 수행하는가"라는 결과 수준 증거 위에 다시 설계해야 한다.
- `critic_gap`·`blend_gain`에서 same과 cross 차이가 없다면, 이 환경에서는 팀을 정확히 찾아도 critic 공유 대상으로서의 이점이 작다. 이 경우 실험 B의 oracle도 all-peer와 비슷할 것으로 예상된다. 그러면 팀마다 value가 실제로 달라지는 환경 설계가 먼저 필요하다.

## 실험 B — 정답 팀 critic 공유 학습 (장기 학습, 사용자 승인 후 실행)

`exp_b_oracle_consensus.py`, `run_exp_b.ps1`. 기존 E1(또는 E2) PGCT/IPPO seed1과 같은 Stage03 weights, 환경, PPO 설정, 예산(4,000,768 step), 평가 seed, **같은 canonical probe bank**를 쓴다. 기존 critic peer loss의 donor 집합만 바꾼다.

| mode | trainer `graph_mode` | donor |
|---|---|---|
| `oracle` | `oracle` | 물리 이웃 중 같은 팀(정답 라벨). 팀 추론의 상한 |
| `all` | `unrestricted` | 물리 이웃 전부 |
| `wrong` | `wrong` | 물리 이웃 중 다른 팀만 |

- 기존 PGCT와 같은 시점에 전이를 시작하도록 처음 100 update 동안 전이를 보류한다(`--transfer-warmup-updates`).
- 실행 전에 새 설정이 기존 PGCT run과 `graph_mode` 하나만 다른지 검사한다.
- `transfer.jsonl`에 update별 실제 receiver×donor allocation을 남긴다. 팀/물리 범위를 벗어난 전이가 있으면 즉시 중단한다.
- 완료된 기존 PGCT/IPPO run은 읽기만 해서 `comparison.md`로 비교한다.
- W&B와 영상은 기본적으로 끈다(`-Track`, `-SaveVideo`로 켤 수 있음). 영상은 평가 에피소드를 렌더링만 하므로 평가 수치는 바뀌지 않는다.
- 예상 시간: 기존 E1 run 속도로 run당 약 30시간. 기본 oracle+all 순차 실행이면 약 60시간이다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\latent_team\run_exp_b.ps1 -DryRun                 # 계획만
.\latent_team\run_exp_b.ps1                         # E1 seed1: oracle, all
.\latent_team\run_exp_b.ps1 -Modes oracle,all,wrong # wrong 포함
```

출력: `runs/lt_b_<시각>/`. 기존 폴더가 있으면 덮어쓰지 않고 거부한다.

결과 해석(단일 seed는 방향 확인용이다):

- oracle도 IPPO보다 낫지 않다면, 팀 추론을 개선하기 전에 consensus 방식이나 환경의 팀-가치 구조를 먼저 다시 봐야 한다.
- oracle은 낫지만 PGCT는 낫지 않다면, 팀 추론이 병목이다.
- oracle과 all이 비슷하다면, 이 환경에서는 팀 선택 자체가 성능에 중요하지 않다.

## 추가 검토 (2026-10-13, 외부 검토 의견 반영)

### 국소 증거가 팀 ID·내부 상태를 간접 사용했는가?

- `exp_a`는 env 이벤트(pickup/delivery)와 관측자 **자신의** 팀 요청·목표 집합으로 증거를 분류한다. 상대 agent의 팀 ID는 읽지 않는다.
- `verify_evidence_observability.py`로 같은 판정이 관측자 자신의 post-step 관측 텐서에 그대로 보이는지 확인했다. 결과는 `output/latent_team/exp_a_20261013/observability_check.json`이다. E1 IPPO, E2 IPPO, Stage04 각 5×500 step에서 pickup 985건, delivery 1,161건을 확인했고, 모든 건에서 해당 칸이 "agent+loaded"였다. `requested_shelf`/`relevant_goal` 채널 값도 판정과 100% 일치했다.
- **남는 가정**: 관측의 agent 칸은 익명이다. 본 사건을 특정 이웃 j에게 귀속하려면 이웃 식별이 필요하다. 이는 PGCT가 통신 상대를 아는 것과 같은 수준의 가정이다. 또 이 환경은 팀을 요청 선반·목표로 정의하므로, 이 증거는 "팀을 구분할 정보가 국소적으로 존재한다"는 확인이다. 일반적인 잠재 팀 추론 알고리즘의 성과로 볼 수는 없다.

### blend gain의 정의와 한계

`blend_gain(i←j) = MSE(V_i(h_i), G_i) − MSE((V_i(h_i)+V_j(h_i))/2, G_i)`. h_i는 agent i 자신의 rollout 관측 이력이고, 각 network는 자기 hidden을 사용한다. G_i는 i의 실제 보상(shaping 포함)의 할인 return이며, 500-step 중 초기 200 step만 평가한다(유한 horizon, zero tail). 양수면 j 예측을 섞을 때 i의 return 예측 오차가 줄어든다는 뜻이다.

이 값은 학습된 critic을 **정적으로 평균**했을 때의 오차 변화다. 학습 중 peer loss가 주는 regularization이나 표본 효율 효과는 측정하지 않는다. 따라서 이 값이 낮아도 학습 중 consensus가 해롭다고 결론 내리지 않는다. 실제 효과는 실험 B로만 판단한다.

### `team_reward_mode="team"`으로 바뀌는 것 (코드 확인, 실행 안 함)

- `_reward_delivery`/`_reward_shelf_return`: delivery(1.0)와 return(1.0) 보상이 운반자 대신 **선반 팀의 모든 팀원**에게 간다. pickup·progress·penalty 등 shaping은 계속 개인별이다.
- critic target: 팀원은 같은 task-return 성분을 공유하고 다른 팀은 공유하지 않는다. 팀원 간 요청 경쟁도 사라진다. "공통 목표 팀"의 정의에 맞는다.
- 그대로 Stage03에서 이어갈 수 없는 이유:
  - task return 규모가 약 2배가 되어 기존 critic 보정이 깨진다.
  - 팀원의 배송이 내 advantage에 섞여 credit assignment 잡음이 생긴다.
  - 행동 유인이 바뀐다.
  - 따라서 team reward 하의 공통 사전학습 → IPPO/oracle/all 분기로 다시 정렬해야 한다.
- **주의**: 관측은 여전히 자기 팀 기준이고 두 팀 과제는 대칭이다. 그래서 같은 입력 x에서 team-0 critic과 team-1 critic이 배우는 함수는 team reward에서도 거의 같아질 수 있다. 관측에서 팀원과 상대 팀 agent는 구분되지 않는다. PGCT peer loss는 공통 probe 입력에서 critic **함수**를 맞추므로, 팀 보상 공유만으로 "다른 팀 critic은 틀린 target"인 상황이 생긴다는 보장은 없다. 팀별 value 함수가 실제로 갈라지려면 다음 중 하나가 필요할 수 있다(가설).
  - 팀 소유를 절대 기준으로 보여주는 관측(예: 팀 색 채널)
  - 비대칭 팀 과제
  - 현재 env에는 관측을 절대 기준으로 바꾸는 옵션이 없다. 새 wrapper가 필요하다.

### 소규모 oracle 실험

트레이너에는 학습률 schedule이 없다. `total_timesteps`는 update 횟수만 정한다. 따라서 `--timesteps 1024000`의 oracle run은 기존 E1 PGCT/IPPO seed1의 첫 1,024,000 step과 matched prefix다. 비교표는 기준 run을 같은 step에서 자른다. 기존 1M 구간 기준선은 delivery AUC IPPO 43.828, PGCT 44.605다. 예상 소요는 oracle 한 run당 약 7–8시간이다.

```powershell
.\latent_team\run_exp_b.ps1 -Modes oracle -TotalTimesteps 1024000
```

## 처음부터 학습 (커리큘럼 없음) — `train_from_scratch.py`, `run_from_scratch.ps1`

기존 E1/E2/E3 결과는 Stage03까지 약 6M step 커리큘럼을 거친 모델에서 분기한 조건부 비교다. 이 실행기는 체크포인트와 커리큘럼 없이 무작위 초기화로 학습한다.

| mode | 내용 |
|---|---|
| `ippo` | 통신·critic 공유 없음 |
| `pgct` | 기존 JS gate(policy graph, 물리 이웃) |
| `oracle` | 물리 이웃 중 같은 팀 critic 공유(정답 라벨) |
| `all` | 물리 이웃 전부와 공유 |
| `wrong` | 물리 이웃 중 다른 팀과만 공유 |

- 같은 (맵, seed)의 모든 mode는 같은 환경(기본 E1, home .5, 요청 4, individual reward), PPO 설정, 예산, 평가 seed, **같은 무작위 초기 가중치**(`initial_weights_sha256.txt`로 확인), **같은 canonical probe bank**(계획 시 1회 생성)를 쓴다. critic peer loss의 donor 집합만 다르다.
- 설정은 기존 E1 PGCT run에서 가져온다. 단, 이전에 처음부터 학습에 성공한 run의 lr 2e-4와 entropy .05를 쓴다. 기본 예산은 10M step(이전 성공 run 9.93M)이다. 모두 옵션으로 바꿀 수 있다.
- 전이는 PGCT warm-up과 같은 100 update 이후 시작한다. `transfer.jsonl`에 실제 allocation을 남기고, 팀·물리 범위를 벗어나면 중단한다.
- 학습 데이터:
  - `metrics.jsonl`(학습 지표, 51,200 step마다 20×500 평가)
  - `transfer.jsonl`, `communication.jsonl`, `spatial.jsonl`
  - `checkpoint_*.pt`(기본 250 update마다), `final.pt`
- `comparison.md`: mode별 delivery AUC, 마지막 10회 평가, 10/20/30/40 delivery 도달 step, seed 간 표준편차, 같은 초기 가중치 여부.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 1) 학습 가능성 pilot: IPPO와 oracle, 2M step, 병렬
.\latent_team\run_from_scratch.ps1 -Modes ippo,oracle -TotalTimesteps 2000000 -Parallel

# 2) 본 비교: 4개 mode, 10M step, seed 1 (이후 -Seeds 2,3)
.\latent_team\run_from_scratch.ps1 -Modes ippo,pgct,oracle,all -Seeds 1 -Parallel

# 진행 확인과 비교표 (<폴더>는 실행 시 출력되는 runs\scratch_<시각>)
Get-Content runs\scratch_<시각>\e1_oracle_s1.log -Tail 3
C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe latent_team\train_from_scratch.py --compare-only --plan runs\scratch_<시각>
```

1단계가 끝나면 2단계를 자동으로 시작하는 파이프라인:

```powershell
.\latent_team\run_scratch_pipeline.ps1 -Detach
```

- 기본 구성: 1단계는 ippo+oracle 2M, 2단계는 ippo/pgct/oracle/all 10M, 둘 다 seed 1, 병렬.
- 1단계의 모든 job이 `complete`일 때만 2단계를 시작한다. 1단계의 학습 성능을 판단하는 것은 아니다.
- `-Detach`로 실행하면 숨은 PowerShell에서 돌아가므로 창을 닫아도 계속된다. 다만 PC 재부팅에는 대비되지 않는다.
- 출력은 `runs/scratch_<시각>_stage1`, `_stage2`, `_pipeline.log`다.
- 단계별 mode·step·seed는 `-Stage1Modes`, `-Stage2Timesteps` 등의 옵션으로 바꾼다.

`-Parallel` 없이 실행하면 같은 PowerShell 창에서 순차 실행한다. `-Wait`는 병렬 job이 모두 끝날 때까지 기다린다. `-PlanOnly`는 계획과 bank만 만든다. 기존 E1 단일 프로세스 속도(약 37 step/s)로 보면 run당 2M은 약 15시간, 10M은 약 75시간이다. 병렬 실행 시 각 run 속도는 더 느려질 수 있다.

## W&B, 학습 구간, 비교 알고리즘 (2026-10-13 추가)

- **W&B**
  - `-Track`으로 켠다. project는 `rware-from-scratch`, run 이름은 `e1_<mode>_s<seed>`다.
  - group은 출력 폴더 이름(stage별)이고, tag에 stage·맵·seed·mode가 들어간다.
  - config에는 trainer 설정 전체와 algorithm·stage·seed·home ratio·env_kwargs·예산이 들어간다.
  - 모든 run이 `step=env_steps`로 기록되므로 W&B의 Step 축이 env step이다.
  - gate 지표: `update/gate_ratio`, same/cross arc 수.
  - `-Detach`에서도 `_netrc` 로그인으로 동작함을 확인했다.
  - PowerShell 파이프라인 로그는 UTF-8로 남는다.
- **학습 구간 근거**
  - 이전 from-scratch PGCT(home 1.0, complete graph, 로컬 W&B 기록 `run-bhh1l6x8`)에서 pickup은 약 1.5M, delivery는 1.7M에 시작했다.
  - 평가 기준 20 이상은 2.3M, 40 이상은 3.3M이었고, 6–9M에 56–60으로 수렴했다.
  - 현재 home .5는 아직 기록이 없다.
  - 그래서 기본값을 Stage1 4M(pilot), Stage2 15M으로 바꿨다.
  - Stage1은 Stage2의 초기화가 아니며 같은 mode의 앞 구간과 겹친다. 생략하려면 `-SkipStage1`을 쓴다.
- **비교 알고리즘**: `mappo`, `seps`, `hetdps` mode. 구현 차이와 비용은 [BASELINES.md](BASELINES.md)에 있다.

```powershell
# 예: Stage1 생략, 7개 알고리즘 15M, W&B, 백그라운드 (7 프로세스 → 스레드 2개씩 권장)
.\latent_team\run_scratch_pipeline.ps1 -Detach -Track -SkipStage1 -ThreadsPerJob 2 `
  -Stage2Modes ippo,pgct,oracle,all,mappo,seps,hetdps -Stage2Seeds 1
```

## E1-F 환경과 probe bank v2 (2026-10-15 추가, 옵션)

- **`--env-variant e1f`**(PowerShell `-EnvVariant e1f`): E1 맵에서 팀 목표만 두 아래 모서리로 옮긴다(빨강 (0,10), 파랑 (9,10)).
  - 관측(팀 기준), 보상, 선반 배치는 그대로다.
  - 환경의 기존 `team_goal_positions` 설정만 쓰며, 환경 코드는 바꾸지 않았다.
- **`--probe-bank v2`**(`-ProbeBank v2`): `probes_v2.py`가 만드는 새 bank.
  - 학습 env 설정 그대로 상태를 배치하고 실제 관측 함수로 probe를 뽑는다. 역할별 고유 입력만 남기고, 시나리오가 선언한 조건(적재·반납·목표 가시성)을 관측이 위반하면 거부한다.
  - **gate**(PGCT JS용, 48개): 요청 선반 적재 + **어느 팀 목표도 안 보이는** 상태. 같은 입력이 두 팀 모두에게 유효하며, 목표가 분리된 E1-F에서만 팀에 따라 정답 방향이 달라진다.
  - **transfer**(critic 증류용, 64개): 탐색 14, 적재·목표 보임 16, 적재·목표 안 보임 16, 반납 16, 목표 도착 2.
  - Oracle·All·PGCT 모두 증류는 transfer bank로 한다(`transfer.jsonl`의 `distill_probe_count`=64).
  - 모든 agent가 같은 결정적 규칙으로 생성하므로 관측 교환은 없다.
- **기본값**은 지금과 같다(`e1`, `canonical`). 실행 중인 `scratch_20261015_093349` run에는 영향이 없다.
- **확인한 것**: `output/env_view_20261015/probes_v2/`(그림·요약), smoke `runs/smoke_e1f_v2_20261015`, `tests/test_latent_team.py`의 v2/E1-F 테스트(모서리 목표에서 실제 배송 확인 포함), 전체 183 passed.

```powershell
# E1-F + v2 probe, 4개 비교군, 15M (승인 후 실행)
.\latent_team\run_scratch_pipeline.ps1 -Detach -Track -SkipStage1 -EnvVariant e1f -ProbeBank v2 -Stage2Seeds 1
```

## E1-F 실험 실행기 (2026-10-18)

E1-F에서 팀별 목표가 실제로 다를 때 행동·probe·critic 전이가 어떻게 달라지는지 본다. 두 단계로 나눠 각각 따로 시작한다(승인 후).

| 단계 | 명령 | 내용 | 프로세스 |
|---|---|---|---|
| pilot | `.\latent_team\run_e1f.ps1 -Phase pilot -Track` | IPPO만, 개인 보상 / 팀 보상 각 1개, 기본 5M | 2 |
| main | `.\latent_team\run_e1f.ps1 -Phase main -Track` | IPPO/PGCT/Oracle/All(canonical bank) + PGCT(v2 bank), 기본 10M | 5 |

- **pilot의 질문**: 두 팀이 서로 다른 운반 방향을 배우는가? 그 차이가 probe에 보이는가? 결과를 보고 main 진행 여부와 보상 방식(`-TeamReward individual|team`)을 정한다.
- **main의 canonical vs v2 PGCT**: 같은 E1-F 환경에서 probe bank만 다르므로, probe 효과와 환경 효과를 분리한다. canonical probe 생성기도 E1-F의 모서리 목표를 그대로 쓴다(확인함).
- **팀 보상**: `--team-reward team`(`-TeamReward team`). 배송·반납 보상이 배송한 팀 전원에게 간다. pickup·진행 등 shaping 보상은 개인 그대로다. 환경의 기존 `team_reward_mode` 설정만 쓴다. 기본값은 지금과 같은 `individual`이다. job 이름에는 `_teamrew`가 붙는다.
- **분석**(읽기 전용, 학습 중에도 가능): `python latent_team/e1f_team_divergence.py --run runs/<plan 폴더>`
  - checkpoint마다 확률적 rollout(기본 5×500)을 돌려, 요청 선반을 배송하러 갈 때 왼쪽 이동 비율(`left_share`)을 agent별로 잰다. E1-F에서 팀을 아는 정책이면 빨강(왼쪽 목표)은 1, 파랑(오른쪽 목표)은 0에 가깝다. E1 모델로 재면 두 팀 모두 약 0.5다(목표가 붙어 있음).
  - 같은 checkpoint에서 v2 gate bank와 canonical bank의 같은 팀/다른 팀 JS를 잰다.
  - 결과는 `output/e1f_divergence_<run>/divergence.json`, `divergence.png`에 저장된다.
- **확인한 것**: smoke `runs/smoke_e1f_runner_20261018_*`(pilot 2개, main 5개 모두 complete, 팀 보상·모서리 목표가 checkpoint 설정에 반영됨), 분석 스크립트 smoke와 E1 checkpoint 검증, 테스트 2개 추가(전체 185 passed).

## Random-gated 대조군 (2026-10-19, 옵션)

"어떤 상대를 고르느냐"가 중요한지, 아니면 "얼마나 공유하느냐"만 중요한지를 가르는 대조군이다(D44, D54). Oracle > All 비교는 전이량 차이와 섞이므로, 전이량을 PGCT 실측에 맞춘 무작위 donor와 비교해야 "같은 팀이라서" 생긴 효과를 볼 수 있다.

- **동작**(`latent_team/random_gated.py`)
  - 매 update마다 참조 PGCT run의 `transfer.jsonl`에서 같은 update의 실제 전이 arc 수(`actual_allocation`의 0이 아닌 칸)를 읽는다.
  - 그 수만큼 지금 물리 이웃(다른 mode와 같은 범위) 중에서 receiver→donor arc를 균등 무작위로 고른다. 팀 label은 읽지 않는다.
  - 가중치는 trainer처럼 receiver별로 합이 1이다.
  - 물리 이웃이 모자라면 부족분을 다음 update로 넘긴다(최대 12개). 그래서 누적 전이량을 맞춘다. 갚지 못한 양은 `random_unmet_total`로 기록한다.
  - 무작위 추출은 run seed에서 만든 별도 난수기를 쓰므로 trainer의 난수 흐름은 바뀌지 않는다.
  - trainer는 All과 같이 `graph_mode=unrestricted`로 peer loss를 준비하고, wrapper가 donor 집합만 바꾼다. trainer·환경 파일은 수정하지 않았다.
- **실행**: `--modes random --random-reference <PGCT transfer.jsonl>`. 참조는 같은 rollout 길이에 run 예산 이상의 update를 덮어야 한다(plan 단계에서 검사). 그래서 본 학습은 참조 PGCT가 끝난 뒤에 할 수 있다. 예: E1-F main PGCT가 10M을 마친 뒤 `runs/e1f_20261018_124343_main_canonical/e1f_pgct_s1/transfer.jsonl`.
- **기록**: `transfer.jsonl`에 `random_target_arcs`, `random_used_arcs`, `random_candidate_arcs`, `random_debt`, `random_unmet_total`이 추가된다.
- **한계**: arc 수를 맞춘다. receiver별 가중치 합이 1이라 peer loss 크기는 "donor가 있는 receiver 수"에 비례하는데, 이 값(`update/pgct_peer_weight`)은 PGCT와 다를 수 있다. 사후에 함께 비교한다.
- **확인**
  - 테스트 3개를 추가했다(전체 188 passed).
  - 실제 E1-F PGCT 기록 1,650 update를 재생하면 목표 8,686 arc를 그대로 사용한다. 같은 팀 비율은 PGCT .329, 무작위 .362다.
  - smoke `runs/smoke_random_gated_20261019`(All·Random, 64 step, v2 bank, 참조는 PGCT 101–140 update를 다시 번호 매긴 scratch 파일): Random update 1은 후보 12개 중 4개, update 2는 목표 8에 후보 6개라 6개를 쓰고 2개를 넘겼다. peer loss가 작동했고 초기 가중치 hash는 All과 같다.

## E1-F 대체 후보 환경 (2026-10-19, 옵션, D62)

`latent_team/env_candidates.py`에 정의했고, `--maps e1 --env-variant <후보>`로 쓴다. 환경 코드는 수정하지 않고 기존 인자만 쓴다.

**모든 후보의 공통 설정**
- E1-F처럼 팀 목표를 아래쪽 양 모서리로 나눈다.
- **팀 보상**(배송·반납 보상을 팀 전원이 받음. shaping은 개인)을 쓴다. 후보 설정이 `--team-reward`보다 우선한다.
- **seed가 짝수면 목표 좌우를 바꾼다.** 좌우 배치에 대한 seed 간 강건성 시험일 뿐이다. seed 하나 안에서는 이동 방향과 팀이 여전히 대응하므로, 방향과 독립적인 팀 발견을 주장할 근거가 되지 않는다(D63 정정).

| 후보 | 맵(높이×너비) | agent/팀 | 센서·통신 | 요청/팀 |
|---|---|---|---|---|
| `c1_e1f_team` | 11×10 | 4/2 | 5·5 | 2 |
| `c2_e1f8_team` | 11×16(shelf_columns 5) | 8/2 | 5·5 | 4 |
| `c3_wide_s3_team` | 20×16(shelf_columns 5, rows 2) | 4/2 | 3·3 | 2 |

- trainer는 `cfg.sensor_range`와 `cfg.agent_count`로 환경 인자를 덮어쓴다. 그래서 후보별 `CONFIG`가 이 두 값도 맞춘다(처음에는 이 때문에 C2가 4명으로 돌고 C3에서 bank 크기 오류가 났다).
- 8명 실행에서는 `examples/train_pgct_spatial_experiments.SpatialMonitor`(4×4 고정)를 실행 중에만 인원수에 맞춘 `NAgentSpatialMonitor`로 바꿔 끼운다. 원 파일은 수정하지 않았다. `spatial.jsonl`의 `mean_physical_neighbors`는 4로 나누므로 8명이면 실제의 2배로 읽힌다. 원자료인 `geometric_candidate_step_counts`는 정확하다.
- `exp_b_oracle_consensus.py`의 4×4 고정 부분과 `random_gated.py`를 인원수에 맞게 일반화했다. 4명일 때 동작은 같다.
- **smoke**: `runs/smoke_candidates_20261019b/{c1,c3}*`, `runs/smoke_candidates_20261019c/c2*`. IPPO/PGCT/Oracle/All/Random과 PGCT-v2가 모두 complete다.
  - 같은 폴더 이름의 실패 기록(20261019의 C3, 20261019b의 C2)과, 설정을 고치기 전 기록(20261019의 C2, 실제로는 4명)은 보존용이다.
- 테스트 1개를 추가했다(전체 189 passed). **학습은 하지 않았다.** 다음 단계는 후보별 IPPO 2M pilot이고, 승인이 필요하다.
- **pilot 실행기**: `latent_team/run_candidate_pilot.ps1`(IPPO, seed 1, 기본 2M, 후보별 plan 폴더 `runs/<Name>_<후보>`, PID는 `runs/<Name>_pids.json`).
  - Claude 앱 밖에서 직접 연 PowerShell에서 실행한다: `powershell -ExecutionPolicy Bypass -File .\latent_team\run_candidate_pilot.ps1 -Track`.
  - `-PlanOnly`로 계획만 만들 수 있다. 확인용 `runs/plancheck_*`는 계획과 bank만 있고 학습은 없다.

### 현재 후보 A/B/C와 진단 도구 (2026-10-19, D63·D64)

- 후보: `a_e1f4_team`(11×10, 4명, 통신 5), `b_e1f6_comm3`(11×10, 6명, 통신 3, **1순위**), `c_e1f6_wide_comm3`(11×16, 6명, 통신 3). 모두 팀 보상이다.
- wrapper(`exp_b_oracle_consensus.matched_transfer`)의 이웃 후보·검사·Random-gated는 환경의 `communication_range`를 쓴다.
- 연결률: `python latent_team/connectivity_check.py --variants b_e1f6_comm3 a_e1f4_team [--checkpoint ck.pt]`
- 보상 상관 기준선: `python latent_team/reward_correlation_baseline.py --checkpoint <ck.pt> ...` (팀 보상 환경의 probe 비교에 반드시 포함한다)
- pilot: `powershell -ExecutionPolicy Bypass -File .\latent_team\run_candidate_pilot.ps1 -Track`. 기본값은 B·A × seed 1·2이고, Claude 앱 밖에서 실행한다.

### pickup 감점 제거 변형 (2026-10-21, D69·E45)

- `a_e1f4_team_nopen`, `b_e1f6_comm3_nopen`: A·B와 같고, `unrequested_shelf_pickup_penalty`와 `premature_drop_penalty`만 0이다. 초기 탐색에서 TOGGLE이 가장 손해인 행동이 되어 0.1–0.2M에 pickup이 사라지는 문제(E44)의 인과 확인용이다.
- pilot(앱 밖 PowerShell): `powershell -ExecutionPolicy Bypass -Command "& '.\latent_team\run_candidate_pilot.ps1' -Candidates b_e1f6_comm3_nopen,a_e1f4_team_nopen -TotalTimesteps 1000000 -Track"`. `-File`로 실행하면 쉼표로 이은 후보 목록이 문자열 하나로 넘어가 ValidateSet에서 거부된다(T093).
- 결과(E46): 감점 0이면 TOGGLE이 무비용이 되어 episode당 100–300회 남발한다.
- `b_e1f6_comm3_pen005`: B와 같고 두 감점만 .005다. 무작위 정책에서 TOGGLE이 가장 손해인 행동이 아니게 되는 경계(.0056–.0059, E47) 바로 아래 값이다.
  - pilot: `powershell -ExecutionPolicy Bypass -Command "& '.\latent_team\run_candidate_pilot.ps1' -Candidates b_e1f6_comm3_pen005 -TotalTimesteps 1000000 -Track"`

### 식별 가능성 2×2 (D74)

- 기반은 B pen005(6 agent 3+3, 센서 5, 통신 3, 두 감점 .005)로 고정하고, goal 배치와 보상만 바꾼다. 모서리·팀 칸은 `b_e1f6_comm3_pen005`다.
  - `b6_corner_indiv_pen005`: 모서리 goal, 개인 보상(E1-F형)
  - `b6_adjacent_team_pen005`: 인접 goal (4,10)/(5,10), 팀 보상
  - `b6_adjacent_indiv_pen005`: 인접 goal, 개인 보상(E1형)
  - goal 쪽은 짝수 seed에서 바뀐다. 계획 대조에서 B pen005와 다른 값은 `team_reward_mode`와 `team_goal_positions`뿐이다.
- 대기열 launcher `latent_team/run_queue.ps1`: 후보마다 계획을 만들고, 이 컴퓨터의 학습 프로세스(`train_from_scratch.py --run-job`, 다른 곳에서 띄운 것 포함)가 `-MaxConcurrent`개 미만일 때만 다음 job을 띄운다. 출력 폴더가 이미 있는 job은 다시 띄우지 않는다. 같은 `-Name`으로 다시 실행하면 계획 단계에서 멈춘다(기존 폴더 보호). 앱 밖 PowerShell에서 실행하고, 마지막 job이 시작될 때까지 창을 닫지 않는다.
  - `powershell -ExecutionPolicy Bypass -Command "& '.\latent_team\run_queue.ps1' -Candidates b6_corner_indiv_pen005,b6_adjacent_team_pen005,b6_adjacent_indiv_pen005 -TotalTimesteps 1000000 -MaxConcurrent 4 -Track"`
- 감시·판정·보고 `latent_team/pilot_verdict.py`: plan의 job이 모두 complete가 될 때까지 기다린 뒤 다섯 기준(T104 숫자, 4번은 마지막 0.1M 배송 20건 미만이면 보류)을 판정한다. 결과는 `output/pilot_verdicts/<plan>/verdict.{json,md}`에 쓴다. 학습은 시작하지 않는다.
  - `python latent_team/pilot_verdict.py --plan runs/pilot_20261022_112521_b_e1f6_comm3_pen005 --spam-reference runs/pilot_20261021_133652_b_e1f6_comm3_nopen --compare runs/pilot_20261019_180044_b_e1f6_comm3 runs/pilot_20261021_133652_b_e1f6_comm3_nopen`
- 비교 대상은 기존 `runs/pilot_20261019_180044_*`의 같은 seed 첫 1M이다. 학습률 일정이 없어 같은 step에서 바로 비교할 수 있다.

## 10M 재실행·최근 영상 10개 보존 (2026-11-04)

앱 밖 PowerShell에서 `./latent_team/run_main_amax_v2.ps1` 실행. 넓은 지도·개인 보상·r_c=5,
PGCT/ IPPO/ Oracle/ AllK Max-GAE × seed 1–3, action_max·mirror 없음, 10M, 최대 4개 동시 실행이다.
W&B 프로젝트는 `rware-main-amax-v2`다. `-PlanOnly`를 주면 학습을 시작하지 않는다.

- 새 옵션 `train_from_scratch.py --save-video --video-keep-last 10`은 **job별 최근 10개**만
  `videos/`에 보존한다. `final.mp4`도 10개에 포함한다. 기존 실험에는 소급 적용하지 않는다.
- 수치 지표는 W&B에 실시간으로 기록한다. 영상은 학습 종료 시 남아 있는 최대 10개만
  `eval/tail_01`–`eval/tail_10`으로 한 번 업로드한다. 학습 중 영상은 로컬 `videos/`에서 확인한다.
- 정상 종료와 Python 예외/Ctrl+C 시 업로드를 시도한다. `Stop-Process`, 창 강제 종료, 시스템 종료는
  업로드를 보장할 수 없다. 이때도 저장된 로컬 최근 10개는 남는다.
- W&B 로컬 데이터는 `wandb_data/<고유 ID>/`로 분리하여 긴 실험 폴더명에 따른 Windows 경로
  제한을 피한다. 실제 경로와 영상 목록·업로드 상태는 job의 `video_retention.json`에 기록한다.
- 이 옵션을 생략하면 기존 동작을 유지한다. trainer·PPO·환경·gate는 바꾸지 않는다.

## 아직 하지 않은 것

- 추론된 팀(learned-team) 모드는 구현하지 않았다. 실험 A에서 쓸 만한 팀 신호가 확인된 뒤에 설계한다.
- 실험 B 장기 학습은 실행하지 않았다. 검증은 `runs/lt_b_smoke_20261013`(oracle/all/wrong 각 64-step)과 `tests/test_latent_team.py`뿐이다.
