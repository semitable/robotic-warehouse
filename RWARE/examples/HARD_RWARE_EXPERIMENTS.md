# 분리된 장애물·배송 제약 RWARE 환경

`run_hard_rware_experiments.ps1`은 기존 trainer를 재사용하는 **별도 실행기**다. 원래 환경·실행기·PPO·PGCT gate·probe 파일을 수정하지 않았으며 기존 학습을 중단하지 않는다. 새로운 환경 ID는 `rware-constrained-4ag-2teams-v0`이고 이 실행기에서 명시적으로 등록한다.

## 네 환경

| Profile | 실제 제약 | 공통 조건 |
|---|---|---|
| `control` | 장애물·dock 대기 없음 | E2와 같은 16×16, 선반108개, agent4개, 2팀 |
| `detour`(기본) | 아래쪽 교차로 (3,14)/(12,14)에 벽2개를 놓아 일부 적재 경로를 우회 | 중앙 통로는 유지 |
| `bottleneck` | y=7 통로의 14개 cell을 벽으로 막고 x=3·12의 1-cell 통로 두 곳만 개방 | 선반 수·요청 수·보상 계수 유지 |
| `dock` | bottleneck + 배송 성공 후 해당 목표 지점을 8 step 동안 적재 robot이 새로 진입할 수 없음 | 빈 robot 통과와 기존 carrier의 퇴장은 허용 |

벽은 원래 통로에만 놓는다. 원래 shelf 위치·수와 목표 위치 (7,15)/(8,15)는 같다. 전체 적재 가능 통로 그래프와 모든 shelf home의 접근성을 검사한다. 모든 cell의 경로가 연결돼도 여러 robot이 충돌 없이 협력해야 하므로 성공이 자동으로 보장되지는 않는다.

벽이 있는 cell로의 FORWARD는 물리적으로 NOOP 처리된다. Mask를 껐을 때는 기존 NOOP 비용과 추가 constraint penalty .005가 부과된다. Mask를 켠 기본 학습에서는 벽·busy dock 진입을 사전에 차단한다. 기존 collision resolver를 그대로 사용하므로 원래 동시 이동·교환 금지 규칙도 유지한다.

Dock cooldown은 step 종료 때 감소하고, 그 step에서 새 delivery가 생기면8로 설정한다. 이후8개의 환경 step 동안 적재 robot의 신규 진입을 막는다. 화면에는 남은 숫자가 표시된다. 관측에는 timer 숫자 전체를 추가하지 않고 로컬 목표 지점의 현재 가용성만 표시한다.

## 실행

기본은 동일한 기존 Stage03 checkpoint에서 `detour` PGCT/IPPO seed1을 각4,000,768 환경 step 학습한다. 두 분기의 초기 actor/critic weights는 같으며 optimizer와 graph는 초기화한다. 학습 코드와 입력 차원이 호환돼 기존 가중치를 사용할 수 있지만, 장애물에 대한 즉시 일반화는 보장되지 않는다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 완만한 우회 환경: PGCT/IPPO matched 학습, 두 조건 순차 실행.
.\examples\run_hard_rware_experiments.ps1

# 강한 병목 환경 또는 병목 + 배송 지점의 8-step 대기.
.\examples\run_hard_rware_experiments.ps1 -Profiles bottleneck
.\examples\run_hard_rware_experiments.ps1 -Profiles dock

# 장애물 효과와 dock 제약 효과를 단계별 비교: 4 profiles × 2 modes.
.\examples\run_hard_rware_experiments.ps1 -Profiles control,detour,bottleneck,dock

# 반복 seed를 추가하면 계산 예산도 비례해 늘어남.
.\examples\run_hard_rware_experiments.ps1 -Profiles detour -Seeds 1,2,3

# 학습 전 확인. 각 실행은 새 출력 폴더만 사용하며 덮어쓰기를 거부.
.\examples\run_hard_rware_experiments.ps1 -PreviewOnly
.\examples\run_hard_rware_experiments.ps1 -Profiles control,detour,bottleneck,dock -DryRun
.\examples\run_hard_rware_experiments.ps1 -Profiles control,detour,bottleneck,dock -SmokeTest

# W&B/영상은 명시적으로 켠다. 기본은 둘 다 off.
.\examples\run_hard_rware_experiments.ps1 -Profiles bottleneck -Track -SaveVideo
```

`-InitCheckpoint`, `-OutputDir`, `-TotalTimesteps`, `-Requests`도 지정할 수 있다. 기본 requests4를 바꾸면 난이도 외 작업량도 달라진다. 요청 수가 늘어난다고 항상 더 어려운 것은 아니다. Profile의 geometry는16×16으로 고정한다.

기본 출력은 짧은 `runs/hard_<timestamp>/`다. 원래 실행기의 출력 폴더와 겹치지 않는다. 같은 경로가 있으면 덮어쓰지 않고 종료한다. 학습은 기존 실행과 자원을 공유하므로 장기 실행 시작 시점을 사용자가 선택한다. 이 환경 개발 과정에서는 CPU smoke만 실행했고 추가4M 학습은 시작하지 않았다.

## 알고리즘·관측·평가 조건

- 기존 PPO 설정과 PGCT의 canonical probe48개, zero hidden, unmasked JS, EMA < .12, warmup100, coefficient .01을 유지한다. Smoke는 기존 smoke 규칙과 동일하게 축소한다.
- 기존 probe builder는 일반 MultiTeamWarehouse를 만들기 때문에 새로운 환경 인자3개만 temporary adapter에서 제거한다. 벽·dock 전용 probe를 새로 넣지 않는다. **네 profile의 canonical bank가 기존 E2 bank와 tensor 단위로 같다는 테스트를 통과했다.**
- Semantic observation은20 ego +13×11×11 =1593차원을 유지한다. 기존 out-of-bounds channel0에 내부 벽도 표시하고, 벽 cell의 나머지 channel은0이다. Busy goal은 goal channel11을 유지하되 available relevant-goal channel12를0으로 표시한다.
- 이 채널 의미 확장은 새 실행기에만 적용된다. 원래 wrapper와 기존 학습의 관측은 바뀌지 않는다. 새 환경은 반드시 전용 실행기의 `constrained_runtime()`와 semantic wrapper로 사용한다. 원래 trainer를 새 ID만 지정해 직접 실행하면 필요한 어댑터가 설치되지 않는다.
- Action mask는 기존 mask와 새 로컬 제약을 교집합으로 적용한다. 인접 cell과 현재 local goal 가용성 이상의 정보를 agent에게 주지 않는다.
- 통신은 기존 Chebyshev range5다. 벽이 통신의 line-of-sight를 차단하는 모델은 추가하지 않았다.
- 보상 shaping의 기존 Manhattan distance도 유지한다. 장애물 우회와 shaping 방향이 다를 수 있으므로 delivery/return과 실제 이동 경로를 함께 해석한다.
- Task 평가는 공통 reset seeds5,000,001–5,000,020, deterministic masked policy,20×500이다. 새로운 실행기는 step0의 고정 평가도 저장한다.
- 기본 초기 모델은 PGCT Stage03이므로 이 실험도 **해당 사전학습 모델 이후 추가 전이 효과**를 본다. 처음부터 독립 학습한 IPPO의 대조군은 아니다.

## 출력

- `experiment.json`: 모든 조건·상태·checkpoint·원본 핵심 파일 SHA256.
- `maps.png`, profile별PNG, `layouts.json`: 장애물 지도와 모든 shelf의 적재 최단 경로.
- 각 run의 `metrics.jsonl`, `spatial.jsonl`, `communication.jsonl`: 기존 성능/활동/전이 로그.
- `constraints.jsonl`: rollout마다 wall/dock block, busy dock-step, 인근 loaded agent-step, 병목 영역 체류량. Action mask가 불법 행동을 막으면 block 횟수는0일 수 있다.0은 제약이 작동하지 않는다는 뜻이 아니다.
- `initial_eval_metrics.json`, `final_eval_metrics.json`, `summary.json`, `final.pt`, 주기 checkpoint.
- `-SaveVideo` 시 실제 벽·agent·goal 대기 상태를 CPU RGB renderer로 저장한다. OpenGL 창을 열지 않는다.

기존 차트 도구도 사용할 수 있다.

```powershell
& "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe" `
  examples/plot_pgct_probe_charts.py `
  --experiment-dir "runs/hard_실제출력폴더" `
  --output-dir "output/hard_실제차트폴더"
```

## 검증 기록 — 2026-10-13

- 새 환경17 tests를 포함한 관련32 tests 통과.
- 50개 reset seed에서 벽 spawn·중복 위치가 없음을 확인했고, 모든 선반의 적재 경로 접근성을 검사했다.
- 네 profile 모두 실제 pickup→delivery→home return 경로 실행을 통과했다.
- 원래 환경과 control의 관측·reward·동시 이동·미지정 seed reset RNG가 동일했다.
- 네 profile×PGCT/IPPO 총8조건의 CPU64-step 학습·평가·final 저장 완료: `runs/hard_smoke_20261013_final`.
- Control64-step PGCT/IPPO의128 tensor가 기존 E2 smoke 결과와 bitwise 일치했다. 원래 핵심10개 파일 SHA256도 그대로였다.
- 동일 preview 배치의 적재 평균 최단 경로는 control13.370칸, detour14.444칸(약8.0% 증가), 병목17.074칸(약27.7% 증가), 최대23→30칸이다. Dock은 같은 경로에 서비스 대기 제약을 추가한다.
- 이것은 경로 제약·실행 검증이며, 새 환경의 학습 난이도나 PGCT 성능 이득을 입증한 결과가 아니다.

초기 시행에서 발견한 preview 정수 직렬화 및 기존 probe builder의 새 인자 호환 문제를 새 파일 안에서 수정했다. 이전 실패 폴더 v1/v2는 완료 결과에 포함하지 않는다.

## 초기 난이도 보정

기존 E2 PGCT final을 고정한5×500 평가에서 control은37.8 deliveries/episode, detour는5.4, 강한 bottleneck/dock은각2.2였다. 이는 새 환경으로의 초기 일반화 평가이며 재학습 후 성능이나 본질적인 학습 난이도를 입증하지 않는다. 변화가 커서 적재 경로 증가는 더 완만한 detour를 기본으로 추가하고 bottleneck/dock은 강한 선택지로 남겼다. 최종 수치는 `output/hard_rware_validation_20261013/frozen_difficulty_pilot_v2.json`에 저장한다.

Detour에서도 기존 정책의 즉시 성능은 크게 떨어졌다. 장애물 수와 구조적 경로 증가가 작아도 기존 정책이 우회에 취약할 수 있다. 따라서 여기서 완만하다는 말은 구조적 제약의 정도이며, 실제 학습 난이도가 조금만 높아졌다는 검증은 아니다.
