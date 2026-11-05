# PGCT Cross-team 연결과 RWARE 공간 확대: 구현·로그 분석 보고서

작성일: 2026-10-08. 기준 모델: `stage04_4ag_physical_home0.500/final.pt`.

## 1. 판단과 분석 범위

**현재 same-team gate 소멸은 물리적 접촉 부족만으로 설명되지 않는다. 같은 팀의 공통 probe 응답이 임계값을 벗어난 것이 직접적인 차단 조건이다. 공간 분업은 가능한 배경 원인이지만, 정책 유사도와 공간 점유 유사도를 구분해서 검증해야 한다. 맵 확대는 이 가설을 검증할 수 있는 환경 변경이며, 개선이 보장된 해결책은 아니다.**

이번 작업에서는 학습 코드·환경 코드·설정·체크포인트·기존 로그를 수정하지 않았다. 저장된 자료를 읽고, 별도로 생성한 환경을 reset하여 맵 구조를 확인했다. 최종 모델의 평가도 가중치를 고정한 추론으로 수행했다. 새 산출물은 이 보고서와 같은 폴더에만 저장했다. 이전 턴에서 요청받아 수행한 구버전 코드 정리는 이번 분석과 별개다.

분석 대상:

- [기존 curriculum manifest](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/runs/pgct09700_physical_mixing_20260929_155338_079437/curriculum.json)
- 위 폴더의 네 단계 `communication.jsonl`, `canonical_probes.pt`, `final_eval_metrics.json` 및 stage04 `final.pt`.
- [기존 gating 진단](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/gating_diagnosis/README.md), [공통 probe 정책 응답](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/gating_diagnosis/policy_responses.json).
- [이번 분석 수치와 맵 snapshot](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/evidence.json).

아래에서 **확인**은 실제 코드·저장 자료·이번 평가에서 확인한 내용, **가설/계획**은 아직 인과적으로 입증하거나 실행하지 않은 내용이다.

## 2. 현재 환경의 실제 구조

### 2.1 크기·shelf·배송 지점

**확인:** 환경 ID는 `rware-multiteam-tiny-4ag-2teams-v0`이다. 등록 기본값은 `shelf_rows=1`, `shelf_columns=3`, `column_height=8`이며, manifest에 크기 override가 없어 이 기본값을 사용한다. 실제 환경 생성으로도 확인했다.

관련 구현:

- [환경 등록](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/__init__.py:44)
- [Warehouse._make_layout_from_params](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/warehouse.py:294)
- [Warehouse.reset](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/warehouse.py:775)

| 항목 | 현재 값 |
|---|---|
| 격자 높이 H × 너비 W | **11 × 10**, 110 cells |
| 크기 공식 | `H=(column_height+1)*shelf_rows+2`, `W=3*shelf_columns+1` |
| 초기 shelf 수 | **32**; 저장 cell마다 한 개 생성 |
| 저장 cell 밀도 | 32/110 = **29.09%** |
| 수직 통로 | `x % 3 == 0`; 일반 통로 폭 1 cell |
| 수평 통로 | `y % 9 == 0`, 마지막 배송 행 및 중앙 queue corridor |
| 팀 0 배송 지점 | **(4,10)** |
| 팀 1 배송 지점 | **(5,10)** |
| 관측/통신 범위 | sensor range 5 / communication range 5 |
| 에피소드 길이 | 최대 500 환경 step |

좌표는 **(x,y), 0부터 시작**, 격자 크기는 **(H,W)**다. 중앙 queue corridor가 가장 아래 shelf band의 중앙 shelf 두 열을 통로로 바꾸므로, 단순히 `3×2×8=48 shelves`로 계산하면 틀린다. 실제 tiny에는 32개가 있다. 두 배송 지점은 하단 중앙에 나란히 있으므로 서로 다른 지역에서 작업하더라도 배송 때는 중앙으로 모일 수 있다.

### 2.2 팀·혼합·보상

**확인:** [MultiTeamWarehouse._balanced_assignments](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/multi_team_warehouse.py:31)의 `agent_index % 2`로 팀을 정한다. 화면 번호 1–3은 팀 0, 2–4는 팀 1이다. 에피소드 reset 때 팀을 섞는 옵션은 기본 false이며 현재 override가 없다.

[MultiTeamWarehouse._assign_shelf_teams](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/multi_team_warehouse.py:349)는 `balanced_soft_zones`에서 좌우 저장 cell 수가 같아야 한다. 각 구역에서 `round(zone_shelf_count*(1-home_ratio))`개의 **shelf 팀 소속**을 반대 팀으로 바꾼다. 좌표나 통로는 바뀌지 않는다. 현재 두 구역은 각각 16개이며 home ratio 1/.875/.75/.5에서 구역당 소속 교환 수는 0/2/4/8개다. stage04는 좌우 각 구역에 팀별 shelf 8개씩 있다. 배치는 reset마다 RNG로 다시 선택한다.

`goal_team_mode=zones`는 원래 중앙 배송 지점 두 개를 팀별로 나눈다. `require_matching_team_goal=True` 기본값이 적용된다. [팀 배송 지점 배정](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/multi_team_warehouse.py:498).

현재 `team_reward_mode=individual`이다. 따라서 “같은 팀”은 shelf·목표 지점을 공유하는 작업 집단이지만, delivery/return 보상이 모든 팀원에게 동일하게 배분되는 설정은 아니다. 이 설정에서도 역할 분화는 가능하나, 팀 보상 최적화가 원인이라고 단정하면 안 된다. [배송 보상 구현](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/multi_team_warehouse.py:612).

현재 핵심 보상은 delivery 1.0, home return 1.0, requested pickup .15, shelf/goal/return progress .002/.005/.02이다. 잘못된 pickup·drop, 충돌·blocking·deadlock 및 NOOP 등의 벌점도 있다. [기본 보상 설정](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/env_kwargs_stage2_pgct_success_unblock_zones_individual.json). 맵 비교에서는 checkpoint의 실제 `effective_env_kwargs`를 복사하여 이 설정을 유지해야 한다.

### 2.3 에이전트 초기 위치

**확인:** 초기 위치는 고정 좌우 배치가 아니다. `Warehouse.reset`은 **전체 110 cells에서 서로 다른 4개 cell을 무작위 선택**한다. shelf 밑 cell이나 배송 지점도 포함된다. 방향도 무작위다. 팀 소속에 따른 시작 구역 제한은 없다.

이번에 현재 stage04 설정으로 `reset(seed=1)`했을 때 위치/방향은 다음과 같았다. 이 표는 재현 가능한 예시이며 기존 평가 영상의 시작 상태나 학습된 담당 구역을 뜻하지 않는다.

| Agent | 팀 | 위치 | 방향 |
|---|---:|---|---|
| 1 | 0 | (2,8) | RIGHT |
| 2 | 1 | (5,5) | UP |
| 3 | 0 | (4,10) | DOWN |
| 4 | 1 | (0,5) | RIGHT |

## 3. Cross-team 연결: 관측과 직접적인 gate 조건

### 3.1 알고리즘이 실제로 측정하는 것

[sample_policy_probe_sequences](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:2009)는 `objective`, `shared`, `fixed_canonical_probes=True`에서 별도 RNG seed `cfg.seed+9173`으로 외부 probe bank를 만들고 환경 인스턴스에 캐시한다. 현재 bank shape은 **[48,1,1593]**다. “fixed”는 한 단계 안에서 고정된다는 뜻이며 네 단계의 저장 bank hash는 서로 다르다.

[policy_distance_similarity_matrices](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:2090)는 모든 actor에 동일한 probe를 주고 **zero hidden state**, sequence length 1에서 **action mask 없는 softmax**를 비교한다. 거리 D는 probe별 JS divergence의 평균이며 자연로그를 사용한다. 실제 rollout의 위치별 행동 빈도나 평균 probability vector 사이의 JS와는 다르다.

[TeamGraphEstimator.update_from_policy_distance](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:554)는 probe round 시점에 물리적 이웃인 쌍의 EMA만 갱신한다. 현재 probe interval 5 updates, EMA beta .35, minimum count 8 rounds다. 48 probe를 한 번 비교해도 count는 **1** 증가한다. 이웃이 아니면 EMA가 오래된 상태로 유지된다.

[TeamGraphEstimator.pgct_gate_matrix](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:718)의 실제 threshold 비교는 **`D_ema < 0.12`**다. equality에서는 닫힌다. update 100부터 warmup이 끝나고, 측정 횟수 ≥8, finite distance, information 조건을 추가로 확인한다. 현재 minimum policy information은 0이다. 이후 [train의 current-neighbor 마스크](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:4925)가 적용된다.

정리하면 현재 승인 연결은 `physical neighbor AND warmup ready AND probe evidence ready AND D_ema<.12`다. true team label을 이용해 gate를 열거나 막지 않는다. 팀 label은 oracle 비교 지표를 위한 분석에 쓰인다.

PGCT transfer는 [update_ippo](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:2889)에서 donor critic의 고정된 공통 probe 예측을 receiver critic이 따르도록 하는 보조 MSE 손실이다. peer loss coefficient .01을 사용한다. actor 입력에 다른 agent의 실시간 메시지를 주는 구조가 아니다. 따라서 평가 영상의 초록 선과 평가 중 실시간 critic 학습 효과를 동일시하면 안 된다.

### 3.2 물리적 후보와 승인 결과를 분리한 기존 로그 집계

다음 수치는 **1024 환경 step rollout의 끝에서 기록한 update snapshot**을 집계했다. 전 환경 step의 접촉 빈도는 아니다. 무방향 upper triangle을 사용하며 same-team 가능한 쌍은 2개, cross-team은 4개다. 후보/승인 수는 `(쌍, update)` 발생 수라 한 update에 여러 쌍이 포함될 수 있다.

| 단계 | Updates | 평균 이웃/agent | Same 후보 → 승인 | Same 승인/후보 | Cross 후보 → 승인 | Cross 승인/후보 |
|---|---:|---:|---:|---:|---:|---:|
| 01, home1.0 | 977 | 1.644 | 1,499 → 1,131 | 75.45% | 1,713 → 0 | 0% |
| 02, home.875 | 1,954 | 1.761 | 2,906 → 380 | 13.08% | 3,977 → 0 | 0% |
| 03, home.75 | 2,930 | 1.810 | 3,741 → 0 | 0% | 6,868 → 443 | 6.45% |
| 04, home.5 | 3,907 | 1.805 | **4,364 → 0** | **0%** | **9,743 → 1,221** | **12.53%** |

warmup·측정횟수·information 조건까지 준비된 물리 후보를 분모로 삼으면 stage04의 cross 승인율은 1,221/9,500 = **12.85%**다. 같은 팀은 준비된 후보 4,238회에서도 승인 0회다. 모든 기존 snapshot에서 저장된 승인 행렬을 실제 gate 조건으로 재구성했고 일치함을 확인했다.

stage04에서는 1–3이 후보였던 update 2,157회, 2–4는 2,207회다. 따라서 **현재 같은 팀 연결 소멸의 직접 원인은 단순한 접촉 기회 소멸이 아니다**. 실제 cross 승인 쌍은 3–4 1,170회와 1–4 51회다. 승인된 쌍의 팀 구성은 cross 100%지만, 모든 가능한 cross pair가 자주 승인된다는 뜻은 아니다. stage04의 가능한 cross pair 전체 대비 승인율은 7.81%다.

최종 같은 팀 EMA는 1–3 .32416, 2–4 .41194이며 모든 쌍의 최종 EMA가 .12 이상이다. **학습 중 cross transfer가 있었다는 사실과 최종 gate가 모두 닫혔다는 사실은 함께 성립한다.**

### 3.3 공통 probe의 최종 정책 응답

기존 forward-only 분석은 네 단계 모델을 같은 stage04 bank에 평가했다. 최종 모델의 거리와 preferred action 불일치는 아래와 같다. EMA와 다른 값인 이유는 즉시 계산된 final policy response이며 neighbor 샘플링 EMA가 아니기 때문이다.

| 쌍 | 팀 관계 | 평균 JS | Argmax 불일치 |
|---|---|---:|---:|
| 1–3 | same | .33852 | 50.00% |
| 2–4 | same | .39423 | 58.33% |
| 1–2 | cross | **.58408** | 85.42% |
| 1–4 | cross | .20617 | 35.42% |
| 2–3 | cross | .43928 | 66.67% |
| 3–4 | cross | **.12999** | 18.75% |

**특정 cross pair가 same pair보다 유사하다는 것은 확인된다. 모든 cross pair의 유사도가 높아진 것은 아니다.** 문서의 “1과 2가 같은 왼쪽 담당” 예시는 가설의 설명용 예시이며, 실제로 1–2는 현재 가장 비슷하지 않은 쌍이다. 실제 연결 쌍 3–4, 1–4의 경로·작업 상태를 우선 조사해야 한다.

최종 기존 20-episode 평가의 평균 delivery는 47.25, home return은 45.95다. 기존 stage04 초반/후반 5회 평가 평균 delivery 42.17/45.87은 추세를 보여주지만 평가 seed가 다르다. PGCT 성능 이득이나 분업의 인과 증거로 사용할 수 없다. `eval_full_cycle_success_rate=1`도 모든 요청의 완주율이 아니라 **delivery와 return이 각각 한 번 이상 있었던 episode 비율**이다.

### 3.4 이번에 추가 확인한 공간 점유·작업 분화

**확인:** 기존 `T.evaluate`를 그대로 사용하되 실행 중 메모리에만 관측 wrapper를 붙여 stage04 최종 모델을 평가했다. 최종 평가와 같은 seeds **5,000,001–5,000,020**, deterministic actions, action mask on, episode당500 step을 사용했다. 학습을 하지 않았고 모델 state tensor hash가 평가 전후 동일함을 확인했다. 평균 delivery **47.25**, home return **45.95**를 재현했으며 저장된 최종 평가 전체 수치와 최대 차이는 **4.68e-9**였다.

총10,000 환경 step에서 매 행동 직전 상태를 기록했다. 왼쪽은 `x<5`, 오른쪽은 `x>=5`다. 모든 agent는 매 step 하나의 cell을 점유하므로 agent별 분모는10,000이다.

| Agent | 팀 | 왼쪽 체류 | Episode별 왼쪽 체류 min–max | 요청 shelf pickup의 왼쪽 비중 | 운반/return 상태 비중 |
|---|---:|---:|---|---|---:|
| 1 | 0 | **99.57%** | 98.2–100% | **202/202 = 100%** | 47.27% |
| 2 | 1 | **0.55%** | 0–1.6% | **0/174 = 0%** | 52.48% |
| 3 | 0 | **38.01%** | 22.0–51.6% | **57/298 = 19.13%** | 76.57% |
| 4 | 1 | **56.53%** | 39.8–72.4% | **215/299 = 71.91%** | 75.95% |

운반/return 비중은 shelf를 들고 있던 pre-action 상태의 비율이다. 요청 pickup은 `requested=True AND return_phase=False`인 event만 세었다. 전체 pickup 집계에는 다른 event도 포함되어 이 표의 분모와 다를 수 있다.

![Stage04 최종 고정 모델의 에이전트별 공간 점유](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/frozen_occupancy.png)

이 평가에서는 **Agent1은 팀0 배송 지점 쪽의 왼쪽 작업, Agent2는 팀1 배송 지점 쪽의 오른쪽 작업에 강하게 집중**했다. 같은 팀의 Agent3은 요청 pickup 대부분을 반대쪽 오른쪽에서, Agent4는 상당 부분을 반대쪽 왼쪽에서 수행하면서 양쪽을 이용했다. 따라서 문서의 “1·2 왼쪽 / 3·4 오른쪽” 예시보다 **팀별 자기 구역에 집중하는 agent와 반대 구역까지 이동하는 agent가 공존하는 형태**가 이번 최종 평가를 더 잘 설명한다. 정확한 역할 명칭은 여전히 해석이다.

전체 cell 점유 histogram 사이의 JS(nats)는 다음과 같다. 이는 **policy JS와 다른 변수**다. episode별로 계산한 JS의 평균도 함께 제시해 pooling으로 인한 과도한 유사도 해석을 피했다.

| 쌍 | 공간 점유 JS: 전체 pooled | 공간 점유 JS: episode 평균 |
|---|---:|---:|
| 1–3 same | .36493 | .43577 |
| 2–4 same | .33451 | .40478 |
| 1–2 cross | .66985 | .68405 |
| 1–4 cross | .22077 | .32578 |
| 2–3 cross | .24633 | .33882 |
| 3–4 cross | **.15283** | **.30894** |

3–4는 이 평가에서 공간 점유도 가장 유사하고, 공통 probe의 정책 응답도 가장 유사하다. 1–4도 same pair보다 공간 점유가 유사했다. 이는 **공간·작업 분화가 정책 유사도 역전과 연결될 수 있다는 H1의 관측상 근거**다. 그러나 2–3은 공간 점유가 비교적 유사해도 정책 JS는 .43928로 크다. 공간 점유 유사성만으로 policy compatibility를 설명할 수 없다. 또한 3과4가 pickup하는 구역은 주로 서로 반대이며, 단순한 “동일한 좌우 구역 담당” 가설보다 이동형 작업 상태나 경로 구조의 유사성까지 검토해야 한다.

이 결과는 학습 seed1의 **최종 모델 한 개**를 평가한 것이고, cross gate는 최종 상태에서 모두 닫혀 있다. 과거 3–4 gate가 열렸던 시점의 경로를 측정한 것은 아니다. gate가 열렸던 update 주변 저장 checkpoint를 동일 평가 seed로 비교해야 시간적 연관성을 강화할 수 있다. 인과관계와 다른 학습 seed에서의 재현성은 아직 확인하지 않았다.

다른 설명도 남아 있다. 초기09700은 home ratio1.0에서 사전학습했으므로 1·2의 구역 집중이 그때 형성된 습관을 유지한 것일 수 있다. 3·4의 확대된 활동 범위가 커리큘럼에서 새로 형성됐는지는 이전 checkpoint 경로와 비교해야 한다. 관측된 분화가 효율적인 최적 분업인지, 일부 agent의 고정 습관·탐색 제한인지는 별도 성능 비교가 필요하다. probe의 좌표/방향/목표 단서와 중복 가중, zero hidden state에서의 정책 sharpness도 policy JS를 설명하는 대안이다.

이번 평가의 per-step 평균 geometric 이웃 수는 **1.86855/agent**였다. 이것은 §3.2의 rollout-end 평균1.80535와 샘플링 시점·정책 시점이 다르다. 최종 모델에서도 물리 접촉 자체가 사라진 것은 아니라는 추가 확인이다.

자료: [Frozen 평가 요약](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/frozen_rollout_summary.json), [step별 위치·행동·phase·event 기록](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/frozen_rollout.jsonl). 새로운 평가 trajectory이며 기존 학습 trajectory 복원물이 아니다.

## 4. Probe의 공간적 편향과 평가 방법

### 4.1 확인된 공간 정보와 coverage

[Semantic observation](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/semantic_observation.py:25)은 ego 20개와 13×11×11 spatial channels, 총 1,593차원이다. 입력에 정규화된 절대 x/y, 방향, task phase, home displacement, 통로·경계·shelf·agent·goal·relevant goal이 들어간다. 팀 ID는 직접 입력하지 않지만 자기 팀 요청과 관련 배송 지점이 관측에 반영된다. 따라서 정책 거리를 순수한 이동 패턴만의 유사도라고 정의하는 것도 정확하지 않다.

[RwareObjectiveProbeMapBuilder._make_probe_env](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:269)는 실제 환경의 highways/goals를 layout으로 복제한다. 맵을 키우면 geometry가 반영된다. 그러나 [구체 probe 생성](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:654)의 dual 시나리오는 `_dual_state`를 공통으로 호출하며 일부 좌표가 고정되어 있다.

확인할 위치:

- [_dual_state](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:623): 기본 x=4를 sensor range와 격자 폭으로 clip, row 2 또는 7 등 사용. `goal`/`lane` 이름의 row7은 확장 맵에서 실제 goal 행과 멀어진다.
- [_scenario_shelf](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:477): variant에 따라 shelf 선택. 현재 48 slots는 48-slot cycle 한 번이므로 variant는 모두 0이다.
- [_place_background_agents](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:531): focal에서 먼 통로부터 background agent 배치. 실제 좁은 곳에서 여러 agent가 상호작용하는 상황을 충분히 대표한다고 볼 수 없다.
- [_place_shelf](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:566): 겹치는 shelf를 구석으로 옮기고 좌표를 clip한다. 작은 맵에서는 의도한 offset이 잘릴 수 있다.

**현재 저장 bank는 48 slots지만 고유한 입력은 26개다.** 중복 입력은 평균 JS에서 여러 번 가중된다. 이름이 다른 시나리오라도 현재 realized input이 같을 수 있다. 이번 환경 생성에서 E2/E3 bank 고유 입력 수도 25/24개였다. 따라서 공간 확대만으로 probe가 넓은 맵을 고르게 샘플링한다고 볼 수 없다. 이 사실은 측정 범위의 한계이며 이번 작업에서 probe 구현을 바꾸지는 않았다.

### 4.2 권장 로그·시각화 — 계획

| 목적 | 수집 항목 | 분석/시각화 |
|---|---|---|
| 공간 역할 | episode, step, agent, team, x/y, 방향, carrying/return phase | agent별 점유 heatmap, 이동 경로, 좌우/상하 체류 비율, episode별 역할 안정성 |
| 작업 역할 | shelf ID/home, 요청 팀, pickup/delivery/return event, carrier | pickup 출발 구역→배송→return 흐름, agent별 작업 단계 시간, 이동거리·cycle latency |
| Probe 응답 | bank hash/입력 중복 ID, scenario/variant, 각 action probability, entropy, per-probe JS | scenario×pair heatmap, 좌표/방향별 응답, 중복 입력 동일 가중 재집계 |
| 접촉과 승인 | geometric adjacency, evidence-ready, raw distance/EMA/count, gate, allocation | 후보·준비된 후보·승인 곡선, pair별 acceptance, EMA 나이와 접촉 누락률 |
| 전이 유익성 | peer loss, critic 예측/TD error, matched on/off 성능 | 같은 초기 상태에서 분기한 성능·학습 속도 비교 |

지금 `communication.jsonl`은 position/trajectory나 action probability를 저장하지 않는다. 기존 `collect_rollout`/`evaluate`에는 위치와 event를 읽는 함수가 있지만 결과는 대부분 집계다. **과거 학습 시점의 전체 경로를 최종 checkpoint와 집계 로그만으로 복원할 수 없다.** 이번처럼 고정 모델에 외부 분석 wrapper를 붙여 새 평가 경로를 얻는 것은 가능하다.

공간 정보의 영향량은 forward-only matched probe로 평가할 수 있다. 동일한 task phase·국소 배치에서 좌표/방향을 바꾼 realized state의 policy change를 측정하고, 목표/요청 단서만 바꾸는 실험과 비교한다. 방향을 돌릴 때 LEFT/RIGHT/FORWARD 행동 의미도 함께 맞춰야 한다. 좌표 두 숫자만 0으로 지우는 ablation은 불가능한 입력 조합을 만들 수 있으므로 보조 sensitivity 분석으로만 쓴다. 학습에 쓰는 gate bank는 그대로 유지하고 진단용 bank를 별도로 둔다.

서로 다른 맵에서의 비교에는 **각 맵에서 실현한 probe**와 **같은 저장 reference bank**를 병행한다. 전자는 환경 대응을, 후자는 입력 고정 상태의 정책 변화 자체를 본다. 후자는 확장 맵에서 out-of-distribution일 수 있다. 단계마다 bank가 달랐던 기존 EMA를 동일 조건의 정책 변화로 직접 비교해서는 안 된다.

## 5. 맵 1.5배·2배 확대의 구현 가능성과 구조 유지

### 5.1 기존 코드로 만들 수 있는 후보

아래는 **설정을 메모리에서 override하여 환경 생성/reset만 검증한 후보**다. 학습용 설정 파일로 적용하지 않았다. 모든 후보에서 agent4, request4(팀당2), sensor/communication range5, home.5, 보상은 동일했다. balanced zone과 1,593차원 관측 및 probe 생성도 확인했다.

| 후보 | rows / columns / height | H×W | H배 / W배 | 면적배 | Shelves / 밀도 | 팀별 goal 좌표 |
|---|---|---|---|---:|---|---|
| E1 기준 | 1 / 3 / 8 | 11×10 | 1 / 1 | 1 | 32 / 29.09% | (4,10), (5,10) |
| E2 길이 근사 | 2 / 5 / 6 | **16×16** | **1.45 / 1.60** | 2.33 | 108 / **42.19%** | (7,15), (8,15) |
| E3 길이 근사 | 2 / 7 / 8 | **20×22** | **1.82 / 2.20** | 4.00 | 208 / **47.27%** | (10,19), (11,19) |
| E2 모듈 유지 대안 | 2 / 5 / 8 | 20×16 | 1.82 / 1.60 | 2.91 | 144 / 45.00% | (7,19), (8,19) |

![생성기로 확인한 맵 구조와 seed1 초기 위치](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/map_geometry.png)

그림의 원은 seed1 초기 agent, 네모는 배송 지점이다. 담당 구역이나 평가 성능을 나타내지 않는다. 각 panel은 독립 좌표축이므로 그림의 화면 크기로 배율을 읽지 않는다.

현재 생성기는 shelf columns가 **홀수**여야 하며 W는 10/16/22…처럼 증가한다. column height8을 고정하면 H는 11/20/29…다. 따라서 두 변을 정확히 1.5배인 약17×15, 또는 2배인22×20으로 만드는 것은 기본 parameter 조합만으로 어렵다. E2에서 길이를 맞추면 shelf block 높이8→6도 바뀐다. 이 제약을 숨기고 `small/medium`을 각각 1.5/2배라고 부르면 안 된다.

### 5.2 어떤 파일/설정이 대상인가

| 위치 | 현 상태 | 향후 확대 시 예상 작업 |
|---|---|---|
| `rware/warehouse.py::_make_layout_from_params/_make_layout_from_str` | 정수 parameter 또는 `layout` 지원 | 기존 generator 사용이면 함수 변경 불필요. 엄격한 밀도/비율 제어는 `layout` 문자열로 가능 |
| [공통 학습기 make_env](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:1146) | Gym kwargs override 지원 | `--env-kwargs-json`에 rows/columns/height 또는 layout 포함 |
| [현재 from-scratch 실행기](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/run_semantic_cnn_marl_pgct_from_scratch.ps1:1) | EnvKwargsJson 전달 가능 | 크기 키는 전달 가능. 고정 LR2e-4/기타 옵션은 기존 physical curriculum과 다르므로 matched 비교에 주의 |
| [physical curriculum build_stages](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_pgct_physical_mixing_curriculum.py:23) | checkpoint geometry 상속; PGCT만 하드코딩 | 향후 map override와 PGCT-off 분기를 명시하는 별도 실행 설정 필요. 현 wrapper에 map 크기/off 옵션 없음 |
| [eval build_env_kwargs](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/eval_checkpoint_generalization.py:40) | tiny/small/medium 지원, height8 고정 | rows2/columns7/height6 같은 임의 후보에는 옵션 확장 또는 독립 평가 설정 필요 |
| [Probe builder](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/utils/probe_maps.py:239) | geometry 복제 + 일부 fixed coordinates | 확대 후 coverage/중복/goal-relative 배치 확인. gate 알고리즘 수정과 분리하여 취급 |

특히 [기존 map variants wrapper](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/run_pgct_09700_map_variants.ps1:1)는 기본 agent counts가 **6,8**, home ratio **1.0**, 기존 **09700** checkpoint의 frozen zero-shot 평가다. 문서의 **4-agent stage04 home.5 맵 효과 비교**와 조건이 다르다.

### 5.3 유지 가능한 조건과 함께 바뀌는 조건

기본 generator를 쓰면 일반 통로 폭 1, storage block 폭2, 하단 중앙 배송 지점2개와 그 인접 관계를 유지할 수 있다. column height8까지 유지하면 local block 규격도 유지된다. 그러나 **전역 shelf 밀도는 유지되지 않는다.** 작은 tiny에서는 중앙 queue 통로가 전체 저장 공간의 큰 몫을 제거하기 때문이다. 이는 agent가 다닐 수 있는 unloaded/loaded 경로와 병목 구조에도 영향을 준다.

엄격한 “공간 크기만” 비교를 원하면 후속 작업에서 custom layout을 설계해야 한다. 목표 규격은 다음과 같다.

1. 동일한 통로 폭·연결성, 팀별 배송 지점 개수와 하단 중앙 상대 위치를 유지한다.
2. storage cell 비율을 약29.09%에 맞추고 좌우 storage 수를 동일하게 만든다. 예컨대 16×16은 74개, 20×22는 128개가 이 비율에 가까운 짝수 shelf 수다. 이 수만으로 shelf module·통로 분포까지 같아지는 것은 아니다.
3. 일반 통로와 새 빈 storage 구역의 분포, 중앙 배송 queue 폭/길이, shelf-to-goal 경로 길이 분포를 함께 검증한다.
4. `balanced_soft_zones`의 동일 zone 크기, 요청 가능 shelf 수, 배송/return 도달성을 확인한다. layout이 제공되면 parameter 크기는 무시된다.

이 설계는 아직 생성·학습하지 않았다. 권장 해석은 “기본 generator로 넓은 창고를 만든 pilot”과 “밀도를 맞춘 통제 비교”를 구별하는 것이다. 어느 안에서도 shelf 수 증가는 **동시 요청 수 증가와 별개**다.

맵 변경 시 관측 tensor 차원은 sensor range를 유지하면 그대로라 checkpoint 가중치 로딩이 가능하다. 다만 normalized x/y 및 home dx/dy, 경계가 보이는 빈도는 바뀐다. 같은 tensor shape이 동일한 입력 분포를 보장하지 않는다. 현재 `normalize_shaping_rewards`는 default false이므로 progress reward 계수와 cell당 거리 변화는 유지된다. 이를 true로 바꾸면 H+W 기반 scaling까지 달라지므로 첫 비교에서는 바꾸지 않는다. 큰 맵의 긴 cycle, 더 많은 이동·충돌 벌점, gamma=.99의 할인도 성능 차이에 포함된다.

## 6. 물리적 접촉과 Gating 승인 기록

**확인:** [get_neighbor_adjacency](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/multi_team_warehouse.py:886)는 team label·방향·shelf 가림과 무관하게 `max(|dx|,|dy|) <= 5`를 사용한다. 물리 통신은 실제 메시지 bandwidth나 latency 모델이 아니라 학습 관계 추정/전이 대상의 공간적 자격 조건이다.

현재 [communication.jsonl 기록](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:5057)에는 `available`, `transfer_weights`, `directed_probe_counts`, `policy_distance_ema`, `policy_information`, gate requirements가 있다. 따라서 기존 자료로 update 시점의 후보와 승인, same/cross 분모를 구분할 수 있다. 그러나 IPPO `none`은 communication 자체를 끄므로 `available=0`이며 geometric 후보가 아니라 **기능 off 상태**다. IPPO와 PGCT 접촉을 비교할 때는 알고리즘 off와 별개로 좌표에서 geometric adjacency를 계산해야 한다.

권장 정의(무방향 쌍 기준):

- `A_ij(t)`: 현재 geometric neighbor.
- `R_ij(u)`: warmup·finite distance·count·information 준비 완료.
- `G_ij(u)`: 준비 조건, threshold 및 현재 physical mask를 통과한 활성 전이 쌍.
- physical neighbor count: `sum_j A_ij`; candidate edge count: `sum_(i<j) A_ij`.
- accepted gate count: `sum_(i<j) G_ij`.
- acceptance: `sum G / sum A`; evidence-ready acceptance: `sum G / sum(A*R)`.
- same/cross 가능한 쌍 대비 활성률: `sum G_same/(2T)`, `sum G_cross/(4T)`.
- 승인 구성비: `sum G_same/sum G`, `sum G_cross/sum G`. 승인0이면 미정/NA로 기록한다.

gate가 가중치형으로 바뀌면 count와 weight/allocation mass를 별도로 기록한다. 기존 `neighbor_arcs`는 방향 있는 링크 수라 upper-triangle pair 수와 2배 차이 날 수 있다. 그래프가 비대칭이면 두 방향을 별도로 유지해야 한다. 이번 현재 physical threshold 자료는 대칭임을 확인했다.

특히 probe는 5 update마다, 전이 자격 확인은 매 update, 위치는 매 환경 step마다 바뀐다. rollout 중 이웃이었다가 마지막에 떨어진 쌍은 해당 update 전이에서 제외된다. per-step candidate/accepted 표시와 실제 optimizer transfer count를 구분해야 한다. EMA 마지막 갱신 update와 age를 함께 기록한다. stage04의 1–3은 측정 이후 최대79 update 동안 EMA가 유지된 구간이 있었다.

초기 위치가 전체 cell에서 균등·중복 없이 선택되는 현재 reset 규칙에 대해서는 range5 접촉 확률을 정확히 계산할 수 있다. `S(n)=n+2*sum_(d=1..min(5,n-1))(n-d)`일 때 특정 두 agent의 접촉 확률은 `(S(W)S(H)-WH)/(WH(WH-1))`다.

| 맵 | Reset 순간 pair 접촉 확률 | 기대 이웃/agent |
|---|---:|---:|
| 11×10 | 59.80% | 1.794 |
| 16×16 | 32.26% | .968 |
| 20×22 | 20.63% | .619 |
| 20×16 모듈 유지 | 26.86% | .806 |

이는 **reset 직후 분포의 계산**, 학습된 agent의 rollout 접촉 예측은 아니다. 배송 지점에 모이면 평가 접촉은 더 높을 수 있다. 그래도 agent/range를 고정한 확대가 통신 기회를 줄일 수 있다는 H3의 구조적 근거는 된다. 실제 판별 개선 여부는 acceptance의 분모와 측정 freshness를 함께 봐야 한다.

## 7. 요청 수 증가·지연 페널티의 가능성

### 7.1 6–8개 요청

**확인:** 환경은 이미 `request_queue_size`와 `request_queue_size_per_team`을 지원한다. `_make_team_request_queues`가 팀별 shelf에서 중복 없이 뽑고 실제 전역 큐는 팀별 큐를 합친다. 후속 Q6은 total6/per-team3, Q8은 total8/per-team4를 모두 지정하면 된다. 현재 shelf 수로도 가능하다. 후보 shelf가 부족하면 팀별 큐 크기는 `min`으로 줄어든다.

단, 현재 from-scratch PowerShell 실행기, physical `build_stages`, generalization `build_env_kwargs`는 이 값을 **agent_count와 agent_count/2로 다시 덮어쓴다.** 기존 JSON의 Q만6/8로 바꾸고 해당 wrapper를 실행하면 의도한 실험이 되지 않는다. 향후 공통 trainer에 직접 명시한 kwargs를 전달하거나 wrapper의 workload 옵션을 추가해야 한다. [공통 trainer의 큐 override](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/train_recurrent_ippo_consensus.py:941)는 명시한 큰 팀별 큐를 auto-scale에서 낮추지는 않는다.

중요한 lifecycle: 현재 `require_delivered_shelf_return=True`이므로 **배송만 끝났다고 새 요청으로 바뀌지 않는다.** 배송된 shelf는 `shelves_awaiting_return`에 들어가 기존 queue slot을 차지한다. home return 완료 때 [_replace_team_request](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/rware/multi_team_warehouse.py:592)가 새 요청을 뽑는다. 따라서 Q4는 순수한 “아직 배송 안 된 주문 4개”가 아니라 배송·return을 포함하는 cycle slots4개다. Q6/8 효과를 해석할 때 unserved delivery 수와 awaiting return 수를 분리한다.

### 7.2 기다린 시간에 비례하는 벌점

**확인:** 요청별 생성 시간/age와 `-lambda*sum age`를 적용하는 설정은 현재 없다. NOOP·정체·step 벌점은 요청 대기시간 벌점과 다르다. 이번 작업에서 추가하지 않았다.

향후 필요 위치는 `MultiTeamWarehouse.__init__`의 coefficient/state, `reset`의 초기 요청 timestamp, `_replace_team_request`의 새 요청 timestamp 및 `step`의 age update/reward, `_get_info`의 latency 통계다. reward/task/shaping 분류와 checkpoint 실험 설정도 함께 기록해야 한다.

설계 전에 다음을 정해야 한다.

1. 배송 완료까지만 age를 계산할지, return 완료까지 포함할지. 문서의 **미배송 화물** 정의라면 awaiting-return shelf는 age 벌점에서 제외하고 새 요청 때 다시 시작한다.
2. 총 penalty를 team members에게 분배할지, individual carrier에게 줄지. 아직 pickup되지 않은 요청에는 carrier가 없으므로 현재 individual reward에 자동으로 배분할 수 없다.
3. sum/mean/team-normalization 중 무엇을 쓸지. Q를 늘리면 같은 평균 대기시간에서도 sum penalty가 커지므로 workload 효과와 reward 규모 효과가 섞인다.
4. 장기 대기 요청의 penalty 폭증을 어떻게 평가할지. 단일 주문을 T step 기다리면 누적 age penalty가 대략 `-lambda*T*(T+1)/2`로 커진다. 먼저 age·p95 delay·미완료 요청 age를 reward 변화 없이 측정한다.

맵 비교 Q4를 먼저 끝낸 뒤 고정한 큰 맵에서 Q4/Q6/Q8을 비교하고, delay penalty는 그 다음 별도 축으로 비교하는 것이 효과 분리에 적절하다.

## 8. 공정한 PGCT–IPPO 최소 실험 계획 — 아직 실행하지 않음

### 8.1 공간 원인 조사와 학습 이득 검증을 분리

고정된 stage04 모델을 각 맵에 zero-shot 평가하는 것은 경로 변화·contact sparsity·입력 분포 변화 조사다. **동일 모델에 평가 그래프만 off로 바꿔도 PGCT 학습 이득 ablation이 되지 않는다.** `T.evaluate`는 학습을 하지 않고 통신 gate는 표시/환경 그래프에만 적용한다. 현재 actor에 메시지 입력도 없다.

기존 `eval_checkpoint_generalization.py::checkpoint_pgct_gate`는 saved graph를 복제하여 사용하고 새 맵에서 policy graph를 재추정하지 않는다. 최종 stage04는 gate가 모두0이므로 확대 맵에서 선이 생기지 않는 zero-shot 영상으로 새로운 gating 판별 능력을 평가할 수 없다.

### 8.2 최소 학습 비교

**공간 효과와 추가 학습의 PGCT 이득을 함께 비교할 최소안:** 맵 E1/E2/E3 × PGCT-on/IPPO-off × paired 학습 seeds3개 = **18 runs**. 처음 1 seed의 6 runs는 pilot이며 논문 결론용 반복 검증과 구분한다. 기본 generator 후보를 쓰면 구조·밀도가 함께 바뀌는 pilot이라고 명시하고, 밀도 통제 layout을 확정한 후 본 비교를 수행한다.

현재 stage04와 연결되는 비교라면 두 branch 모두 같은 stage03 `final.pt`의 actor+critic을 읽고, home ratio .5를 고정해 **추가 4,000,000 steps** 학습하는 안이 적절하다. rollout 단위 때문에 실제 budget은 각각 **4,000,768**이다. 현재 stage04 `final.pt`에서 두 branch를 시작하는 것은 이미 gate가 닫힌 모델을 추가 적응시키는 다른 질문이다. 어느 경우든 같은 초기 checkpoint를 써야 한다.

| 통제 항목 | 양쪽 공통 설정 |
|---|---|
| 모델 초기화 | 같은 source checkpoint, `transfer_components=all`; optimizer/graph 모두 새로 시작 |
| Agent/작업 | agent4, teams2, Q4/per-team2, home.5, shuffle false |
| 관측·정책 | semantic CNN, 20 ego + 13×11×11, hidden128, action mask on |
| PPO | LR1e-4, entropy .02, gamma .99, GAE .95, clip .2, rollout1024, sequence64, epochs4, chunks8 |
| 보상/종료 | 같은 effective kwargs, 500-step episodes; delay penalty 추가 없음 |
| 평가 | 같은 고정 seed 목록, 20 episodes×500, 같은 deterministic/action-mask 규칙; 영상 별도 |
| 단계/예산 | 같은 checkpoint 시점과 각 branch env steps; update/초 단위 성능도 기록 |

PGCT-on은 현재 physical/probe48/interval5/min-count8/EMA.35/threshold.12/warmup100/peer loss.01을 유지한다. IPPO-off는 `--no-communication`, `graph_mode=none`, `comm_graph_mode=none`, `peer_transfer_mode=none`, consensus0/peer loss0이다. off에서 probe가 비활성화되더라도 **진단용 geometric contact와 frozen/reference probe 분석은 외부에서 동일하게 측정**한다.

같은 PGCT stage03에서 분기한 IPPO-off는 **PGCT로 사전학습한 공통 모델에서 추가 transfer의 효과**를 보는 ablation이다. 전체 학습 과정에서 pure IPPO와 PGCT를 비교했다는 주장을 할 수 없다. 전체 이득은 같은 초기 random weights·같은 pretraining/curriculum budgets로 두 방법을 처음부터 별도 학습하여 비교한다.

기존 physical 커리큘럼을 정확히 재현하는 비교는 같은 초기09700, 각 단계1/2/3/4M, LR1e-4/entropy.02, 같은 reset 규칙을 양쪽에 적용해야 한다. [일반 mixing wrapper](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/examples/run_semantic_cnn_mixing_curriculum.ps1:1)는 scratch 시작·단계당 동일5M·LR2e-4/entropy.05이므로 현재 physical run의 대조군으로 그대로 사용할 수 없다. 별도 IPPO pretraining 8–9M도 방법 전체 비교의 재료는 되지만 matched curriculum 비교를 대신하지 않는다.

주요 성능은 delivery/return throughput, task return, 고정 eval seed에서의 학습곡선 면적·목표 성능 도달 env steps, training seed별 변동이다. success rate만 쓰면 현재처럼1.0에 포화될 수 있다. 넓은 맵에서 horizon을 임의로 늘리면 다른 조건을 바꾸는 것이므로 주 비교는500으로 고정하고, 긴 horizon 결과는 별도 보조 평가로 제시한다. 동일 seed를 써도 두 정책의 경로가 달라 RNG 소비가 달라질 수 있으므로 모든 요청 sequence가 완전히 같다고 가정하지 않는다.

### 8.3 각 가설의 판별 기준

| 가설 | 지지할 수 있는 관측 | 결론의 한계 |
|---|---|---|
| H1 공간 역할 | 실제 연결 쌍의 occupancy/path/task-state가 유사하고 matched spatial probe 응답도 유사; 연결 직전 checkpoint에서도 재현 | 최종 heatmap이나 policy JS 하나만으로 학습 중 공간 역할이 gate 원인이라고 확정 불가 |
| H2 맵 확대 | 밀도/작업량을 통제한 맵 사이에서 역할 패턴·reference response·same/cross acceptance가 재현성 있게 변화 | cross count 감소만으로 판별 개선 주장 불가 |
| H3 접촉 희소성 | geometric 후보/agent, 접촉 episode coverage 감소; probe count 부족·EMA age 증가 | reset 계산은 rollout 결과를 대신하지 않음 |
| H4 작업량 | 고정 맵에서 Q4→6→8로 작업 선택/경로/latency가 변화 | 맵·Q·보상을 동시에 바꾸면 원인 분리 불가 |

Cross-team transfer의 유익성은 on/off branch의 critic quality와 후속 task performance로 판단한다. 선이 사라지거나 same-team 선이 늘었다는 결과만으로 PGCT 성능이 좋아졌다고 판단하지 않는다.

## 9. 권장 순서와 산출물

1. 현재 저장된 pair·probe 결과와 최종 frozen rollout을 비교하고, 공간 역할 주장을 데이터에 맞춰 좁힌다.
2. probe 중복과 fixed-coordinate coverage를 별도 진단으로 확인한다. 학습 gate 알고리즘은 유지한다.
3. 확대 맵의 실제 규격·밀도·통로·goal 배치를 확정한다. 기본 generator pilot과 밀도 통제 비교를 구별한다.
4. geometric contact, evidence-ready candidate, accepted transfer, EMA age를 분리하는 분석 기록을 준비한다.
5. paired E1/E2/E3 × on/off 실험을 수행하고, 그 후 Q6/8과 요청 대기 벌점을 검토한다.

산출물은 [evidence.json](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/evidence.json), [맵 구조 그림](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_spatial_analysis/map_geometry.png), §3.4의 frozen 평가 요약·trajectory·점유 그림이다. JSONL 재집계에서 승인 조건을 재구성해 저장된 gate와 일치함을 확인했고, 후보 맵의 balance/관측/probe 생성과 frozen 모델 가중치 보존·기존 평가 수치 재현을 확인했다. 모든 학습 계획은 아직 실행하지 않았다.
