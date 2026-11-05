# 2026-10-13 학습 결과 분석

**현재 seed 1에서는 E1의 평가 학습곡선 평균에 PGCT 이득 신호가 있지만, 최종 성능 우위나 critic 정확도 개선은 확인되지 않았다. E2는 PGCT와 IPPO가 거의 같다. 공간 확대 후 같은 팀 gate가 다시 생겼으나, 그것이 성능 향상으로 이어진다는 증거는 아직 없다.**

로그 스냅샷: 2026-10-13 14:37 KST. E3는 실행 중이므로 이후 수치는 달라진다. 학습 코드·설정·gate·probe·기존 로그·체크포인트를 변경하거나 학습을 중단하지 않았다. 이 폴더에만 분석 산출물을 추가했다.

## 비교 조건과 진행 상태

실험: `runs/sp_20261008_120638_983201/experiment.json`.

모든 분기는 동일한 Stage03 final actor/critic에서 출발한다. 각 분기의 optimizer와 graph 상태는 새로 시작한다. Home ratio 0.5, 4 agents, 요청 4개, sensor/communication range 5, 학습 seed 1, learning rate 1e-4, entropy coefficient 0.02, rollout 1024, PPO epochs 4, peer coefficient 0.01(PGCT)/0(IPPO)다. 맵별 on/off의 환경 차이는 communication 활성/비활성 및 topology뿐이며, PPO·보상·평가 설정은 동일했다.

기존 PGCT 사전학습을 공유하므로 이 비교는 **Stage03 이후 추가 전이의 효과**를 본다. 처음부터 PGCT를 전혀 사용하지 않은 IPPO와 전체 curriculum의 효과를 비교한 것은 아니다.

| 조건 | 저장된 학습 step | 마지막 평가 delivery | 마지막 평가 shelf return | 상태 |
|---|---:|---:|---:|---|
| E1 11×10 PGCT | 4,000,768 | 44.35 | 42.65 | 완료 |
| E1 11×10 IPPO | 4,000,768 | 44.65 | 43.70 | 완료 |
| E2 16×16 PGCT | 4,000,768 | 37.85 | 36.15 | 완료 |
| E2 16×16 IPPO | 4,000,768 | 38.20 | 36.85 | 완료 |
| E3 20×22 PGCT | 2,252,800 | 27.25 | 25.95 | 진행 중; 평가는 2,201,600 step |
| E3 20×22 IPPO | — | — | — | 시작 전 |

평가는 공통 reset seeds 5,000,001–5,000,020, deterministic masked actor, 에피소드당 500 step이다. Delivery/return은 에피소드 평균이다. 개별 agent의 원본 `*_count`는 20개 에피소드 합계이므로 반드시 20으로 나눴다. 64-step smoke test, dry-run, 계획 manifest는 본 실험 반복 수에 포함하지 않았다. 실제 완료된 seeds 2·3, random donor, drop34의 장기 학습 결과는 발견되지 않았다.

## 학습곡선 해석

| 지표 | E1 PGCT | E1 IPPO | E2 PGCT | E2 IPPO |
|---|---:|---:|---:|---:|
| 평가 delivery 시간 평균 | 45.856 | 44.573 | 34.167 | 34.245 |
| 마지막 10회 평가 delivery 평균 | 46.170 | 44.515 | 37.470 | 37.605 |
| 최고 평가 delivery | 47.95 | 46.80 | 38.15 | 38.65 |
| 최종 평가 delivery | 44.35 | 44.65 | 37.85 | 38.20 |

시간 평균은 관측된 51,200–4,000,768 step 평가 곡선을 사다리꼴 적분하고 구간 길이로 나눈 정규화 AUC다. 학습 중 실제 누적 delivery가 아니다. 기록되지 않은 step 0 점수는 만들지 않았다. 최고 성능은 고른 체크포인트의 성능이므로 최종 모델과 구분한다.

E1 시간 평균은 PGCT가 1.283 delivery/episode, 상대적으로 2.88% 높다. 79개 평가 시점 중 PGCT가 높은 시점은 60개, IPPO가 높은 시점은 16개, 동률은 3개다. 이 시점들은 한 학습 경로의 상관된 평가이므로 79개 독립 seed로 취급할 수 없다. 마지막 약 100만 step 구간의 시간 평균도 PGCT 46.511, IPPO 44.364였으나, 최종 snapshot은 IPPO가 0.30 높다. 따라서 최종 점수 하나만으로 PGCT가 계속 열세였다고 해석해서도 안 된다.

다만 PGCT가 모든 학습 효율 지표에서 앞선 것도 아니다. 처음으로 평가 delivery 45를 넘긴 시점은 PGCT 460,800, IPPO 358,400 step이다. 마지막 10회 점수의 시간 방향 표준편차는 PGCT 1.118, IPPO 0.712이며, 이는 seed 간 variance나 평가 confidence interval이 아니다. 이 run으로 PGCT가 안정성도 개선했다고 주장할 근거는 없다.

E2 시간 평균 차이는 PGCT−IPPO = −0.078(−0.23%)다. 첫 100만 step에서 약 18.8→32 수준으로 빠르게 적응하고 이후 약 37–38에 도달하는 양상은 두 조건에 공통이다. 현재 seed에서는 추가 critic transfer의 뚜렷한 성능 이득을 확인하지 못했다.

E3 PGCT는 첫 기록된 평가 10.00에서 27.25로 향상했다. IPPO 대조군이 없어 이 향상을 PGCT의 효과로 귀속할 수 없다. E1/E2/E3의 이동 거리, shelf 배치·밀도가 함께 다르므로 맵 간 delivery 차이를 알고리즘 우열로 해석하지 않는다.

## 전이 및 공간 활동

아래 횟수는 무방향 pair-update다. 한 쌍을 양방향 arc 2개로 중복 집계하지 않았다. 물리 후보는 update 종료 시점에 실제 범위 안에 있던 쌍이다.

| Pair | E1 PGCT 승인 | E2 PGCT 승인 | E3 PGCT 승인(진행 중) |
|---|---:|---:|---:|
| 1–3 same | 0 | 89 | 410 |
| 2–4 same | 0 | 781 | 348 |
| 1–2 cross | 0 | 932 | 306 |
| 1–4 cross | 0 | 1,276 | 441 |
| 2–3 cross | 0 | 39 | 72 |
| 3–4 cross | 2,005 | 113 | 102 |

E1 same-team 물리 후보는 4,435회였지만 승인 0회다. 3–4는 물리 후보 2,525회 중 2,005회(79.4%) 승인됐다. 약 103,424 step부터 전이가 기록됐고, 100만 step 이후에도 지속됐다. 이번 E1은 **초기에만 전이하고 후반에는 완전히 꺼진 run이 아니다.** 기존 curriculum Stage04의 3–4 1,170회와 이번 E1의 2,005회는 별개의 학습 결과다.

E2는 same-team 후보 3,176회 중 870회(27.4%), cross-team 후보 6,434회 중 2,360회(36.7%) 승인됐다. Same-team 연결이 회복됐지만 가장 빈번한 쌍은 cross-team 1–4다. E3 역시 same/cross 연결이 공존한다. 따라서 맵 확대가 팀 식별 문제를 해결했다고 볼 수 없다.

학습 전 구간 평균 물리 이웃 수는 E1 PGCT 1.815, E2 PGCT 1.243, E3 PGCT 0.788이다(E3는 미완료). 접촉은 줄었지만 E2의 총 승인 pair-update는 E1보다 많았다. 연결 변화는 접촉 감소 하나만으로 설명되지 않는다. 모든 분석 대상에서 물리 범위 밖 전이는 0회였고, IPPO의 전이도 0회였다.

E1 학습 전체에서 왼쪽 체류 비율은 PGCT [99.0%, 0.6%, 37.7%, 61.8%], IPPO [98.0%, 0.6%, 47.7%, 62.0%]다. Agent 1/2의 좌우 편중과 Agent 3/4의 상대적으로 넓은 활동은 양쪽 조건에서 나타난다. 이 공간 분화를 PGCT만이 만들어냈다고 할 수 없다. 이 값은 과거 20회 frozen evaluation의 occupancy와 다른 **학습 전체 rollout** 통계다.

## 에이전트별 성과

| 조건 | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---|---:|---:|---:|---:|
| E1 PGCT 최종 delivery/episode | 8.70 | 8.50 | 13.65 | 13.50 |
| E1 IPPO 최종 delivery/episode | 9.30 | 8.10 | 13.35 | 13.90 |
| E2 PGCT 최종 delivery/episode | 9.60 | 9.35 | 9.00 | 9.90 |
| E2 IPPO 최종 delivery/episode | 9.40 | 9.60 | 9.75 | 9.45 |
| E3 PGCT 마지막 delivery/episode | 7.00 | 6.60 | 6.70 | 6.95 |

네 agent 모두 작업을 수행하고 있다. E1은 3/4가 전체 delivery의 약 61%를 담당하지만 E2/E3는 더 균등하다. 기록된 모든 update에서 개별 agent의 `actor_update_skipped`는 0이었다. 따라서 이번 학습은 한 agent만 업데이트하고 나머지를 나중에 순차 학습하는 구조로 보이지 않는다.

E1 마지막 10회 평균 delivery는 PGCT [9.400, 8.830, 13.950, 13.990], IPPO [9.135, 8.350, 13.210, 13.820]다. PGCT 평균 이득이 3–4에게만 나타난 것도 아니다. 최종 snapshot만 보면 Agent 3은 PGCT가 0.30 높고 Agent 4는 0.40 낮다. 3–4 둘 다 성능이 향상됐다고 단정할 수 없다.

## 저장 모델의 추가 critic 평가

이번 분석에서 완료된 네 최종 모델을 고정하여 각각 10×500 step을 추가 평가했다. 별도 seeds 5,100,001–5,100,010, stochastic masked actor, 독립적인 local NumPy action RNG를 사용했다. 모든 40개 에피소드는 time-limit truncation이었다. 평가 전후 state-dict hash가 같음을 확인했다.

MC target은 실제 reward(작업 보상+shaping+penalty)의 유한 500-step discounted return이며 horizon 끝의 tail은 0이다. 학습 critic은 계속되는 task의 value를 근사하므로, 끝부분 MC MSE에는 tail bias가 있다. 이를 줄여 보기 위해 남은 step이 300 이상인 초기 200 step MSE도 별도 보고한다. 이 보정도 실제 미래 tail을 복원하거나 bias를 완전히 제거하지는 않는다. TD는 truncation에서 bootstrap하고 자연 종료에서 0으로 처리했다.

| 최종 모델 | 전체 MC MSE | 초기 200-step MC MSE | 평균 absolute TD error |
|---|---:|---:|---:|
| E1 PGCT | 4.190 | 1.725 | 0.06486 |
| E1 IPPO | 3.713 | 1.103 | 0.06076 |
| E2 PGCT | 2.190 | 0.286 | 0.04522 |
| E2 IPPO | 2.146 | 0.281 | 0.04408 |

E1 Agent 3의 초기 MC MSE는 PGCT 1.547/IPPO 0.407, Agent 4는 0.938/0.453이다. 현재 최종 모델에서 이 두 critic의 정확도가 개선됐다는 증거는 없다. E2는 거의 같다. 다만 모델마다 자기 정책으로 방문한 상태에서 계산한 값이고, reward target의 분산도 다르다. 따라서 이 비교만으로 PGCT 전이가 critic을 손상시켰다고 단정할 수 없다. 학습 중 value loss 또한 독립 평가 정확도와 동일하지 않다.

과거 공통 rollout 진단에서 3–4 critic 예측 간 MSE가 작았다는 사실과 모순되지 않는다. **서로 비슷한 예측과 실제 return에 대한 정확도는 다른 지표다.** 또한 과거 transfer가 예측을 비슷하게 만들었을 가능성이 있으므로 예측 일치를 donor 선택의 사전 타당성으로 소급하지 않는다.

## 최종 probe 민감도

완료된 PGCT/IPPO 모델을 각 맵의 동일한 저장 canonical bank에 입력했다. Zero hidden, unmasked, forward-only 평가다. E1은 48개/고유 26개, E2는 48개/고유 25개 입력이다. 중복 제거는 진단에만 사용했고 실제 gate는 변경하지 않았다.

E1 PGCT에서 3–4 JS는 48개 평균 0.0921, 고유 입력 평균 0.1421이다. 1–4는 0.2114→0.1232다. **최종 모델에서도 probe 가중 방식에 따라 3–4/1–4의 유사성 순위가 바뀐다.** 기존 임계값을 재적용해 고유 입력 gate가 더 좋다고 주장하지 않는다. 과거 EMA의 연결 기록과 최종 raw JS도 구분해야 한다.

E1 IPPO의 3–4 JS는 0.2029/0.2659로 PGCT보다 높았다. 이는 추가 PGCT 여부와 최종 actor 응답 차이가 동반된 관측이며, 그 자체가 정확도·전이 유익성의 증명은 아니다.

E2에서 1–4는 PGCT 0.0074/0.0102, IPPO 0.0330/0.0326으로 두 기준 모두 가장 유사했다. 팀 구분과 행동 응답 유사성이 분리되는 현상은 IPPO에서도 나타난다. 맵별 bank 자체가 다르므로 E1/E2 raw JS를 동일 상태 분포의 거리처럼 직접 비교하지 않는다.

## 과거 curriculum과의 관계

기존 stage01→04 최종 delivery는 61.00→50.60→49.85→47.25였고, same-team 승인 pair-update는 1,131→380→0→0이었다. 마지막 두 단계는 cross-team 전이가 나타났다. Home ratio가 단계마다 바뀌고 당시 평가 seed도 체크포인트마다 달라졌으므로 이 숫자를 성능 퇴행이나 PGCT 인과 효과로 해석할 수 없다.

별도 home1 IPPO continuation은 최종 delivery 61.45, shelf return 59.50을 저장했다. 학습 예산과 초기 모델이 현재 Stage03→home0.5 비교와 다르므로 현재 E1 PGCT의 대조군으로 섞지 않는다.

## 연구 판단과 다음 순서

1. E1에서 추가 PGCT가 학습곡선 평균을 높인다는 탐색적 신호는 있다. 최종 성능·critic 정확도·안정성까지 함께 향상됐다는 주장은 아직 불가하다.
2. E2에서는 gate가 더 많이 열리고 same-team도 연결됐지만 on/off 성능은 거의 같았다. Gate 활성량이나 same-team 비율을 성공 지표로 대체하지 않는다.
3. 이미 진행 중인 E3를 보존하고 IPPO 대조군 완료를 기다린다. 현재 분석 때문에 학습을 중단하거나 probe/threshold를 바꾸지 않았다.
4. 우선 E1 seeds 2·3을 동일 checkpoint에서 반복하여 2.9% 수준의 평균 이득이 재현되는지 확인한다. 이 역시 하나의 사전학습 모델에 조건부인 반복이라는 한계를 명시한다.
5. E1 random donor와 drop34는 JS 선택 및 3–4 연결의 기여를 구분하는 후속 실험이다. 전달량 감소·auxiliary loss 크기·gradient 크기 등의 차이를 따로 기록한다. 이번 분석에서 추가 장기 학습은 시작하지 않았다.

산출물: `analysis.json`(학습 로그 스냅샷), `critic_summary.json`(추가 평가 전체 수치), `probe_responses.json`, `learning.png`, `agents.png`, `gates.png`, `probes.png`, `critic.png`. 재현용 분석 스크립트는 이 폴더 안에 있으며 학습 실행기와 별개다.
