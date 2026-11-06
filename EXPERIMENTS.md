# PGCT 실험·평가 기록

최초 정리: 2026-10-13 Codex. 최신 정량 스냅샷은 14:37 KST. 원본 로그·체크포인트는 보존한다. 이 파일의 계획과 명령은 실행 사실을 뜻하지 않는다.

## E00. 기존 curriculum — 완료

경로: `runs/pgct09700_physical_mixing_20260929_155338_079437/`.

| Stage | Home ratio | 추가 환경 step | 최종 delivery | Shelf return | Same 전이 | Cross 전이 |
|---|---:|---:|---:|---:|---:|---:|
| stage01 | 1.000 | 1,000,448 | 61.00 | 58.95 | 1,131 | 0 |
| stage02 | .875 | 2,000,896 | 50.60 | 48.85 | 380 | 0 |
| stage03 | .750 | 3,000,320 | 49.85 | 47.95 | 0 | 443 |
| stage04 | .500 | 4,000,768 | 47.25 | 45.95 | 0 | 1,221 |

전이는 무방향 pair-update로 집계했다. Stage04 cross 내역은 3–4: 1,170회, 1–4: 51회다. Stage마다 혼합률과 모델, 당시 평가 seed가 달라 delivery 차이를 PGCT의 인과 효과로 해석하지 않는다.

Stage04의 same-team 물리 후보는 4,364회지만 승인 0회였고, cross 후보는 9,743회 중 1,221회 승인됐다. 최종 raw JS는 1–3 .338519, 2–4 .394231, 1–2 .584076, 1–4 .206173, 2–3 .439278, 3–4 .129985다. Gate는 이웃에서 갱신되는 EMA < .12를 사용하므로 최종 raw 값과 과거 승인을 구분한다.

근거: [Gate 진단](output/gating_diagnosis/summary.json), [공간 분석](output/pgct_spatial_analysis/REPORT.md).

## E01. 기존 home1 IPPO continuation — 완료, 대조 조건 미일치

Run: `runs/recurrent_ippo_consensus/ex5_semantic_cnn_none_from_scratch_4ag_sr5_square_home1.00_seed1_pretrain_off_9700_continue/`.

- Graph none, communication off, peer coefficient 0, consensus 0인 IPPO다.
- 최종 delivery 61.45, shelf return 59.50. 기존 checkpoint900의 가중치에서 이어졌으며 새 counters가 시작된다.
- 사전학습 예산·초기 상태가 Stage03 분기와 달라 E1 PGCT의 matched baseline으로 사용하지 않는다.
- 과거 영상의 agent별 성공 시점 차이를 순차 optimizer 업데이트로 해석하지 않는다.

근거: 해당 run의 `final_eval_metrics.json`, 기존 from-scratch 실행기와 과거 진단.

## E02. 공간 분기 — E1/E2 완료, E3 진행

경로: `runs/sp_20261008_120638_983201/experiment.json`.
초기 모델: E00의 `stage03_4ag_physical_home0.750/final.pt`.
실행기: `examples/run_pgct_spatial_experiments.ps1`, `examples/train_pgct_spatial_experiments.py`.

조건: 4 agents/2 teams, individual reward, home .5, Q4(팀당 2), sensor/range 5, LR1e-4, entropy .02, rollout1024, training seed1. 각 분기는 4M 요청 예산을 올림하여 4,000,768 step을 학습한다. 같은 초기 actor/critic에서 출발하고 optimizer/graph는 초기화한다. 따라서 Stage03 이후 추가 전이의 조건부 효과를 비교한다.

Task 평가는 deterministic masked policy, 공통 reset seeds 5,000,001–5,000,020, 20회×500 step이다.

| Run | 상태 | 저장 step | 최종/최근 delivery | Shelf return | 정규화 평가 delivery AUC | 마지막 10회 평균 |
|---|---|---:|---:|---:|---:|---:|
| e1_pgct_s1 | 완료 | 4,000,768 | 44.35 | 42.65 | 45.8559 | 46.170 |
| e1_ippo_s1 | 완료 | 4,000,768 | 44.65 | 43.70 | 44.5734 | 44.515 |
| e2_pgct_s1 | 완료 | 4,000,768 | 37.85 | 36.15 | 34.1674 | 37.470 |
| e2_ippo_s1 | 완료 | 4,000,768 | 38.20 | 36.85 | 34.2449 | 37.605 |
| e3_pgct_s1 | 진행 | 2,252,800 | 27.25 | 25.95 | 22.2601(부분) | 26.835 |
| e3_ippo_s1 | 시작 전 | — | — | — | — | — |

E3 최근 평가는 2,201,600 step이다. AUC는 기록된 51,200 step부터 최종/최근 평가까지 사다리꼴 적분하고 구간 길이로 나눈 값이다. 기록에 없는 step0 점수는 만들지 않았다. E3 부분 AUC를 완료된 4M AUC와 직접 비교하지 않는다.

| Pair | E1 PGCT 승인 | E2 PGCT 승인 | E3 PGCT 승인(부분) |
|---|---:|---:|---:|
| 1–3 same | 0 | 89 | 410 |
| 2–4 same | 0 | 781 | 348 |
| 1–2 cross | 0 | 932 | 306 |
| 1–4 cross | 0 | 1,276 | 441 |
| 2–3 cross | 0 | 39 | 72 |
| 3–4 cross | 2,005 | 113 | 102 |

Pair-update는 양방향을 중복 세지 않는다. E1 same 후보 4,435회/승인 0회, 3–4 후보 2,525회/승인 2,005회다. E2 same 후보 3,176회/승인 870회, cross 후보 6,434회/승인 2,360회다. 물리 범위 밖 전이는 0회, IPPO 전이도 0회다.

| 최종/최근 delivery/episode | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---|---:|---:|---:|---:|
| E1 PGCT | 8.70 | 8.50 | 13.65 | 13.50 |
| E1 IPPO | 9.30 | 8.10 | 13.35 | 13.90 |
| E2 PGCT | 9.60 | 9.35 | 9.00 | 9.90 |
| E2 IPPO | 9.40 | 9.60 | 9.75 | 9.45 |
| E3 PGCT | 7.00 | 6.60 | 6.70 | 6.95 |

원본 agent count는 20개 에피소드 합계이므로 20으로 나눴다. 해당 로그의 개별 actor update skip은 모두 0이었다.

근거: [전체 분석](output/learning_review_20261013/REPORT.md), [수치](output/learning_review_20261013/analysis.json), [학습곡선](output/learning_review_20261013/learning.png), [Agent별 성과](output/learning_review_20261013/agents.png), [전이 차트](output/learning_review_20261013/gates.png).

## E03. 기존 Stage04 공간 활동 — 고정 모델 평가 완료

기존 Stage04 final을 20×500 step 평가했고 가중치가 바뀌지 않았음을 확인했다. 왼쪽 체류율은 Agent1 99.57%, Agent2 .55%, Agent3 38.01%, Agent4 56.53%다. 3–4 occupancy JS는 .152835로 가장 낮았다.

다만 Agent3 요청 pickup의 약 81%는 오른쪽, Agent4의 약 72%는 왼쪽이었다. 동일 작업이나 역할을 수행했다고 결론 내리지 않는다.

근거: [Frozen 평가](output/pgct_spatial_analysis/frozen_rollout_summary.json), [분석](output/pgct_spatial_analysis/REPORT.md).

## E04. 기존 Stage04 probe 중복 민감도 — 진단 완료

동일 canonical 입력, zero hidden, unmasked, natural-log JS다. 48개 중 고유 입력은 26개다.

| Pair | 기존 48개 평균 | 고유 26개 평균 |
|---|---:|---:|
| 1–3 same | .338519 | .358212 |
| 2–4 same | .394231 | .404173 |
| 1–2 cross | .584076 | .492518 |
| 1–4 cross | .206173 | .137414 |
| 2–3 cross | .439278 | .494423 |
| 3–4 cross | .129985 | .226997 |

3–4의 두 actor 모두 FORWARD 확률이 .9 이상인 입력은 30/48개이며 그 평균 JS는 .007116, 나머지 18개의 평균은 .334768이다. 고유 입력에서는 각각 10/26개와 16/26개다. 정책 출력으로 사후 분류한 그룹이며 실제 작업 중요도를 나타내지 않는다.

중복 제거가 과거 EMA 연결 역전을 증명하지 않는다. 실제 학습 gate는 변경하지 않았다.

근거: [수치](output/pgct_transfer_review/probe_bias_check.json), [차트](output/pgct_transfer_review/probe_bias_check.png).

## E05. 공통 rollout 이력의 policy/critic 일치 — 진단 완료

모델은 기존 Stage04 final과 새 E1 PGCT checkpoint600(614,400 step)이다. 둘 다 PGCT이므로 on/off 비교가 아니다.

Reference는 기존 Stage04 stochastic policy, 20×500 step, seeds 5,000,001–5,000,020이다. 전체 이력으로 hidden을 전개하고, 10-step 간격×4개 관측 stream으로 4,000개 표본을 측정했다. 각 network가 동일 이력으로 자기 hidden을 형성하며 다른 정책에는 off-policy 이력일 수 있다.

| 기존 Stage04 pair | Recurrent masked JS | Critic 예측 간 MSE |
|---|---:|---:|
| 1–3 | .191996 | 13.976555 |
| 2–4 | .212130 | 11.427813 |
| 1–2 | .335800 | 7.109442 |
| 1–4 | .180121 | 14.928075 |
| 2–3 | .213066 | 10.376292 |
| 3–4 | .059547 | .106799 |

새 E1 checkpoint600의 3–4는 JS .068073, critic 예측 간 MSE .120176이다. 모델 hash 불변을 확인했다. 3–4 유사성이 canonical 중복에만 존재하지는 않지만, 예측 일치·정확도·전이 유익성은 다르다. 전체 6쌍에서 관계가 단조롭지도 않았다.

근거: [진단 수치](output/pgct_transfer_review/common_rollout_stage04_e1_u600/summary.json), [이력 cache](output/pgct_transfer_review/common_rollout_stage04_e1_u600/histories.npz), [차트](output/pgct_transfer_review/common_rollout_stage04_e1_u600/compatibility.png).
코드: `examples/diagnose_pgct_rollout_compatibility.py`.

## E06. 2026-10-13 최종 critic의 실제 return 예측 — 평가 완료

완료된 E1/E2 PGCT/IPPO final 각각 10회×500 step을 평가했다. Stochastic masked actor, reset/action RNG seeds 5,100,001–5,100,010이며 모델별 seed는 같아도 방문 상태는 달라질 수 있다. 모든 40개 에피소드는 time-limit truncation이었고 모델 hash는 불변이었다.

Target은 shaping/penalty를 포함한 실제 reward의 유한 discounted return, zero tail이다. TD는 truncation에서 bootstrap한다.

| 모델 | 전체 MC MSE | 초기 200-step MC MSE | Mean absolute TD |
|---|---:|---:|---:|
| E1 PGCT | 4.190251 | 1.724620 | .064855 |
| E1 IPPO | 3.713456 | 1.103477 | .060764 |
| E2 PGCT | 2.189594 | .286130 | .045222 |
| E2 IPPO | 2.146145 | .281301 | .044082 |

초기 200 step은 미래 300 step 이상이 남아 tail 편향을 줄이지만 없애지는 않는다. E1 Agent3의 초기 MC MSE는 PGCT1.546733/IPPO .407245, Agent4는 .937810/.453493이다.

현재 최종 critic 정확도 개선은 미확인이다. 서로 다른 방문 분포, target 분산, 단일 training seed 때문에 전이가 critic을 손상시켰다고 단정할 수도 없다.

근거: [전체 수치](output/learning_review_20261013/critic_summary.json), 같은 폴더의 개별 `*_critic.json`, [차트](output/learning_review_20261013/critic.png).
재현 코드: `output/learning_review_20261013/evaluate_critics.py`.

## E07. 2026-10-13 최종 canonical 응답 — 진단 완료

맵마다 해당 PGCT의 저장 bank를 두 조건에 공통 적용했다. E1은 48개/고유26개, E2는 48개/고유25개다. Zero hidden/unmasked 평가이며 모델 hash 불변을 확인했다.

| 모델·pair | JS48 | 고유 입력 JS |
|---|---:|---:|
| E1 PGCT 3–4 | .0921 | .1421 |
| E1 PGCT 1–4 | .2114 | .1232 |
| E1 IPPO 3–4 | .2029 | .2659 |
| E2 PGCT 1–4 | .0074 | .0102 |
| E2 IPPO 1–4 | .0330 | .0326 |

E1의 순위 민감성은 지속된다. E2의 1–4 유사성은 두 가중 방식과 두 학습 조건 모두에서 나타났다. 맵별 bank가 달라 raw JS의 맵 간 직접 비교에는 주의한다.

근거: [응답 수치](output/learning_review_20261013/probe_responses.json), [차트](output/learning_review_20261013/probes.png).

## E08. Transfer 전용 실행기 — 구현 완료, 장기 실험 미실행

- 실행기: `examples/run_pgct_transfer_experiments.ps1`, `examples/train_pgct_transfer_experiments.py`.
- Modes: pgct/ippo/random/drop34/drop34_refill. 동일 Stage03→E1, 각 4,000,768 step, 기본 seeds1·2·3.
- 과거 검증 기록: 관련 35 tests 통과, 조건별 64-step smoke. 경로는 `runs/tr_smoke_20261008_final`, `runs/sp_check_20261008_120401_732`다. PGCT/IPPO 64-step weights가 기존 공간 실행기와 bitwise 일치한 기록이 있다.
- 공통 이력 진단의 추가 2 tests 통과 기록도 있다. 이번 공용 노트 생성에서 테스트 전체를 재실행한 것은 아니다.
- `runs/tr_reuse_plan_20261008_final`은 계획이며 seeds2·3 완료가 아니다. Random/drop의 64-step 결과를 장기 효과로 해석하지 않는다.
- Ablation의 실제 전이는 `transfer.jsonl`을 사용한다. `communication.jsonl`과 영상은 원래 gate를 표시할 수 있다.

완료한 기존 E1 seed1을 재사용하고 seeds2·3만 새로 학습하는 실행 예시다. **이 기록 작성에서는 실행하지 않았다.**

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\examples\run_pgct_transfer_experiments.ps1 -Seeds 1,2,3 `
  -ReuseExperiment "runs/sp_20261008_120638_983201"
```

Random schedule, 부족량, 평가 지표, 모드별 제약은 [실행기 문서](examples/PGCT_TRANSFER_EXPERIMENTS.md)를 확인한다.

## E09. E2 연결 성분과 완전 연결 구분 — 2026-10-13 진단 완료

대상: 완료된 `e2_pgct_s1/communication.jsonl`의 전체 3,907 update. `transfer_weights`의 어느 한 방향이라도 양수인 쌍을 무방향 edge로 계산했다. 이 run은 양방향 support가 비대칭인 update가 0회였다. 분모에는 warmup과 연결 없는 update도 포함한다.

| 최대 연결 성분 크기 | Update 수 | 완전 연결 여부 |
|---|---:|---|
| 1 | 1,636 | 전이 edge 없음 |
| 2 | 1,592 | 두 개씩 분리된 경우는 이 중 13회 |
| 3 | 661 | 세 쌍 모두 연결 244회, 일부만 연결 417회 |
| 4 | 18 | 모두 일부만 연결; 네 agent의 여섯 쌍이 모두 열린 경우는 0회 |

예시: 1,266,688 step의 1·2·4는 세 쌍 모두 열렸다. 2,538,496 step에서는 1–2, 1–4, 2–3 세 edge가 네 agent를 연결했지만 1–3, 2–4, 3–4는 열리지 않았다. 모든 agent가 서로 유사하거나 실제 팀 구성이 바뀌었다는 뜻은 아니다.

이 분석은 probe가 이동 상태에 치우쳤는지나 전이가 유익했는지를 측정하지 않는다. 학습·원본 로그·모델은 변경하지 않았다.

근거: [연결 수치와 예시](output/learning_review_20261013/e2_components.json), 재현 코드 `output/learning_review_20261013/analyze_e2_components.py`.

## E10. 분리된 장애물·배송 제약 환경 — 2026-10-13 구현 및 시험 완료

새 환경ID는 `rware-constrained-4ag-2teams-v0`다. 기존ID 등록 파일과 원래trainer는 수정하지 않았다. 전용 실행기 `examples/run_hard_rware_experiments.ps1`의 기본은동일 Stage03→detour의PGCT/IPPO seed1 각4,000,768step이며, **해당 장기 학습은 아직 실행하지 않았다.**

| Profile | 제약 | 선반 수 | 평균 적재 최단 경로 | 고정 E2 모델 delivery/episode |
|---|---|---:|---:|---:|
| control | 원래 E2와 같은 동역학 | 108 | 13.370 | 37.8 |
| detour(기본) | (3,14)/(12,14)에 벽2개 | 108 | 14.444 | 5.4 |
| bottleneck | y=7의벽14개, x=3·12만개방 | 108 | 17.074 | 2.2 |
| dock | bottleneck + 배송후8-step적재진입대기 | 108 | 17.074 | 2.2 |

경로 수치는preview reset seed5,000,001의모든shelf→해당팀goal 최단거리다. Frozen평가는`runs/sp_20261008_120638_983201/e2_pgct_s1/final.pt`의deterministic masked policy, seeds5,000,001–5,000,005,5×500이다. 모델hash불변. 이표의성능은초기distribution shift이며재학습난이도·성능상한·PGCT이득을판단하지않는다.

검증: 새환경17개를포함한관련32 tests통과. 4 profiles×PGCT/IPPO8조건각64-step CPU학습·평가·final저장완료. 새control weights는기존E2 smoke의128 tensor와bitwise일치. 기존핵심10개파일SHA256보존. 모든shelf의적재가능경로, 50reset seed의spawn, 벽/마스크/로컬관측, dock종료시간, 실제배송/반납, RGB영상저장검증.

성공smoke: `runs/hard_smoke_20261013_final/experiment.json`.
근거: [검증 결과](output/hard_rware_validation_20261013/validation.json), [Frozen 평가](output/hard_rware_validation_20261013/frozen_difficulty_pilot_v2.json), [지도](runs/hard_smoke_20261013_final/maps.png), [실행 문서](examples/HARD_RWARE_EXPERIMENTS.md).

수동canonical probe와gate/threshold는유지했다. 기존probe constructor에새환경인자3개를전달하지않도록임시호환adapter만사용한다. 장애물전용probe는추가하지않았다. Shader/OpenGL없는별도RGBrenderer와1593차원호환semantic wrapper를사용한다.

원래W&B·학습을건드리지않도록전용runner는기본W&B/영상off이며필요시`-Track -SaveVideo`로켠다. 자세한제약·reward해석·실행명령은문서를참조한다. v1/v2실패폴더와v3의초기3-profile검증은최종장기학습결과로세지않는다.

## E02 추가 — E3 PGCT 중단 (2026-10-13 Claude Code 실측)

18:21 KST Windows 재부팅으로 `e3_pgct_s1` 프로세스가 종료됐다. 시스템 로그에는 정상 종료와 Kernel API에 의한 reboot가 기록됐다. 마지막 metrics 2,855,936 step, 마지막 checkpoint `checkpoint_02700.pt`(2,764,800 step), 마지막 평가 2,816,000 step delivery 28.6 / return 26.8, 관측 구간 AUC 23.49. `experiment.json`은 `running`/`planned`로 남아 있으며 수정하지 않았다. 기존 실행기는 같은 폴더로 이어 학습을 지원하지 않는다. 재개하지 않았다.

## E11. 실험 A — 국소 행동으로 팀 식별 가능성과 critic consensus 전제 — 2026-10-13 Claude Code, 완료

- 코드: `latent_team/exp_a_team_identifiability.py`. 출력: `output/latent_team/exp_a_20261013/`(`REPORT.md`, `summary.json`, `identifiability.png`, 모델별 npz).
- 모델: Stage03(E1 환경에서 평가), E1/E2 IPPO·PGCT seed1 final, Stage04 PGCT final. 각 모델 자체의 stochastic masked policy로 20회×500 step, reset seeds 5,200,001–5,200,020. 모든 모델 weight hash 불변. 학습·gate·probe 변경 없음.
- 측정: canonical JS(각 run의 48개 bank, zero hidden), 이력 정책 JS(두 agent 각자의 관측 이력, 각 network의 자기 hidden), 방문 분포 JS, 국소 목격 증거(관측자 센서 범위 안의 상대 pickup·delivery를 관측자 팀 기준으로 분류, Beta(1,1)), critic 예측 차이, blend gain(초기 200 step 유한 MC return에서 `MSE(V_i) - MSE((V_i+V_j)/2)`).

| 모델 | canonical JS AUROC / 팀원식별 | 이력 JS | 방문 JS | 국소 증거 | critic gap | blend gain |
|---|---|---|---|---|---|---|
| Stage03→E1 env | .500 / .250 | .282 / .013 | .379 / .062 | 1.000 / 1.000 | .242 / .050 | .325 / .150 |
| E1 IPPO | .375 / .000 | .267 / .000 | .408 / .113 | 1.000 / 1.000 | .001 / .000 | .251 / .113 |
| E1 PGCT | .375 / .000 | .267 / .000 | .333 / .013 | 1.000 / 1.000 | .000 / .000 | .290 / .125 |
| E2 IPPO | .375 / .250 | .810 / .675 | .867 / .825 | 1.000 / 1.000 | .452 / .250 | .493 / .263 |
| E2 PGCT | .500 / .250 | .853 / .850 | .821 / .725 | 1.000 / 1.000 | .554 / .450 | .470 / .225 |
| Stage04 PGCT | .500 / .250 | .291 / .000 | .365 / .075 | 1.000 / .994 | .013 / .000 | .289 / .113 |

AUROC는 같은 팀 쌍 점수가 높을수록 1이다. 팀원식별은 각 agent의 최고 점수 상대가 팀원인 비율이고, 우연 수준은 .333이다.

- 국소 증거는 모든 모델에서 첫 episode(500 step)만으로 6쌍 모두 증거를 얻어 팀원을 맞혔다. 20 episode 누적 동안 같은 팀 쌍의 반대 증거는 0–13회였다(최대는 Stage04 2–4 13회, 같은 쪽 증거는 수백 회). 다른 팀 쌍의 같은 팀 증거는 0회였다. Episode 단위 팀원식별은 Stage04만 .994였다.
- E1 계열 critic: 같은 팀 쌍의 예측 차이가 더 크다. E1 IPPO gap 1–3 12.19, 2–4 14.56, 3–4 .54. 초기 MC MSE는 Agent1·2(1.7–3.7)가 Agent3·4(.37–.69)보다 높아 agent별 숙련/역할 차이가 팀보다 지배적이다. E2는 모든 쌍의 gap .03–.06, blend gain 약 0이다.
- 한계: 단일 training seed 체크포인트, 종속 표본, 2 same/4 cross 쌍이다. 국소 증거는 env 이벤트와 위치로 관측자가 볼 수 있는 사건을 재구성한 것이며, 관측 텐서에서 학습된 추론기가 아니다. 이 환경은 팀을 요청·목표로 정의하므로 증거가 거의 정답 수준이다. 일반화는 미확인이다. Blend gain은 유한 horizon MC(zero tail)와 각 모델의 방문 분포에 의존한다. 결과 해석(관측이 팀 기준이고 individual reward에서 팀원이 경쟁)은 가설이다.

## E12. 실험 B 실행기 — oracle/all/wrong critic peer-loss 분기 — 2026-10-13 Claude Code, 구현·smoke만

- 코드: `latent_team/exp_b_oracle_consensus.py`, `latent_team/run_exp_b.ps1`, 설명 `latent_team/README.md`. 기존 trainer의 `graph_mode=oracle/unrestricted/wrong`을 그대로 사용한다. 기존 E1 PGCT seed1과 같은 Stage03·환경·PPO·예산 4,000,768 step·평가 seed를 쓰고, 그 run의 canonical bank를 미리 넣는다. 처음 100 update는 전이를 보류한다. `transfer.jsonl`에서 실제 allocation과 팀/물리 범위 위반 여부를 검사한다. 기존 PGCT/IPPO run은 읽기 전용으로 비교한다.
- 검증: dry-run에서 세 mode 모두 기존 E1 PGCT와 `graph_mode`만 다름을 확인했다. `runs/lt_b_smoke_20261013` 64-step CPU smoke에서 oracle 같은 팀 arc 6/다른 팀 0, all 6/12, wrong 0/12, 모두 물리 이웃 범위 안이었다. 세 run의 canonical bank가 기존 E1 PGCT bank와 동일했다. 비교표가 기존 E1 AUC 45.8559/44.5734를 재현했다. `tests/test_latent_team.py` 6 tests, 관련 기존 테스트 포함 19 passed.
- **장기 학습 미실행.** 예상 소요는 기존 E1 속도 기준 run당 약 30시간이다. Learned-team mode는 구현하지 않았다.

## E13. 국소 증거의 관측 가능성 검증 — 2026-10-13 Claude Code, 완료

- 코드: `latent_team/verify_evidence_observability.py`. 결과: `output/latent_team/exp_a_20261013/observability_check.json`.
- 조건: E1 IPPO, E2 IPPO, Stage04 PGCT final. 자체 stochastic masked policy로 5회×500 step, seeds 5,200,001–5,200,005. weight 불변.
- 결과: 실험 A가 센 (관측자, 대상, 사건)마다 관측자의 post-step semantic 관측에서 대상 칸을 확인했다. pickup 985건(362/193/430)과 delivery 1,161건(474/214/473) 모두 agent+loaded로 보였다. `requested_shelf`(pickup)·`relevant_goal`(delivery) 채널 값도 증거 라벨과 100% 일치했다.
- 한계: 관측의 agent 칸에는 ID가 없다. 사건을 특정 이웃에게 귀속하는 것은 이웃 식별 가정이다. 정지 상태로 운반 중인 agent와 막 pickup한 agent는 관측상 구분되지 않는다.

## E12 추가 — 짧은 matched-prefix 예산 지원 (2026-10-13 Claude Code)

트레이너에 학습률 schedule이 없음을 확인했다(`total_timesteps`는 update 수만 결정). 이에 따라 `--timesteps`가 기준 run보다 짧으면 `total_timesteps` 차이만 추가로 허용하고, 비교표는 기준 run을 같은 step에서 자르도록 했다. Dry-run에서 1,024,000 step oracle 계획을 확인했다. 기존 E1 첫 1,024,000 step의 delivery AUC는 IPPO 43.828, PGCT 44.605(평가 20회)다. 관련 test 1개를 추가해 20 passed. **학습은 실행하지 않았다.**

## E14. 처음부터 학습 실행기 — 2026-10-13 Claude Code, 구현·smoke만

- 코드: `latent_team/train_from_scratch.py`, `latent_team/run_from_scratch.ps1`(병렬은 job별 별도 프로세스, 스레드 4). `exp_b_oracle_consensus.matched_transfer`에 초기 가중치 hash 기록과 ippo/pgct 허용 조건을 추가했다.
- 기본 조건: E1 맵 11×10, 4 agents/2 teams, home .5, 요청 4, individual reward, 무작위 초기화, lr 2e-4, entropy .05(기존 Stage03 분기 실험은 1e-4/.02), rollout 1024, 10M step, 평가 51,200 step마다 20×500(seeds 5,000,001–), checkpoint 250 update마다.
- 검증:
  - `runs/scratch_smoke_20261013`: 5 modes 순차 64-step CPU.
  - `runs/scratch_smoke_parallel_20261013`: 4 modes 병렬. 첫 병렬 시도 2회는 상태 파일 읽기 경합으로 비교/상태 기록 오류가 났고, 원자적 교체와 재시도로 수정한 뒤 재실행해 전부 complete.
  - 모든 mode의 초기 가중치 hash가 동일했다(`baaa10bf…`, Stage03 `b5e2158b…`와 다름).
  - 전이 arc는 ippo 0/0, pgct 6/12, oracle 6/0, all 6/12, wrong 0/12(같은 팀/다른 팀)였다.
  - 전체 tests 179 passed(plan 검사 test 포함).
- **장기 학습 미실행.** 기존 E1 속도 기준 예상 소요는 run당 2M 약 15시간, 10M 약 75시간이며, 병렬 시 더 느려질 수 있다.

## E14 추가 — 2단계 자동 파이프라인 (2026-10-13 Claude Code)

사용자 요청("1번 끝나면 자동으로 2번 돌아가게")에 따라 다음을 추가했다.
- `latent_team/run_scratch_pipeline.ps1`: 1단계 pilot의 모든 job이 complete이면 2단계 본 비교를 시작한다. `-Detach`로 백그라운드 실행.
- `run_from_scratch.ps1`의 `-Wait` 옵션.

`-SmokeTest -Detach` 검증(`runs/scratch_pipeline_smoke_20261013_*`, 각 64-step): 19:23:37 시작, 1단계 2 jobs 완료 후 19:23:46 2단계 시작, 4 jobs 완료 후 19:23:55 종료. 장기 학습은 시작하지 않았다.

## E14 점검 — 실행 전 코드·설정 검토 (2026-10-13 Claude Code, 학습 미실행)

사용자 붙여넣기 점검 요청 4항목. 실제 크기 plan을 scratchpad에만 만들었고(plan-only, bank 생성), env 학습은 하지 않았다.
1. **Stage1 checkpoint 재사용 없음**:
   - 두 단계 모두 `init_checkpoint=""`이고, pipeline은 출력 폴더만 다르다.
   - Stage1과 Stage2의 같은 mode 설정 차이는 `total_timesteps`, `save_dir`, `wandb_group`뿐이며 probe bank는 동일하다.
   - 트레이너에 학습률 schedule이 없으므로 **Stage1 2M ippo/oracle은 Stage2 10M ippo/oracle의 앞 2M과 같은 학습(prefix)**이다. GPU 비결정성으로 bitwise 일치는 보장되지 않는다.
2. **같은 초기 파라미터**:
   - 실제 트레이너 smoke(CPU)에서 stage1/stage2 6개 run의 초기 actor+critic hash가 모두 같았다(`baaa10bf…`).
   - 트레이너는 네트워크를 CPU에서 만든 뒤 `.to(device)` 하므로(`train_recurrent_ippo_consensus.py` 4757–4766행) device=auto(CUDA)에서도 초기화는 CPU RNG를 따른다. 별도 재현 코드에서 cpu와 cuda hash가 같음을 확인했다.
   - 다만 그 재현 hash(`69ca…`)는 트레이너 hash와 달라 재현이 완전하지 않다. 실제 GPU run의 일치 여부는 `initial_weights_sha256.txt`/`comparison.md`로 확인해야 한다.
3. **성능 gate 없음**: 현재 pipeline은 Stage1 job이 오류 없이 끝났는지만 본다. 학습이 안 돼도 약 75시간짜리 Stage2가 시작된다. 위 1의 prefix 관계 때문에 Stage1은 Stage2와 중복 계산(약 15시간×2 프로세스)이다.
4. **GPU/CPU**:
   - GPU: RTX 5070 Ti 16,303 MiB(사용 중 780 MiB). agent당 1,268,550 파라미터. 4 agents + PPO 크기 minibatch(16×64 step) fwd/bwd의 peak 할당은 143 MiB이고, CUDA context는 프로세스당 수백 MiB로 추정한다. 4 프로세스 병렬도 메모리 여유가 크다.
   - CPU: Core Ultra 7 265K 20코어, RAM 31.3GB(여유 17.4GB). 4 job×4 thread=16. 기존 E1 37 step/s는 Python 환경 step과 평가가 지배하는 것으로 추정한다(GPU 사용률 측정은 미실시). 병렬 시 job별 속도는 실측이 필요하다.
   - Windows Update 활성 시간이 8–17시이고, E3를 중단시킨 18:21 재부팅은 활성 시간 밖이었다. 자동 업데이트 재시작일 가능성이 있다(가설).

## E15. W&B 통합 + MAPPO/SePS/HetDPS 구현 — 2026-10-13 Claude Code, smoke만

- **W&B**: 트레이너의 `init_wandb`/`log_to_wandb(step=env_steps)`를 재사용하고 runner에서 project·group·tags·config·gate 지표를 추가했다.
  - `runs/wbsmoke_20261013_*`(`-Detach -Track`)에서 project `rware-from-scratch-smoke`에 3 run이 finished로 기록됐다. Step은 [32, 64]이고 config에 algorithm/stage/seed가 들어갔다.
  - pipeline 로그를 UTF-8로 바꿨다.
- **학습 구간 근거**: `runs/recurrent_ippo_consensus/ex5_semantic_cnn_policy_complete_from_scratch_.../wandb/run-bhh1l6x8.wandb` history(9,766 rows)를 읽기 전용으로 파싱했다(수치는 D21).
  - 같은 폴더의 IPPO from-scratch 기록(tkz072ou/kfomo827)은 약 1M step뿐이다. pickup은 0.76M에 시작했고 delivery는 1M 안에 거의 없었다.
  - 이전 run과 현재 계획의 차이: home 1.0 vs .5, communication complete vs physical. 보상과 맵은 같다.
- **baseline smoke**(각 64-step CPU):
  - `runs/baseline_smoke_mappo_20261013`, `runs/baseline_smoke_seps_20261013`(sklearn 부재로 seps 실패 기록), `runs/baseline_smoke_seps2_20261013`, `runs/baseline_smoke_hetdps_20261013`
  - `runs/baseline_smoke_all_20261013`: ippo/mappo/seps/hetdps, W&B 포함, 4 run finished
- **추가 비용 측정**(GPU, 무작위 데이터, env 학습 아님): HetDPS 측정 1회 약 4.4초(15M step 기준 4,687회 약 5.8시간), SePS VAE 1회 약 9.1초.
- **테스트**: 전체 182 passed.
- **본 학습은 실행하지 않았다.** 상세 구현·차이: `latent_team/BASELINES.md`.

## E16. 본 학습 전 실제 설정 검증 — 2026-10-13 Claude Code, 검증만

- **실행**: `runs/verify_realcfg_20261013`. `run_from_scratch.ps1 -Modes ippo,pgct,oracle,all -TotalTimesteps 4096 -Parallel -ThreadsPerJob 4`, 실제 학습 설정(device=cuda, rollout 1024, seq 64, 평가 20×500은 종료 시 1회). W&B는 끄고 4 update만 실행했다. **본 학습 아님.**
- **초기 파라미터**: 4 run 모두 첫 update 전 actor+critic hash `f78ec390e0fad60c…`로 같다.
  - CPU smoke 설정의 hash(`baaa…`)와 다른 것은 설정 차이 때문이다. 같은 설정 안에서 mode 간 일치가 요구 조건이다.
  - 모든 run이 complete였고 오류 로그는 없다. PGCT/oracle 전이는 warm-up 100 update 때문에 이 4 update에서는 발생하지 않는다. 전이 경로는 smoke(E14)에서 검증했다.
- **속도**: 4 프로세스 병렬에서 update당 38.5–39.5초(job당 약 26 step/s, 합계 약 105 step/s). 15M이면 약 6.6일 이상이고 평가 시간이 추가된다.
- **용량**: checkpoint 61MB × 250 update 간격 → run당 약 3.6GB. 디스크 여유는 1.5TB다.
- **resume 검증**: 코드 검토 결과 불가능하다(D22).

## E17. From-scratch 핵심 비교 본 학습 — 2026-10-13 20:23:52 KST 시작(사용자 실행), 진행 중

- **경로**: `runs/scratch_20261013_202352_stage2`, 로그 `runs/scratch_20261013_202352_pipeline.log`.
- **조건**: IPPO/PGCT/Oracle/All, seed1, 15M, 병렬 4 프로세스(PID 10392/29856/12684/24828), W&B.
- **20:27 KST 무렵 확인**
  - 4 run 모두 2,048 step, update당 약 40.6초로 15M까지 약 7일 예상.
  - GPU: nvidia-smi 사용률 70–81%(사용자 작업관리자에서는 95–99%로 보임), 전력 78–83W/300W, 온도 50–52°C, 팬 0%, VRAM 2.6/16.3GB.
  - CPU 59%, 프로세스당 RAM 약 2.1GB, 여유 RAM 10GB.
- **해석**: 사용률 수치는 GPU가 작은 kernel을 실행 중인 시간 비율이다. 전력·온도로 보면 하드웨어 부하는 낮다. 6–7일 연속 실행의 주요 위험은 GPU 마모가 아니라 재부팅·절전·정전, 다른 GPU 작업과의 경합이다.

## E17 추가 — Windows Update 재부팅으로 중단 (2026-10-15 Claude Code 실측)

- 2026-10-15 00:59 KST에 MoUsoCoreWorker가, 01:01 KST에 TrustedInstaller가 "운영 체제: 업그레이드/서비스 팩" 사유로 재시작을 실행했다(System 이벤트 1074/109). 원인은 Windows Update다.
- 마지막 metrics: ippo 2,487,296, pgct 2,502,656, oracle 2,471,936, all 2,455,552 step(00:58–00:59 기록). 마지막 checkpoint는 4개 모두 `checkpoint_02250.pt`(2,304,000 step)이고 job_status는 `running`으로 남아 있다.
- 최근 평가 delivery(약 2.4–2.46M): IPPO 2.9, PGCT 2.2, Oracle 10.7, All 4.2. 단일 seed이고 중단된 초기 구간의 값이다.
- 폴더와 로그는 보존했다. 정확한 재개는 불가능하다(D22).

## E18. E17 중단 시점(약 2.5M) 학습 현황 분석 — 2026-10-15 Claude Code, 읽기 전용

- **코드·산출물**: `output/scratch_review_20261015/`(analyze.py, analysis.json, curves.png, REPORT.md, reference_home1_pgct_eval.json).
- **조건**: E17 seed1. 공통 평가 구간 51,200–2,406,400 step. deterministic 20×500 평가. 인접 평가 간 차이의 std는 .2–.8 delivery.
- **평가 delivery**(마지막 3회 평균 / AUC per step)
  - IPPO 3.10 / .41, PGCT 1.92 / .26, **Oracle 9.98 / 1.28**, All 3.45 / .69.
  - 요청 pickup은 20.7 / 2.2 / 10.4 / 15.2, return은 1.12 / .98 / 9.43 / 1.37.
- **onset**(3점 평균): pickup≥1은 IPPO 1.89M, PGCT 1.89M, Oracle 1.64M, All 1.79M. delivery≥1은 2.05 / 2.15 / 1.89 / 1.89M. delivery≥5는 Oracle만 2.25M.
- 모든 mode에서 0.5–1.5M은 보상이 거의 없는 정체 구간이었다.
- **agent별 delivery**(마지막 3회): IPPO [1.63, 1.47, 0, 0], PGCT [1.92, 0, 0, 0], Oracle [0, 9.98, 0, 0], All [0, 2.12, 1.33, 0]. 성능이 1–2명에게서만 나온다. **Oracle 우위는 Agent2 한 명의 선행 학습이며, 단일 seed로는 효과에 귀속할 수 없다.**
- **전이**(warm-up 후 전이 update 비율 / same / cross arc-update)
  - PGCT 77% / 3,496 / 5,590, Oracle 76% / 4,688 / 0, All 99% / 5,526 / 10,524. IPPO는 0이다.
  - PGCT gate의 pair별 개방 횟수는 같은 팀(826, 922)과 다른 팀(500–886)이 비슷해, 초기에 팀을 구분하지 않았다.
- **학습 지표**(1.5M 이후): entropy IPPO .83, PGCT .83, Oracle .76, All .89. value loss .002–.006. approx_kl 약 .003. 속도 job당 약 24 step/s.
- **참고**(조건 다름): 이전 home 1.0·complete graph PGCT는 2.4M에 delivery 26.7이었다.
- **한계**: seed 1개, 초기 구간, 중단된 run이다. 결론이 아니라 방향 관찰이다.

## E19. 신규 from-scratch 재실행 발견 — 2026-10-15 11:49 KST Codex, 읽기 전용 확인

- **경로:** `runs/scratch_20261015_093349_stage2/experiment.json`. 이번 작업 이전에 생성된 실행이다. 본 turn에서는 학습을 시작·중단·재시작하지 않았다.
- **조건:** E1 11×10, home.5, individual reward, 4agents/2teams, sensor/communication range5, IPPO/PGCT/Oracle/All, training seed1, 무작위 초기화(`init_checkpoint=""`), 요청 예산15,000,000/실제 반올림15,000,576 env steps.
- **진행 스냅샷(11:49):** IPPO200,704 / PGCT198,656 / Oracle196,608 / All199,680 env steps. `job_status.json`은 모두running이며 metrics 수정시각은11:48:58–11:49:28이다. 반복 확인에서 로그가 증가했다. CIM 프로세스 조회는 접근 제한으로 확인하지 못했다.
- **초기화:** 4mode의 `initial_weights_sha256.txt`가 모두 `f78ec390e0fad60c9bee9ea045525a79befbd71357f41235f04d61356b91c7e3`다.
- **평가:** 네 mode 모두 최근153,600step, delivery0.0/episode. deterministic masked, reset seeds5,000,001–5,000,020, 20×500. 초기 단계의 결과이며 실패로 판정하지 않는다.
- **E17과의 관계:** 중단된2.5M run의 resume이 아니라 같은 seed1의 처음부터 재실행이다. 독립training seed 추가나 이전곡선의 연장으로 세지 않는다.
- **근거:** `output/ijcai_review_20261015/live_run_snapshot.json`, 해당 run의 manifest/metrics/status/초기hash. 논문용 해석과 후속 제안은 같은 폴더REPORT 및 D23에 기록했다. 새 실험·추가 평가·tests는 실행하지 않았다.

## E20. 환경·canonical probe 시각화 — 2026-10-15 Claude Code, 읽기 전용

- **환경 그림**: `output/env_view_20261015/environment_seed5000001.png`. 실행 중 run의 환경 설정으로 그린 전체 맵, Agent1·2의 11×11 팀 기준 관측. 팀 목표가 (4,10)/(5,10)으로 인접한다.
- **probe 그림**: `output/env_view_20261015/probes/`(official 렌더링 48장과 contact sheet, decoded_bank.png, probes.csv, README.md).
- **bank 검증**: 저장소 builder로 같은 env 상태(reset seed 1)와 rng(seed+9173)로 다시 만든 bank가 run의 `canonical_bank_e1_s1.pt`와 완전히 같다.
- **중복 구조**: 고유 입력 26/48.
  - `dual_requested_*` 7종이 팀마다 하나의 입력으로 겹친다(8개 × 2그룹).
  - goal_pair·goal_zone_pair·lane_choice가 팀마다 하나로 겹친다(4개 × 2그룹).
  - neutral은 2개 × 2그룹이다.
- **겹침 원인(추정)**: 해독한 시야에서 요청 선반이 대부분 가장자리에 1개만 보인다. 서로 다른 두 번째 선반 위치가 11×11 관측 밖이라는 추정이며, builder 코드로는 확인하지 않았다.
- **상태 구성**: 빈 손 38, 요청 선반 적재 10(반납 단계 4). 요청되지 않은 선반을 든 상태는 0이다.

## E21. runs/ 정리 — 2026-10-15 Claude Code (사용자 승인 삭제)

- **승인 범위**: 사용자가 ① smoke·dry·실패 시험, ② 삭제된 09700 일반화 도구의 옛 평가 결과, ③ 중단된 `scratch_20261013_202352_stage2`의 checkpoint, ④ 큰 결과 폴더의 중간 checkpoint를 모두 선택했다. 500개 항목 27.26GB를 삭제해 runs/는 35GB에서 7.4GB가 됐다.
- **삭제 목록**: `output/cleanup_20261015/removed_manifest.json`.
- **위 E-항목에서 이제 존재하지 않는 경로**
  - ①: `sp_smoke/sp_check/sp_dry_20261008*`, `tr_*_20261008*`, `hard_smoke_20261013_v1~v3`, `lt_b_smoke_20261013`, `baseline_smoke_*_20261013`, `scratch_smoke*_20261013`, `scratch_pipeline_smoke_*`, `wbsmoke_20261013_*`, `verify_realcfg_20261013`, `communication_smoke`, `ex5_smoke`, `fov_passive`, `generalization_*`, `pgct09700_curriculum_*`, `pgct_09700_graph_smoke`
  - ②: `pgct_09700_generalization_6_8_seed12345`, `pgct_09700_maps_6_8_seed12345`
  - 해당 항목의 수치 기록은 노트에 남아 있지만, 원본 smoke 출력은 재검사할 수 없다.
- **유지한 것**
  - 실행 중인 `scratch_20261015_093349_stage2` 전체(삭제 후에도 4개 run 모두 로그 갱신 확인)
  - `sp_20261008`, 커리큘럼, `recurrent_ippo_consensus`, `hard_smoke_20261013_final`의 모든 로그와 final.pt
  - final.pt가 없는 run의 마지막 checkpoint: E3 `checkpoint_02700`, IPPO 사전학습 `checkpoint_00900`, PGCT 9700 사전학습 `checkpoint_09700`
  - E05가 참조하는 E1 PGCT `checkpoint_00600`
  - 중단된 run의 metrics/transfer/spatial 로그(E18 분석 근거)
- **영향**: 중간 시점 checkpoint를 재평가할 수 없게 됐다(예: Stage·E1/E2의 50k–3.9M 중간 모델). 전체 tests는 삭제 후 182 passed.

## E22. probe bank v2 / E1-F 구현 검증 — 2026-10-15 Claude Code, 구현·smoke만

- **bank**: `output/env_view_20261015/probes_v2/`(e1f·e1 각 gate 48 + transfer 64, 역할 내 중복 0, 검증 거부 0).
  - transfer 구성: 탐색 14, 목표 보임 16, 목표 안 보임 16, 반납 16, 목표 도착 2.
  - gate 위치는 모두 y≤4로, 어느 목표도 시야 밖이다.
- **smoke** `runs/smoke_e1f_v2_20261015`: e1f × ippo/pgct/oracle/all 64-step 모두 complete. PGCT의 `canonical_probes.pt`는 v2 gate와 같다. PGCT/Oracle/All의 증류 입력은 64개다(`distill_probe_count`).
- **tests**: E1-F 모서리 목표에서 적재 agent가 전진하면 delivery event가 발생함을 확인했다. 전체 183 passed.
- **한계**: E1-F 학습 정책에서 same/cross JS 판별력, Oracle 이득은 아직 미측정이다.

## E19 추가 — HANDOFF 이후 진행 상태 재확인 (2026-10-15 13:17 KST Codex)

- **같은 실행의 후속 확인:** `runs/scratch_20261015_093349_stage2`, E1 11×10/home .5/individual reward/4 agents·2 teams, 무작위 초기화 training seed 1, 기존 canonical bank, 15M 목표. IPPO 321,536 / PGCT 316,416 / Oracle 314,368 / All 318,464 env steps. 로그 시각 13:16:39–13:17:05, 네 status는 running, 속도 23.49–24.04 step/s다. 최초 확인보다 로그가 증가했고 초기 hash 4개도 동일하다.
- **최근 평가:** 네 mode 공통 307,200 step, deterministic masked 20×500, reset seeds 5,000,001–5,000,020. 요청 pickup/delivery/return 각 0.0이다. 초기 단계이며 성능 판정을 하지 않는다. 마지막 checkpoint는 모두 00250(256,000 step)이다.
- **실행 건강 확인:** .err.log의 Traceback/Error/Exception/failed 일치 0건. Gymnasium 정밀도·reward 형식 경고와 W&B 안내가 있다. 이는 전체 학습 정확성의 독립 검증은 아니다.
- **별도 구현 확인:** E22 seed 1의 64-step smoke 4 mode가 complete이며 PGCT/Oracle의 transfer.jsonl에서 distill_probe_count 64와 Oracle cross arc 0을 확인했다. v2 summary는 gate 48/transfer 64, 고유 입력 검증 통과다. 장기 E1-F/v2 실행은 발견되지 않았다. 183 tests 통과는 Claude 기록이고 재실행하지 않았다.
- **산출물/범위:** `output/status_review_20261015_131716/snapshot.json`. HANDOFF·RESEARCH_STATE·대화 원문을 갱신하고 통합 사본을 재생성한다. 새 학습·추가 평가·tests·알고리즘 변경·삭제·학습 중단은 하지 않았다.

## E23. 기존 E1 모델의 probe별 팀 구분 시각화 — 2026-10-15 Claude Code, 읽기 전용

- **코드·산출물**: `output/probe_team_check_20261015/`(probe_team_check.py, js_matrices.png, same_vs_cross.png, results.json).
- **모델**(고정, 실행 전후 hash 동일, CPU): Stage03 final(3.0M), Stage04 PGCT final(4.0M), sp_20261008 E1 IPPO/PGCT final(4.0M). 모두 E1 11×10이고 목표가 인접한다. Stage03 계열 seed 1이라 독립 반복이 아니다.
- **bank**
  - canonical 48: `sp_20261008/e1_pgct_s1/canonical_probes.pt`, 고유 26개.
  - v2 e1 gate 48: `output/env_view_20261015/probes_v2/probe_bank_v2_e1_s1.pt`.
  - v2 transfer 부분집합: 적재+목표 보임 18 / 적재+목표 비가시 16 / 빈손 탐색 14.
- **조건**: 같은 입력, zero hidden, unmasked, seq 1(PGCT 규칙), 자연로그 JS. 같은 팀 = A1–A3, A2–A4.
- **결과**
  - 20개(모델×bank) 모두 같은 팀 평균 JS > 다른 팀 평균 JS.
    - canonical: 같은 팀 .36–.42 / 다른 팀 .30–.36.
    - v2 gate: 같은 팀 .24–.28 / 다른 팀 .18–.22.
  - 쌍 단위 AUROC는 0–.62(대부분 .12–.38)다.
  - 가장 가까운 상대가 팀원인 agent는 4명 중 0–2명이다(우연 기대치 1.33).
  - 가장 가까운 쌍은 대부분 A3–A4(다른 팀)다.
  - 목표 보임과 목표 비가시 부분집합 사이에 뚜렷한 차이가 없다.
- **해석·한계**
  - E1에서는 bank 종류와 관계없이 actor 응답 JS로 팀이 구분되지 않는다. E11의 canonical 결과와 같은 방향이다.
  - v2 gate의 E1-F 효과는 E1-F 학습 모델이 없어 미검증이다.
  - E1 모델의 v2 gate 같은 팀 JS(.24–.28)가 0.12를 넘는다. threshold 우려(D31)의 참고 근거다.

## E23 추가 — agent별 행동 특성 (2026-10-15 Claude Code, 읽기 전용)

- **산출물**: `output/probe_team_check_20261015/agent_profiles.json`.
- **최종 평가 delivery**(A1/A2/A3/A4)
  - Stage04: 197/168/287/293
  - E1 IPPO: 186/162/267/278
  - E1 PGCT: 174/170/273/270
  - Stage03: 237/250/225/285
- **canonical probe 응답**
  - A3·A4는 Forward 위주다(A4 .81–.94, A3 .64–.75). A1·A2는 회전 비중이 크다(A2 Right .47–.57).
  - argmax 일치율: A3–A4 .69–.81, A1–A2 .08–.17, 같은 팀 쌍 .40–.58.
  - entropy는 .04–.3 nats로 정책이 거의 결정적이다.
- **해석**: E1 정책 유사도는 팀이 아니라 성능·행동 습관 묶음({A3,A4} / {A1,A2})을 따른다. Stage03에서도 A3–A4가 가까워 성능만으로는 다 설명되지 않는다.
- **가설(미검증)**: v2 gate에서 같은 팀원의 회전 방향 습관이 반대다(A1 Right, A3 Left). 공간 분담 가능성이며, 공간 기록은 미확인이다.

## E24. 현재 from-scratch run 약 6.4M 중간 분석 — 2026-10-18 Claude Code, 읽기 전용

- **대상**: `runs/scratch_20261015_093349_stage2`. E1 11×10, home .5, individual reward, seed 1, 무작위 초기화, IPPO/PGCT/Oracle/All, 각 15M 목표. 4개 모두 running이다.
- **진행(12:23 KST)**: IPPO 6,553,600 / PGCT 6,467,584 / Oracle 6,414,336 / All 6,502,400 step. job당 약 24 step/s, 15M까지 약 4.0–4.2일 남았다. run 6.4GB, C: 여유 1.5TB.
- **코드·산출물**: `output/scratch_review_20261018/`(analyze.py, analysis.json, overview.png). checkpoint probe JS는 CPU에서 고정 모델로 계산했다.
- **평가**(deterministic masked 20×500, reset seeds 5,000,001–5,000,020)

| mode | 배송 시작 ≥1(3점 평균) | 최근 5회 평균 delivery | 6.4M까지 AUC/step | agent별 최근 5회 delivery |
|---|---|---|---|---|
| IPPO | 2.05M | 24.3 | 10.4 | 12.3 / 11.9 / 0 / 0 |
| PGCT | 1.79M | 27.9 | 11.7 | 10.5 / 12.3 / 5.1 / 0 |
| Oracle | 1.64M | 32.3 | 13.2 | 11.0 / 11.3 / 9.9 / 0 |
| All | 3.99M | 20.4 | 3.7 | 4.3 / 7.0 / 5.3 / 3.8 |

  - 인접 평가 차이의 표준편차는 1.5–1.9다.
  - All은 늦게 시작했지만 4명 모두 배송하며, 현재 가장 가파르게 오른다. 요청 pickup은 52로 가장 많다.
- **실제 전이**(transfer.jsonl, warm-up 이후)
  - Oracle: 같은 팀(1–3, 2–4)만, update의 약 78–86%에서 전이했다.
  - All: 거의 모든 update에서 모든 쌍에 전이했다.
  - PGCT: 0–2M에는 모든 쌍에 열렸다(전이 update 87–95%). 2–3M 55%, 3–6M 12–33%로 줄었고, 같은 팀 arc는 3M 이후 거의 0이다. 남은 연결은 3–4와 1–2이며, 6M 이후에는 1–2 위주로 66%까지 늘었다.
- **canonical probe JS**(checkpoint 1M 간격): 3M 이후 대부분 시점·mode에서 같은 팀 평균 > 다른 팀 평균이다. Oracle 같은 팀 .29–.36이다. 2M 이후 같은 팀·다른 팀 평균이 모두 0.12를 넘는다.
- **E17과 비교**(같은 seed·초기 가중치)
  - IPPO는 2.46M까지 평가값이 완전히 같다.
  - PGCT/Oracle/All은 첫 평가 차이가 1.48/1.38/1.59M에서 생긴다. 원인은 미확인이며, 전이 경로의 비결정적 연산을 추정한다.
  - All 배송 시작이 1.79M → 3.99M으로 달라졌다. Oracle은 두 번 모두 A2가 먼저 배웠다.
  - 단일 seed 결과의 불안정성을 보여주는 근거다.
- **한계**: seed 1, 15M 중간 시점이다. 우열이나 효과 귀속 결론이 아니다.

## E25. E1-F 실험 실행기(pilot/main)·팀 보상 옵션·팀 분화 분석 도구 — 2026-10-18 Claude Code, 구현·smoke만

- **구현**
  - `train_from_scratch.py`에 `--team-reward individual|team`을 추가했다. 기본값 individual은 이전과 동일하며, 환경의 기존 `team_reward_mode`를 쓴다. team이면 배송·반납 보상이 팀 전원에게 가고 shaping은 개인으로 남는다. job 이름에 `_teamrew`가 붙는다.
  - `run_from_scratch.ps1`과 `run_scratch_pipeline.ps1`에 `-TeamReward`를 전달한다.
  - 새 `run_e1f.ps1`
    - pilot: IPPO individual/team, 5M.
    - main: 4 mode(canonical bank) + PGCT(v2 bank), 10M.
  - 새 `e1f_team_divergence.py`: checkpoint별로 배송 중 왼쪽 이동 비율(agent별)과 v2 gate·canonical 같은 팀/다른 팀 JS를 계산한다.
- **검증**
  - smoke `runs/smoke_e1f_runner_20261018_{pilot_individual,pilot_team,main_canonical,main_v2}`: 7 job 모두 complete.
  - checkpoint의 effective_env_kwargs에 team_reward_mode와 goals [(0,10)],[(9,10)]이 들어 있다.
  - canonical probe builder가 E1-F 목표를 사용한다.
  - 분석 도구 검증: E1 `scratch_20261015_093349_stage2/e1_oracle_s1/checkpoint_06000`에서 2×500 rollout, left_share A1–A3 .56/.47/.51(A4는 운반 없음). 출력은 scratchpad에만 남겼다.
  - tests 2개를 추가해 전체 185 passed.
- **미실행**: E1-F pilot/main 본 학습은 사용자 승인 대기다.
- **관련 관측(E24 보완, 현재 E1 run 6.4M, 읽기 전용)**
  - 최근 5회 평가 agent별 pickup: IPPO A3·A4 0(이동 비율 .37/.40), PGCT·Oracle A4 0.
  - 평가에서 episode당 pickup 1 이상이 된 시점: IPPO 1.89/2.05M/없음/없음, PGCT 2.46/1.43/5.27M/없음, Oracle 2.30/1.38/2.30M/없음, All 5.22/3.99/1.38/5.07M.

## E24 추가 — 사용자 요청으로 6.5M에서 중지 (2026-10-18 12:43 KST Claude Code)

- 사용자 지시("현재 실험은 중지하고, 다음 실험으로 run_e1f main 진행")에 따라 pipeline PowerShell(PID 21540)과 학습 4개(PID 27140/23940/2980/15160)를 종료했다. PID는 pipeline 로그 기록과 일치함을 확인했다.
- 종료 직전 상태
  - 마지막 env_steps: IPPO 6,579,200 / PGCT 6,498,304 / Oracle 6,444,032 / All 6,528,000.
  - 마지막 checkpoint는 모두 `checkpoint_06250.pt`다.
  - `final.pt`는 없고, `job_status.json`은 trainer가 쓴 running 그대로다. W&B run은 중단 상태로 남는다.
- 중지 기록 파일: `runs/scratch_20261015_093349_stage2/STOPPED_BY_USER.txt`. 로그·checkpoint는 모두 보존했다.
- 보완(같은 날, 사용자가 W&B running 표시 지적): 강제 종료한 학습 프로세스의 자식 wandb-core 4개와 wandb-xpu 4개가 남아 heartbeat를 보내고 있었다. 부모 PID로 확인해 종료했다. 이후 W&B에서는 crashed로 표시된다. 앞으로 학습을 중지할 때는 프로세스 트리 전체를 종료한다.
- W&B API 확인: 마지막 heartbeat는 13:45:30–33 KST이고, 13:55:14 KST에 4개 모두 `crashed`로 바뀌었다.
- 이 run은 15M 계획 대비 약 6.5M에서 끝난 seed 1 결과다. D36의 10M 고정 예산과도 다르므로, 이후 seed 반복 시 예산을 다시 정해야 한다.

## E26. E1-F main 본 학습 — 2026-10-18 12:43 KST 시작(사용자 승인, Claude Code 실행), 진행 중

- **명령**: `.\latent_team\run_e1f.ps1 -Phase main -Track`
- **조건**
  - E1 맵, 팀 목표 (0,10)/(9,10), home .5, individual reward, seed 1, 무작위 초기화, 10,000,384 step, W&B 프로젝트 rware-from-scratch.
  - `runs/e1f_20261018_124343_main_canonical`: IPPO/PGCT/Oracle/All, canonical bank.
  - `runs/e1f_20261018_124343_main_v2`: PGCT, probe bank v2.
  - 병렬 5개 프로세스, 스레드 4.
- **분석**(읽기 전용, 언제든): `python latent_team/e1f_team_divergence.py --run runs/e1f_20261018_124343_main_canonical`(main_v2도 동일).
- **참고**: E1-F pilot(IPPO 개인/팀 보상)은 실행하지 않고 바로 main을 시작했다. 팀 보상 비교는 미실행이다.
- **시작 확인(12:45)**: 5개 모두 첫 update를 기록했고 Traceback은 없다. job당 약 20–21 step/s로, 10M까지 약 5.5일(10월 24일 무렵) 예상이다. 초기 hash는 5개 모두 `f78ec390…`로 E1 seed 1과 같다.

## E27. E1 run(6.4M 중단) 논문용 자료 — 2026-10-18 Claude Code, 읽기 전용

- **폴더**: `output/paper_e1_20261018/`
  - `make_paper_materials.py`
  - `figures/`: fig1–4, figA1 (PDF+PNG)
  - `tables/`: table1–3, tableA1 (LaTeX)
  - `RESULTS_DRAFT.md`(영어 문단·캡션·한계), `README.md`(한국어, 주장 가능 범위), `numbers.json`
- **조건**: `runs/scratch_20261015_093349_stage2`, seed 1, 공통 평가 구간 0–6.4M. 평가는 51,200 step 간격, deterministic 20×500, reset seeds 5,000,001–5,000,020.
- **결과(seed 1)**
  - 배송 시작(≥1, 3점 평균): IPPO 2.05 / PGCT 1.79 / Oracle 1.64 / All 3.99M.
  - 학습 중 평균 배송: 10.4 / 11.7 / 13.2 / 3.7.
  - 최근 5회 배송: 23.2±1.1 / 27.3±1.4 / 32.2±1.7 / 19.4±1.6.
  - 배운 agent: 2 / 3 / 3 / 4.
  - PGCT 전이 update 비율은 91% → 37% → 29%이고, 같은 팀 arc/update는 1.86 → 0.10 → 0.03이다.
  - probe JS 같은 팀/다른 팀(6M): IPPO .27/.18, PGCT .26/.19, Oracle .29/.23, All .15/.14.
- **정정**
  - E17 All의 delivery 시작은 1.89M이다(E18과 동일). E24·T034에 쓴 1.79M은 pickup 기준과 혼동한 값이다.
  - E24 표의 최근 5회 값은 각 run의 마지막 평가 기준이다. E27은 공통 구간 6.4M 기준이라 IPPO 값 등이 조금 다르다.
- **한계**: seed 1개, 미수렴, E1의 팀 대칭성. 성능 우열·공유 범위 가설은 다중 seed 전까지 관찰로만 쓴다.

## E28. 발견 시점과 발견 이후 속도 분석 + checkpoint 렌더링 영상 — 2026-10-19 Claude Code, 읽기 전용

- **E1-F 진행(10:48 KST)**: 5개 프로세스 running, 약 1.52–1.55M step, job당 약 19 step/s. 평가(20×500 deterministic)의 delivery는 IPPO가 1.48M에 0.05 한 번, 나머지는 0이다. 같은 0.25M 구간 학습 delivery(1.25–1.5M)는 E1 Oracle 0.46, PGCT 0.20에 비해 E1-F는 IPPO 0.06, 나머지 0이다. E1-F가 더 늦게 시작한다.
- **E1-F gate(0–1.5M)**: PGCT canonical은 update당 같은 팀 arc 1.5–2.3/4, 다른 팀 3.5–3.7/8이다(비율 약 0.4 대 0.44). v2는 1.1–1.4/4, 1.9–2.8/8이다(약 0.27 대 0.27). 아직 같은 팀을 고르지 않는다.
- **발견 정의**: train metric에서 agent 자신의 delivery/episode를 20 iteration 이동평균한 값이 0.3M 이후 처음 0.2 이상이 되는 시점이다. 임계값은 임의로 정했다.
- **첫 발견 시점(seed 1, 같은 초기화)**: E17(`runs/scratch_20261013_202352_stage2`) / E24(`runs/scratch_20261015_093349_stage2`) 순서로 IPPO 1.68/1.68M, PGCT 1.63/1.13M, Oracle 1.51/1.32M, All 1.54/3.61M이다. 전이 mode는 비결정성만으로도 0.2–2.1M씩 흔들린다.
- **agent별 발견(E24)**: IPPO [1.68, 1.84, -, -], PGCT [1.53, 1.13, 5.2, 5.38], Oracle [2.14, 1.32, 3.12, -], All [4.9, 3.91, 3.61, 4.88](M).
- **첫 발견 이후 팀 delivery/episode(학습, 20 iteration 평균)**: E24 기준 +0.5M/+1M/+2M에서 IPPO 5.4/8.35/23.25, PGCT 2.35/3.55/17.5, Oracle 2.4/10.6/23.15, All 5.4/9.05/21.45였다. E17의 +0.5M은 IPPO 5.4, PGCT 2.45, Oracle 3.3, All 4.55다. 발견 직후 PGCT가 IPPO보다 느리고, 이는 두 run에서 모두 같다.
- **영상**: `latent_team/render_checkpoint_videos.py`(새 파일, 읽기 전용 평가)를 썼다. trainer evaluate()로 deterministic masked 1 episode, 500 step, reset seed 5,000,001, 10 fps다. 출력은 `output/videos_20261019/{e1f,e1f_v2,e1}/`이다. E1-F는 그 시점의 최신 checkpoint를 썼으므로 job마다 1.28M 또는 1.54M으로 step이 다르다. PGCT는 `e1f_1.54M/`에 1.54M 영상을 추가했다. E1은 06250(6.40M)이고, episode 1개의 delivery는 IPPO 15, PGCT 37, Oracle 35, All 25다. E1-F 영상은 모두 0이다. 렌더링의 "PGCT accepted links" 표시는 고정 평가에서 gate를 0으로 두므로 항상 0이다.
- **한계**: seed 1뿐이다. E17과 E24는 독립 seed가 아니라 같은 seed의 비결정적 반복이다. train delivery는 확률 정책이므로 잡음이 있다.
- **재현**: scratchpad의 `discovery.py`는 보존하지 않았다. 위 정의를 그대로 따르면 된다.
- **추가(2026-10-19, 평가 기반 ΔT_k)**
  - 정의: onset은 평가 delivery가 3회 연속 1 이상이 되는 첫 시점이다. T_k는 평가 3회 이동평균이 처음 k 이상이 되는 시점이다. ΔT_k = T_k − onset.
  - E24 결과(onset, ΔT5/ΔT10/ΔT20)
    - IPPO: 2.05M, 0.82/1.13/3.02
    - PGCT: 1.74M, 0.77/1.54/2.41
    - Oracle: 1.59M, 0.72/1.38/3.07(ΔT30 4.25)
    - All: 3.94M, 0.87/1.38/2.51
  - E17 onset은 IPPO 2.05, PGCT 2.10, Oracle 1.89, All 1.79M이다.
  - 전이 비율 재계산(E24 PGCT): 전이가 있었던 update의 비율은 0–1M 0.95, 1–2M 0.87, 2–3M 0.55, 3–4M 0.19, 4–5M 0.12, 5–7M 0.44다. 같은 팀 arc는 2–3M 0.18, 3M 이후 약 0이다. 첫 전이는 0.1M이다. 기존 E24 기록, GPT 인용과 일치한다.

## E29. probe 팀 식별의 신뢰성(시간 변화 대 팀 신호)과 W&B probe 컬러맵 도구 — 2026-10-19 Claude Code, 읽기 전용

- **gate EMA(`communication.jsonl` policy_distance_ema)**: 같은 팀 쌍이 다른 팀 쌍보다 가까운지 record별 AUROC를 계산했다(1이면 같은 팀이 더 가깝다).
  - E1 PGCT(E24): 0–6.5M 구간별 0.28–0.49.
  - E1-F canonical PGCT: 0–1.5M 0.33–0.61.
  - E1-F v2 PGCT: 0.25–0.52.
  - 모든 쌍의 거리가 학습과 함께 커진다. E1 PGCT 같은 팀 거리는 0.026에서 0.242로, 다른 팀 거리는 0.022에서 0.201로 늘었다. 고정 τ=0.12 때문에 초기에는 거의 모두 연결되고 이후 끊긴다. E1의 전이 비율이 95%에서 12%로 줄어든 것과 같은 방향이다.
- **checkpoint 기준 자기 변화와 팀 신호**: gate와 같은 규칙(zero hidden, unmasked, ln JS, probe 평균)을 썼다.
  - E1-F의 0.256M 간격 자기 변화(agent 평균)는 canonical PGCT 0.012–0.106, v2 PGCT 0.030–0.048, IPPO(v2 bank) 0.037–0.061이다.
  - 같은 시점의 팀 신호(다른 팀 − 같은 팀 JS)는 −0.077~+0.029다.
  - E1 PGCT의 1M 간격 자기 변화는 0.076–0.127, 팀 신호는 −0.072~+0.006이다.
  - agent가 0.25M 동안 스스로 변하는 양이 팀 간 차이보다 크거나 비슷하다.
- **E1-F v2 1.54M 컬러맵**: A1·A2는 FWD 위주, A3·A4는 LEFT 위주다. 가장 가까운 쌍은 1–2(0.045)와 3–4(0.036)로 둘 다 다른 팀이다. 같은 팀은 1–3 0.128, 2–4 0.229다. 행동 습관 묶음이 팀보다 우세하다(E1의 D33·E23과 같은 패턴). 단, E1-F는 아직 배송을 배우기 전이다.
- **도구**: `latent_team/probe_wandb_viz.py`(새 파일). checkpoint별 agent×probe×action 확률 컬러맵, 4×4 평균 JS와 쌍별 per-probe JS, 이전 checkpoint 대비 drift 맵을 만든다. 스칼라는 js_same/js_cross/auroc/pairs_below_tau/drift_mean이다. job마다 별도 W&B run(`<job>-probes`, group=plan 폴더, job_type=probe_viz)에 올리고 학습 run에는 쓰지 않는다. `--watch`로 새 checkpoint를 따라간다.
- **검증**: v2 run 2개 checkpoint로 offline smoke를 했다(`tmp/pv_smoke`). 처음에는 Windows 경로 길이 초과로 wandb init이 실패해 run id와 dir을 줄였다. 온라인 업로드와 watch는 아직 시작하지 않았다.
- **한계**: seed 1뿐이다. E1-F는 1.5M까지만 봤다. 팀별 목표 방향 행동은 배송을 배운 뒤에 나타날 수 있으므로, E1-F 판정은 2M 이후로 미룬다.

## E30. E1 이전 학습 모델에서 probe가 팀을 연결하는가 + 연결 규칙(hard/sigmoid/softmax) 비교 — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/probe_connect_20261019/`(analyze.py, results.json, probe_connect.png).
- **모델**(final, 고정, hash 불변): 커리큘럼 Stage01(home 1.0)·Stage02(.875)·Stage03(.75)·Stage04 PGCT(.5), sp_20261008 E1 IPPO/PGCT, E2 IPPO/PGCT. E3는 final이 없어 제외했다. 모두 seed 1이고 커리큘럼 계열이다.
- **bank**
  - canonical: 모델 자신의 bank. IPPO는 같은 맵 PGCT bank를 썼다.
  - ablated: canonical에서 팀 상대 입력을 0으로 만든 것. requested_shelf·relevant_goal 채널과 ego carrying_requested/return_phase/shelf_home_*를 지웠다.
  - v2_gate(E1), v2_gate_ablated.
  - E11 국소 증거: Beta(1,1) 사후평균, 20 episode. E11 쌍 순서를 반영했다.
- **규칙**
  - hard: 1[D≤0.12].
  - sigmoid: σ((0.12−D)/0.03).
  - softmax: exp(−D/T), T는 비대각 D의 중앙값.
  - 지표는 팀원 가중치 비율(우연 1/3)과 AUROC다.
- **결과(canonical AUROC / ablated AUROC / hard 같은 팀·다른 팀 arc)**
  - Stage01: 1.00 / 1.00 / 4·0. 팀을 정확히 연결한다.
  - Stage02: .62 / 1.00 / 0·0.
  - Stage03: .50 / .75 / 0·0.
  - Stage04: .50 / .38 / 0·0.
  - E1 IPPO: .38 / .62 / 0·0. v2_gate_ablated는 .75.
  - E1 PGCT: .38 / .25 / 0·2.
  - E2 IPPO: .38 / .50 / 2·4.
  - E2 PGCT: .50 / .75 / 2·4.
  - 국소 증거는 6개 모델 모두 AUROC 1.00이고, 4명 모두 가장 가까운 상대가 팀원이었다.
- **연결 규칙**
  - JS 신호(canonical)에서 sigmoid 팀원 비율은 Stage03 이후 0.08–0.35, softmax는 0.30–0.34다. 우연 수준이거나 그 아래다.
  - 신호가 있는 경우(Stage01, Stage02 ablated)에는 sigmoid가 0.97–1.00이다.
  - 학습이 끝난 모델 대부분에서 모든 쌍의 JS가 0.12를 넘어 hard gate는 아무것도 연결하지 않았다.
- **해석**
  - 팀별로 공간을 나눠 쓰면(home 1.0) probe가 팀을 가른다.
  - 공간이 섞일수록(home .875→.5) 같은 입력에 대한 정책이 수렴해 팀 신호가 사라진다(명제 6, D32와 같은 방향).
  - 팀 상대 입력을 지우면 Stage02–03, E1 IPPO, E2 PGCT에서 신호가 일부 살아난다. 하지만 학습 분포 밖 입력이고, Stage04·E1 PGCT에서는 실패한다.
  - 실제 행동 증거는 신호가 강하다.
- **한계**: seed 1, 쌍 6개(같은 팀 2)라 AUROC 해상도가 거칠다. 국소 증거는 관측자가 자기 팀 목표를 안다는 점과 이웃 귀속(익명 agent) 가정에 의존한다(E13). 학습 초기에 증거가 있는지는 확인하지 않았다.

## E31. 보상 probe 오프라인 검증(학습된 보상 예측기, counterfactual 상한, 보상 기록, 보상 가중치) — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/reward_probe_20261019/`(reward_probe.py, results/results.json, 조건별 npz, run.log). agent 학습은 하지 않았다. 정책은 데이터 생성에만 썼고 가중치 hash는 불변이다.
- **조건**: E1-F 무작위·IPPO 0.26/0.77/1.54M, E1 무작위·IPPO 1.54/6.40M(E24). 조건마다 30 episode×500 step, 확률 정책(masked), reset seed 7,100,001부터.
- **예측기**: agent별 MLP(obs+행동 one-hot → 보상), 자기 데이터만, init seed 3개, 25 epoch.
- **probe**: v2 gate 48 상태×5 행동 = 240 query. 거리는 평균 |r̂_i − r̂_j|다.
- **counterfactual 상한**: ProbeRealiser로 각 팀 agent를 probe의 물리 상태에 두고 행동을 실행했다. AUROC는 항상 1.00이다. 하지만 팀에 따라 실제 보상이 다른 query는 **E1-F 18/240, E1 5/240뿐**이다.
- **학습된 보상 probe AUROC**
  - E1-F: 무작위 .38, 0.26M .50, 0.77M .62, 1.54M .50.
  - E1: 무작위 .25, 1.54M .50, 6.40M .25.
  - 예측기와 실제 보상의 상관(자기 팀/다른 팀): .35/.34, .45/.39, .20/.25, .10/.23(E1-F). 자기 팀 보상 구조를 따로 배우지 못한다.
  - label로 판별 query만 골라도(oracle 가중) .25–.75다.
  - 검증 R²는 .13–.65, 희소 사건(|r|≥0.1) MAE는 .13–.35다.
  - 학습된 정책은 "요청 선반 운반" 상태를 거의 방문하지 않는다(E1-F 0.26–1.54M에서 agent별 0–866 샘플). 무작위 정책은 507–953 샘플을 모으지만 진척 보상(0.005)이 잡음에 묻힌다.
- **보상 기록(예측기 없음)**: AUROC .12–.50.
- **보상 가중치**: reward_magnitude, action_sensitivity, SNR 가중치가 판별 query에 주는 비중은 2–10%로, 균등 가중치(2–7%)와 비슷하다.
- **결론**: 학습된 보상 예측기 probe는 현재 데이터 범위에서 팀을 가르지 못한다. GPT의 지적(예측기 ≠ 보상 함수, 외삽, 초기 데이터 부족)과 일치한다.

## E32. probe별 팀 분리 능력 Δ_m과 probe 가중치(정책 probe) — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/probe_weight_20261019/`(delta_m.py, results.json, delta_m.png).
- **정의**: Δ_m = 다른 팀 JS − 같은 팀 JS(probe m별, PGCT 규칙).
- **가중치**: uniform / 보상 크기 / 팀 보상 차이(env counterfactual, 설계자 사전지식) / agent 분산(label 없음) / oracle held-out / oracle in-sample.
  - oracle held-out은 E1 계열에서는 다른 모델들의 Δ_m(leave-one-model-out), E1-F에서는 같은 job의 이전 checkpoint Δ_m으로 정했다.
  - oracle in-sample은 같은 데이터로 고른 것이라 낙관적이다.
- **E1 계열(Stage01–04, sp E1 IPPO/PGCT, 공통 v2 E1 bank)**
  - uniform AUROC: v2 .12–.62, ablated .38–1.00.
  - oracle held-out: v2 .62–.88, ablated .75–1.00. 다른 모델에서 고른 판별 probe가 이 모델에도 통한다.
  - 팀 보상 차이 가중치: ablated에서 6개 모두 .88–1.00(v2에서는 .12–.75).
  - 보상 크기 가중치는 uniform과 거의 같다.
  - Spearman(Δ_m, |r|)은 −.33~+.12다. 보상 크기는 팀 판별력을 따르지 않는다.
- **E1-F(5개 job × 0.51/1.02/1.54M, 공통 v2 E1-F bank)**
  - uniform .25–1.00(대부분 .25–.62).
  - 이전 checkpoint로 정한 held-out 가중치는 .25–1.00(대부분 .25–.50).
  - Δ>0인 probe 비율은 0–69%로 시점마다 바뀐다.
- **한계**
  - 쌍이 6개라 AUROC 해상도가 1/8이다.
  - E1 계열은 같은 커리큘럼 계보라 leave-one-model-out이 독립 검증이 아니다.
  - seed 1뿐이다.
  - 팀 보상 차이 가중치는 팀 목적을 아는 설계자 사전지식이라, 잠재 팀 추론의 공정한 비교가 아니다.

## E33. 팀 판별 probe의 상태 분석과 설계 규칙 시험 bank — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/probe_redesign_20261019/`(inspect_probes.py, probe_table.json, junction_bank.py, junction_results.json, far_band_check.py, far_band_results.json).
- **판별 probe의 공통점**
  - ablated에서 6개 모델 모두 Δ>0인 probe는 #2 (4,1) RIGHT, #26 (4,1) UP, #33 (4,1) RIGHT, #42 (7,0) UP, #38이다.
  - 모두 맵 위쪽 통로(y 0–1, 목표 행에서 9–10칸)에서 선반을 든 채 오른쪽이나 위를 보고 있다.
  - 팀 차이는 회전 선택에서 나온다. 빨강은 LEFT(서쪽 길), 파랑은 RIGHT/FWD(동쪽 길)다. 예: #26 빨강 LEFT .85 / 파랑 RIGHT .88, #42 빨강 LEFT .93 / 파랑 RIGHT .84.
  - 방향별 평균 Δ(ablated)는 RIGHT +.117, UP +.094, LEFT −.067, DOWN −.012다. x 4–5 구간이 +.125로 가장 크다.
  - 원 v2(입력 그대로)에서 일관된 probe는 #6·12·15로 적고, 팀 차이는 주로 RIGHT 회전 확률이다.
- **해석**: E1에서 남은 팀 신호는 목표 방향이 아니라 팀별 경로 습관(서쪽/동쪽 길)이다. 커리큘럼에서 팀별 홈 구역이 있었던 영향(팀 = 구역 혼동)일 가능성이 크다(가설). 팀 상대 입력(특히 운반 선반의 home dx/dy, requested 표시)이 있으면 두 팀이 같은 단서를 따라가 이 습관이 가려진다.
- **설계 규칙 시험(균등 가중치, label 미사용)**
  - 규칙 A: 분기 ≥3인 통로 칸, 목표 비가시, 모든 방향, 팀 상대 입력 중립. E1에서 80개다.
  - 규칙 B: A에 "목표 행에서 9칸 이상 먼 띠"를 더한 것. 32개다. E1 probe를 본 뒤 정했으므로 E1에서는 in-sample이다.
  - AUROC 평균(≥.75인 모델 수)

| bank | E1 계열 6개 | E1-F 15개(0.51/1.02/1.54M) |
|---|---|---|
| v2 (현행) | .42 (0) | .49 (1) |
| v2 ablated | .67 (3) | .48 (2) |
| 규칙 A, 입력 그대로 | .48 (0) | .47 (1) |
| 규칙 A, 중립 | .56 (1) | .53 (4) |
| 규칙 B, 중립 | .67 (4) | .53 (4) |

- **결론**: 규칙 B는 E1 계열에서 v2 ablated와 같은 평균에 ≥.75 모델이 더 많다. 하지만 in-sample이고, E1-F 초기(배송 학습 전)에서는 어떤 bank도 우연 수준이다. 규칙의 일반성은 E1-F 2M 이후에 미리 정한 그대로 평가해야 한다.
- **한계**: seed 1, 쌍 6개, E1 계열 계보 공유. "중립" 입력은 "보상 대상이 아닌 선반을 들고 home을 모르는" 상태로 해석되며, 실제로 방문하는 상태이지만 빈도는 확인하지 않았다.

## E34. GPT 검토 대응: 중복 probe, E1-F Δ_m(t) 일관성, 독립 계보 E1 검증 — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/probe_redesign_20261019/gpt_checks.py`, `gpt_checks.json`.
- **중복**: E1 v2, E1 v2 ablated, E1-F v2, E1-F v2 ablated(각 48개), 규칙 B 중립(32개)은 모두 입력이 고유하다. E32·E33의 결과는 중복 투표로 설명되지 않는다. 중복 문제(고유 26/48, E04)는 canonical bank에만 해당한다.
- **E1-F Δ_m(t) 순위 일관성**(Spearman)
  - 같은 job의 연속 checkpoint(0.51→1.02→1.54M): 평균 +.23(−.14~+.65).
  - 1.54M에서 job 간 평균: −.07.
  - 5개 job 모두에서 Δ>0인 probe는 없다.
  - 배송 학습 전 단계에서는 판별 probe가 시간과 job에 따라 일관되지 않다(GPT의 "부호가 바뀌는 신호" 유형).
- **독립 계보 검증**: from-scratch E1(E24, 커리큘럼 없음) IPPO/PGCT/Oracle/All checkpoint_06250(6.4M). 이 모델들은 규칙을 정할 때 보지 않았다.
  - AUROC: v2 .38–.50, v2 ablated .25–.50, 규칙 B 중립 .25–.62.
  - 커리큘럼 계열에서 찾은 판별 probe와 규칙 B는 독립 계보로 일반화되지 않는다.
  - E33의 E1 신호가 커리큘럼 홈 구역 습관이라는 가설과 맞는다.

## E35. Random-gated 대조군 구현 — 2026-10-19 Claude Code, 구현·smoke만

- **코드**
  - 새 파일 `latent_team/random_gated.py`.
  - `latent_team/exp_b_oracle_consensus.py`: `matched_transfer`에 `allocation_override` 인자를 추가했다(기본 None이면 기존 동작 그대로). `check_allocation`에 random을 허용했다.
  - `latent_team/train_from_scratch.py`: `random` mode와 `--random-reference` 옵션을 추가했다.
  - 테스트 3개를 추가했다. trainer·환경·probe 파일은 수정하지 않았다.
  - 실행 중인 E1-F 프로세스는 시작 시 모듈을 이미 읽었으므로 영향이 없다.
- **규칙**
  - update마다 참조 PGCT `transfer.jsonl`의 실제 arc 수를 목표로, 현재 물리 이웃 arc 중 균등 무작위로 고른다. receiver별 합은 1이다.
  - 부족분은 이월한다(최대 12). 별도 난수기는 seed+7,340,033이다.
  - plan 단계에서 참조의 mode, update 연속성, rollout 길이, 예산 이상의 길이를 검사한다(smoke는 길이 검사 제외).
- **검증**
  - `tests` 전체 188 passed(기존 185 + 3).
  - 실제 E1-F PGCT 기록(1,650 update, 1.69M)을 그 기록의 물리 후보로 재생하면 목표 8,686 arc를 모두 사용한다(이월·미충족 0).
  - 같은 팀 arc 비율은 PGCT .329, Random .362다. 0–1.69M 구간의 PGCT는 무작위보다 같은 팀을 더 고르지 않는다.
  - smoke `runs/smoke_random_gated_20261019`(e1f, v2 bank, All·Random 각 64 step). 참조는 PGCT update 101–140을 1부터 다시 번호 매긴 scratch 파일이다.
    - Random update 1: 목표 4 / 후보 12 / 사용 4.
    - Random update 2: 목표 8 / 후보 6 / 사용 6 / 이월 2.
    - receiver별 합은 1이고 distill probe는 64개다. peer loss는 .0117/.0089이고, 초기 hash는 All과 같다(baaa10bf…).
- **본 학습은 실행하지 않았다.** 참조가 예산 이상을 덮어야 하므로 E1-F main PGCT(10M)가 끝난 뒤 가능하다.
- **한계**: arc 수만 맞춘다. donor가 있는 receiver 비율(`update/pgct_peer_weight`)은 PGCT와 다를 수 있다.

## E36. 잘 학습된 커리큘럼 모델(Stage03·Stage04)로 보상 probe 재시험 — 2026-10-19 Claude Code, 읽기 전용

- **사용자 지적**: E1 이전에 잘 학습된 50:50 모델이 있었다. Stage04 PGCT는 home .5, 4.0M, 평가 delivery 47.25이고, Stage03은 home .75, 3.0M, 49.85다.
- **방법**: E31과 같다(`output/reward_probe_20261019/reward_probe.py --only "Stage03 3.0M" "Stage04 PGCT 4.0M" --results-dir results_stage`, 결과 `results_stage/`). 각 모델의 자기 환경(E1 맵, 인접 목표)에서 30×500 롤아웃을 했고, v2 E1 gate bank 240 query를 썼다. agent 학습은 없다.
- **결과**
  - **데이터 문제는 해결됐다.** 예측기 검증 R²는 .94/.92, 희소 사건 MAE는 .086/.096, 요청 선반 운반 샘플은 agent별 3,136–5,165개다. E31(R² .13–.65, 0–953개)보다 훨씬 좋다.
  - 그래도 **보상 probe AUROC는 .62(Stage03), .38(Stage04)**다. 판별 query만 쓰는 oracle 가중도 .00/.25이고, 보상 기록 기준선은 .12/.25다.
  - 정책 probe는 .50/.25다.
  - counterfactual 상한에서 팀마다 실제 보상이 다른 query는 **5/240(2%)**뿐이다. 예측기와 실제 보상의 상관(자기 팀/다른 팀)은 −.09/−.09, −.10/−.09로 두 팀이 같다.
- **해석**: E1에서 보상은 "팀 기준 관측"의 함수로 보면 두 팀이 사실상 같다. 내 팀 목표·요청 표시를 따라가면 보상을 받고, 두 목표가 붙어 있기 때문이다. 예측기가 r(o,a)를 정확히 배워도 두 팀의 r(o,a)가 같으므로, 같은 입력 probe로는 원리적으로 팀을 가를 수 없다(명제 6의 보상판). E1-F는 목표가 양 끝이라 팀마다 보상이 다른 query가 18/240(7.5%)이므로 결론이 다를 수 있다. 다만 지금 E1-F에는 충분히 학습된 모델이 없다.
- **한계**: seed 1, 쌍 6개, 같은 커리큘럼 계보다.

## E37. 궤적(시퀀스) probe — 정책·보상, 길이 1과 비교 — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/trajectory_probe_20261019/`(trajectory_probe.py, results/results_both.json, results/summary_both.md, run.log). agent 학습은 없고 가중치 hash는 불변이다.
- **공통 궤적**: 고정 source 정책 Stage04 PGCT를 각 환경(E1: Stage04 자기 환경, E1-F: E1-F env)에서 40×500 롤아웃했다. 길이 12 창을 source agent 4명에게 균등하게 뽑았다(팀 label 미사용).
  - carry_memory: 요청 선반 운반 + 처음 3 step 안에 목표가 보였다가 마지막 4 step에는 안 보임.
  - carry: 운반만.
  - random.
  - 창 수: E1은 6/64/64(carry_memory 6개뿐), E1-F는 64/64/64.
- **(a) 정책**: hidden을 이어 가며 step별 JS를 평균한 traj_all·traj_late와, 같은 프레임을 하나씩 hidden 0으로 넣은 len1을 비교했다. 15개 모델이다.
- **(b) 보상**: 평가 모델 자신의 20×500 롤아웃으로 agent별 MLP(len1)와 GRU(traj_step, traj_sum)를 학습하고, 창에 기록된 행동으로 질의했다. 5개 모델이다.
- **결과(AUROC 평균, 1이면 같은 팀이 더 가깝다)**

| | 창 | len1 | 궤적 |
|---|---|---|---|
| 정책, E1 커리큘럼 6개 | carry / random | .56 / .60 | .58 / .56 (late .54 / .54) |
| 정책, E1 처음부터 4개 | carry / carry_memory | .56 / .69 | .41 / .69 |
| 정책, E1-F 1.54M 5개 | carry / carry_memory / random | .38 / .38 / .40 | .35 / .42 / .45 (late .35 / .47 / .53) |
| 보상, Stage03·04(R² .85–.93) | carry / random / carry_memory | 1.00 / 1.00 / .38 | step 1.00 / 1.00 / .25, sum .94 / .50 / .25 |
| 보상, 처음부터 E1 IPPO(R² .50/.37) | carry / carry_memory / random | .62 / .75 / .50 | step .38 / .25 / .50, sum .25 / .25 / .25 |
| 보상, E1-F IPPO·PGCT(R² ≤ .07) | carry / carry_memory / random | .50 / .56 / .50 | step .69 / .75 / .75, sum .75 / .81 / .81 |

- **해석**
  - 정책 궤적 probe는 어느 계열에서도 길이 1보다 일관되게 낫지 않다.
  - 보상 probe는 실제 롤아웃 창(학습 분포 안, 기록된 행동)으로 질의하면 커리큘럼 모델 2개에서 길이 1부터 AUROC 1.00이다. E36의 v2 gate 상태(.62/.38)와 다르다. 질의 집합이 결과를 좌우한다.
  - 하지만 처음부터 학습한 E1 모델에서는 .25–.75로 재현되지 않는다. 커리큘럼 계보의 방문 분포(팀별 구역 습관)가 예측기에 남은 것일 수 있다(가설, E34와 같은 혼동).
  - 궤적은 보상 쪽에서도 길이 1보다 나아지지 않는다.
  - E1-F의 보상 궤적 값(.69–.81)은 예측기 R²가 0 근처라 해석하지 않는다.
- **한계**
  - seed 1, 쌍 6개다.
  - 보상 부분은 5개 모델뿐이다.
  - E1 carry_memory 창은 6개라 평가하기 어렵다.
  - source 정책이 Stage04 하나이고, Stage04는 평가 대상이기도 하다(그 모델에 유리할 수 있다).

## E38. probe 창 생성 정책 교체(source swap) + 같은 GRU의 길이 1·12 비교 — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/trajectory_probe_20261019/source_swap.py`, `results_swap/`(results.json, summary.md), `run_swap.log`. agent 학습은 없고 가중치 hash는 불변이다.
- **설계(D58)**
  - 평가 모델은 E1 10개로 고정했다(커리큘럼 Stage01–04, sp E1 IPPO/PGCT, 처음부터 학습한 E1 IPPO/PGCT/Oracle/All 6.4M).
  - 창을 만드는 정책만 바꿨다. Stage04(커리큘럼)와 처음부터 학습한 All 6.4M(4명 모두 배송)이다. 각자 자기 E1 환경에서 60×500 롤아웃하고, 길이 12 창을 agent 균등으로 뽑았다.
  - 창 수(기억/운반/무작위): Stage04 8/64/64, 처음부터 All 53/64/64.
  - 두 source 모델은 점수에서 뺐다. 표의 "8개"는 어느 쪽 source도 아닌 모델이다.
  - 정책 지표는 len1, traj_all/late/last/last2다.
  - 보상은 agent별 GRU 하나를 len1(프레임 단독)과 traj(hidden 유지), last(마지막 step)로 질의했다. MLP len1도 함께 봤다.
  - 예측기 R²: 커리큘럼 .80–.93, 처음부터 학습 .33–.55.
- **결과(AUROC 평균)**

| | 창 생성 | 운반 창 | 무작위 창 |
|---|---|---|---|
| 정책 len1, 커리큘럼 5개 | Stage04 / 처음부터 | .65 / .55 | .70 / .75 |
| 정책 len1, 처음부터 3개 | Stage04 / 처음부터 | .46 / .42 | .38 / .42 |
| 보상 MLP len1, 커리큘럼 5개 | Stage04 / 처음부터 | **1.00 / .97** | .95 / .93 |
| 보상 MLP len1, 처음부터 3개 | Stage04 / 처음부터 | **.79 / .71** | .71 / .62 |
| 보상 GRU len1 → traj → last, 커리큘럼 운반 | Stage04 / 처음부터 | .90→.90→1.00 / .85→.95→1.00 | |
| 보상 GRU len1 → traj → last, 처음부터 운반 | Stage04 / 처음부터 | .50→.67→.71 / .50→.54→.71 | |

- **해석**
  - **source 편향 우려는 대체로 해소된다.** 보상 probe의 높은 AUROC는 창을 처음부터 학습한 정책이 만들어도 유지된다(커리큘럼 .93–.97).
  - **처음부터 학습한 계보에서도 보상 probe는 우연 이상이다.** 운반 창 MLP len1이 .71–.79로, 같은 모델의 정책 probe(.42–.46)보다 높다. 이 계보의 IPPO는 배송을 배운 agent(A1, A2)가 서로 다른 팀이라, "실력 묶음"이면 AUROC가 낮아져야 한다. 그런데 반대로 나왔다.
  - **같은 GRU 안에서는 궤적과 마지막 step이 len1보다 낫다**(처음부터 운반 .50→.71). 다만 단일 프레임 성능은 MLP len1이 GRU len1보다 높아, 최선의 len1(MLP) 대비 순이득은 없다.
  - 정책 궤적 probe는 여전히 이득이 없다. 처음부터 계보에서 .33–.54다.
- **한계**
  - seed 1, 모델당 쌍 6개다. 처음부터 계보는 3개 모델이고, 같은 초기화(f78ec390…)다.
  - 실제 롤아웃 창에서 팀별 실제 보상 차이(counterfactual)는 계산하지 않았다. 그래서 예측기 차이가 "보상 함수 차이"인지 "agent별 방문 상태 차이"인지 분리하지 못했다.
  - 기억 창은 Stage04 source에서 8개뿐이다.

## E39. 보상 probe의 counterfactual 분리(Q_diff/Q_same)와 예측기-실제 보상 상관 — 2026-10-19 Claude Code, 읽기 전용

- **코드·산출물**: `output/trajectory_probe_20261019/counterfactual_split.py`, `results_cf/`(results.json, summary.md), `run_cf.log`. agent 학습은 없고 가중치 hash는 불변이다.
- **방법**
  - E38과 같은 창을 썼다(source Stage04·처음부터 All, seed 8,300,001, 60×500).
  - query(창의 각 step: source agent의 관측과 기록된 행동)마다 source episode를 reset seed와 기록된 joint action으로 재생했다.
  - 그 step에서 base env를 deepcopy하고, source agent의 팀 id만 빨강/파랑으로 바꿔 같은 joint action을 실행해 그 agent의 개인 보상 r_red, r_blue를 얻었다.
  - 재생 검증: 원래 팀의 보상이 기록된 보상과 3,804개 query 모두 일치했다(불일치 0).
  - 평가는 E38의 10개 E1 모델, agent별 MLP 길이 1 예측기(자기 롤아웃 20×500)다.
- **Q_diff 비율(실제 보상이 팀마다 다른 query)**
  - 운반 창: 604/768(Stage04 창), 392/768(처음부터 창).
  - 무작위 창: 480/768, 276/768.
  - 기억 창: 37/96, 243/636.
  - 평균 |r_red − r_blue|는 .002–.051로, 주로 shaping(요청 선반 진척·pickup) 차이다.
  - 인위적 gate 상태의 2%(E36)와 달리, 실제 상태에서는 팀별 보상 차이가 흔하다.
- **결과(AUROC 평균, 두 source 모델 제외)**

| 계열 | 창 생성 | 창 | AUROC 전체 / Q_diff / Q_same | 상관 자기 팀 / 다른 팀(Q_diff) |
|---|---|---|---|---|
| 커리큘럼 5 | Stage04 | 운반 | 1.00 / 1.00 / .95 | .77 / .02 |
| 커리큘럼 5 | 처음부터 | 운반 | .97 / 1.00 / .95 | .33 / .06 |
| 처음부터 3 | Stage04 | 운반 | .79 / .79 / .67 | .54 / −.03 |
| 처음부터 3 | 처음부터 | 운반 | .71 / .79 / .58 | .46 / .02 |
| 처음부터 3 | 처음부터 | 무작위 | .62 / .67 / .50 | .52 / .10 |
| 8개 전체 | 두 source | 기억 창 | .41–.55 | 약 0 |

- **해석**
  - **가설 A(보상 함수 차이)를 지지한다.** Q_diff에서 각 예측기는 자기 팀의 실제 보상과 상관이 .33–.77이고, 다른 팀 보상과는 −.03–.10(운반 창)이다. 두 계보 모두 같다.
  - 팀 판별 신호는 Q_diff에서 더 강하다. 처음부터 계보에서 Q_diff는 .67–.79, Q_same은 .50–.67이다.
  - 커리큘럼 계보에서는 Q_same에서도 .88–.95라, 보상 차이 외의 추가 신호(방문 분포 등, 가설 B)도 함께 있다.
  - 기억 창은 팀 보상 차이 자체가 작아(.002) 신호가 없다.
- **한계**
  - seed 1, 모델당 쌍 6개다. 처음부터 계보는 3개이고 초기화가 같다.
  - Q_diff와 Q_same은 query 수와 상태 종류가 달라 직접 비교에 주의해야 한다.
  - 팀 보상 차이는 주로 작은 shaping 항이다(설계된 shaping에 의존할 수 있다).
  - E1만 시험했다.

## E26 추가 — E1-F main 5개 외부 종료 (2026-10-19 13:07–13:08 KST, Claude Code 확인)

- **관측**
  - 5개 job(canonical IPPO/PGCT/Oracle/All, v2 PGCT)이 13:07:17–13:08:09 사이에 동시에 기록을 멈췄다. 마지막 metrics는 1.67–1.70M step이다.
  - err.log에 Python traceback이 없다. W&B debug-internal.log도 종료 처리 없이 13:08:05–13:08:09에 끊겼다. 프로세스는 남아 있지 않았고 재부팅도 없었다(마지막 부팅 10-15).
- **원인**
  - Windows 이벤트 로그 13:08:12에 Claude 데스크톱 앱 패키지 업데이트가 있었다. Claude 서비스가 중지·재시작됐고 AppX 배포 기록이 남았다(Service Control Manager 7040/7045, CoworkVMService, AppXDeploymentServer 2562).
  - 학습은 Claude Code 세션이 실행했으므로(E26, `run_e1f.ps1 -Track`) 앱 프로세스 트리의 하위였다. 앱이 업데이트로 재시작되면서 함께 종료된 것으로 판단한다. 코드 오류나 메모리 부족 흔적은 없다.
- **남은 것**
  - 각 job의 마지막 checkpoint는 `checkpoint_01500.pt`(1,536,000 step)다. actor·critic·optimizer·graph 상태가 들어 있다. 이후 약 0.15M step은 checkpoint가 없다.
  - final.pt는 없다. job_status.json은 "running" 그대로 두었다.
  - 각 plan 폴더에 `STOPPED_EXTERNALLY.txt`를 추가했다.
- **재시작은 사용자 승인 대기다.** 선택지:
  - (a) 같은 설정으로 처음부터 다시 실행한다.
  - (b) checkpoint_01500에서 이어 학습한다. 현재 `--init-checkpoint`는 optimizer·graph를 초기화하므로, 완전한 이어 하기에는 새 wrapper가 필요하다. 난수 상태도 저장되지 않아 끊기지 않은 run과 같지 않다.
  - 어느 쪽이든 Claude 앱과 독립된 터미널이나 작업 스케줄러로 실행해야 같은 일을 막을 수 있다.

## E40. E1-F 대체 후보 환경 3개 정의와 smoke — 2026-10-19 Claude Code, 구현·smoke만

- **사용자 요청**: 팀 보상을 포함해 후보 환경 2–3개(E1-F + 팀 보상, 8 agent/2팀, 맵 확대 + 센서 축소)를 정의하고 smoke까지만 하라. 학습은 하지 않는다.
- **코드**
  - 새 파일 `latent_team/env_candidates.py`.
  - `train_from_scratch.py`: `--env-variant` 후보, 후보별 trainer 설정, n-agent 모니터 적용.
  - `exp_b_oracle_consensus.py`, `random_gated.py`: 인원수 일반화.
  - 테스트 1개 추가.
  - trainer·환경·`examples/` 파일은 수정하지 않았다.
- **후보**(모두 목표 양 모서리, 팀 보상, 짝수 seed 좌우 교체. 좌우 교체는 seed 간 강건성 시험일 뿐이고 방향과 팀을 분리하지 않는다. D63 정정)
  - c1: E1 11×10, 4명.
  - c2: 11×16, 8명, 요청/팀 4.
  - c3: 20×16, 4명, 센서·통신 3.
- **smoke 과정에서 발견해 고친 것**
  - trainer가 `cfg.sensor_range`/`cfg.agent_count`로 환경 인자를 덮어써서, 처음에는 C2가 4명으로 돌고(20261019 폴더) C3는 bank 크기 오류가 났다.
  - 8명에서는 `SpatialMonitor`의 4×4 고정 때문에 실패했다(20261019b). 실행 중에만 n-agent 모니터로 바꿔 해결했다.
- **최종 smoke**(64 step, 각 complete)
  - c1·c3: `runs/smoke_candidates_20261019b/`.
  - c2: `runs/smoke_candidates_20261019c/`.
  - 확인 항목: agent 수, 센서·통신, 요청/팀, 목표 좌표, 팀 보상, bank 크기(c3 657차원), allocation 크기.
  - c2 Oracle은 같은 팀 16 arc / 다른 팀 0이다.
  - c3 Oracle은 마지막 update에 같은 팀 이웃이 없어 0 arc였다. 국소성이 실제로 생긴다.
  - Random은 이월이 동작했다.
- **검증**: pytest 189 passed.
- **학습은 하지 않았다.** 다음은 후보별 IPPO 2M pilot이고(승인 필요), "학습 시작 여부"만으로 고른다(D62).

## E41. 후보 A/B/C 재정의, 통신 범위 수정, 연결률·보상 상관 기준선 — 2026-10-19 Claude Code, 구현·smoke만

- **사용자 요청**: 후보를 A/B/C로 재정의하고, wrapper 이웃 후보를 통신 범위 기준으로 고치고, 연결률 진단과 보상 상관 기준선을 추가해 smoke까지 하라. 학습은 하지 않는다.
- **후보**(`env_candidates.py`, 모두 팀 보상, 목표 양 모서리, 짝수 seed 좌우 교체 = 강건성 시험)
  - A `a_e1f4_team`: 11×10, 4명, 센서·통신 5.
  - B `b_e1f6_comm3`: 11×10, 6명(3+3), 센서 5·통신 3, 요청/팀 3.
  - C `c_e1f6_wide_comm3`: 11×16(shelf_columns 5), 6명, 통신 3. shelf 열은 홀수만 가능해 13×12는 만들 수 없다.
  - 기존 c1/c2/c3는 기록용으로 남겼다.
- **wrapper 수정**: `matched_transfer`의 물리 후보와 검사가 `cfg.sensor_range` 대신 환경의 `communication_range`를 쓴다(`communication_range()`). 두 값이 같은 기존 실행에서는 동작이 같다.
  - smoke 검증: A·B·C의 All에서 trainer allocation 지지 집합이 wrapper 통신 후보와 모든 update에서 일치했다. Random arc는 모두 통신 범위 안이었다.
- **연결률 진단**(`latent_team/connectivity_check.py`, trainer와 같은 env 구성, 5×500)

| 환경 | 정책 | 이웃 수/agent | 고립 비율 | 완전 연결 | 팀원 가용 | 같은 팀/다른 팀 쌍 범위 안 |
|---|---|---|---|---|---|---|
| E1-F(기존, 통신 5) | 무작위 | 1.49 | .10 | .07 | .52 | .52/.48 |
| E1-F | IPPO 1.54M | 1.57 | .14 | .06 | .53 | .53/.52 |
| A | 무작위 | 1.49 | .10 | .07 | .52 | .52/.48 |
| B | 무작위 | 1.36 | .23 | .00 | .44 | .27/.28 |
| C | 무작위 | .97 | .37 | .00 | .32 | .18/.21 |
| c3(옛 후보, 20×16, 3) | 무작위 | .27 | .74 | .00 | .10 | .10/.08 |

  - E1-F 학습 기록에서도 update 끝 물리 후보 arc는 평균 6.4/12이고, 완전 그래프는 6.1%다.
  - **D62의 "E1-F는 물리 제약이 사실상 없다"는 틀렸다.** E1-F도 쌍의 약 절반만 범위 안이다.
  - B는 고립 23%, 팀원 가용 44%로 "대부분 연결"도 "대부분 고립"도 아니다.
- **보상 상관 기준선**(`latent_team/reward_correlation_baseline.py`): 보상 시계열 상관, 큰 보상 사건(|r|≥0.5) 상관, 사건 동시 발생(Jaccard)으로 팀을 가르는 기준선이다.
  - 6/8명 checkpoint를 읽도록 자체 loader를 두었다.
  - smoke 결과: B smoke 모델(학습 전)은 사건이 없어 정의되지 않는다. Stage04(개인 보상, 4.0M)는 세 지표 모두 AUROC .12다. 개인 보상에서는 보상 동시성이 팀 신호가 아니다. 팀 보상 학습 모델에서는 아직 측정하지 못했다.
- **smoke**: `runs/smoke_candidates_abc_20261019/`(A·B·C × IPPO/PGCT/Oracle/All/Random, B v2 PGCT), 모두 complete다.
- **pilot 실행기 갱신**: `run_candidate_pilot.ps1` 기본값을 B·A × seed 1·2(4 job)로 바꿨다. 계획 생성 확인용 `runs/plancheck2_*`가 있다.
- **검증**: pytest 191 passed(+2).
- **학습은 하지 않았다.**

## E42. 보상 상관 기준선 AUROC 0.12(Stage04) 재확인 — 2026-10-19 Claude Code, 읽기 전용

- **계기**: 사용자가 붙여넣은 외부 검토(T084). "AUROC가 0.5보다 크게 낮으면 점수 방향을 뒤집으면 약 0.88이므로, 같은 팀 상관이 더 낮은 구조적 신호일 수 있다. 실제 점수 분포를 보라."
- **방법**: `latent_team/reward_correlation_baseline.py`의 `rollouts`·`pair_scores`·`score`를 그대로 import해(scratchpad 스크립트), Stage04 PGCT final(개인 보상, 4.0M, `runs/pgct09700_physical_mixing_20260929_155338_079437/stage04_4ag_physical_home0.500/final.pt`)을 확률 정책으로 rollout했다. reset seed 7,800,001부터, 3 episode(E41 smoke와 같음)와 20 episode, 500 step, 사건 임계 0.5. 가중치 hash 불변(스크립트 assert). 학습 없음.
- **산출물**: `output/reward_corr_check_20261019/stage04_pair_scores.json`(쌍별 점수, episode별 cooccur AUROC).
- **결과**
  - 3 episode: E41 재현(세 지표 모두 0.125).
  - 20 episode(10,000 step, 큰 보상 사건 agent별 365/336/545/576): corr_all AUROC 0.0, corr_events 0.0, cooccur 0.125.
  - 쌍별 corr_all: 1–3(같은 팀) −0.0145, 2–4(같은 팀) −0.0067, 다른 팀 −0.0028~+0.0147. 모든 쌍 |r| ≤ 0.015로, 10,000 step 상관의 표준오차(약 0.01)와 같은 크기다.
  - episode별 cooccur AUROC는 0.0~1.0으로 흔들리고 평균 0.44다.
- **해석**: 집계 순서는 일관되게 "같은 팀이 약간 낮음"이지만, 크기가 잡음 수준이고 episode별로는 우연 수준이다. 방향을 뒤집어 쓸 수 있는 신호라고 볼 근거는 없다. 맞는 기술은 "개인 보상에서는 쌍별 보상 동시성이 거의 0"이다. E41의 "AUROC .12 → 보상 동시성이 팀 신호가 아니다"는 결론은 같지만 근거를 AUROC 대신 상관 크기로 바꿔야 한다. 같은 팀이 약간 음수인 것은 팀원이 같은 요청 선반을 두고 경쟁하기 때문일 수 있다(가설, 미검증).
- **한계**: 모델 1개, seed 1, 쌍 6개(같은 팀 2). episode 단위 bootstrap 같은 통계 검정은 하지 않았다.
- **pilot 상태(같은 시각, 읽기 전용)**: 19:00 KST에 A s1/s2 약 84k, B s1/s2 약 56k step. 1시간 기준 A 약 23, B 약 16 step/s다. 평가 배송은 0이다.

## E43. 후보 환경 IPPO pilot 결과(A·B × seed 1·2, 2M) — 2026-10-21 Claude Code 분석, 실행은 사용자

- **실행**: 사용자가 10-19 18:00 KST에 앱 밖 PowerShell에서 `run_candidate_pilot.ps1 -Track`으로 시작했다. 4개 job 모두 complete(2,000,896 step)다. 폴더는 `runs/pilot_20261019_180044_{a_e1f4_team,b_e1f6_comm3}`다.
  - 설정: IPPO, 무작위 초기화, 팀 보상(env `team_reward_mode=team`; manifest의 `team_reward: individual`은 명령줄 기본값이 남은 표기), 목표 양 모서리(seed 2는 좌우 반대), lr 2e-4, entropy .05.
  - 속도: A 약 23.7, B 약 16.3 step/s.
- **분석**: `output/env_candidates_20261019/pilot_analysis.json`(scratchpad 스크립트, 읽기 전용). 평가는 trainer 기본 20×500 deterministic, 학습 지표는 구간 평균이다.

| run | 평가 배송 첫 >0 | 지속 배송(3회 연속 ≥1) | 최근 5회 평가 배송 | 배송 agent | 학습 배송/ep (1.5–2M) |
|---|---|---|---|---|---|
| A s1 | 없음 | 없음 | 0 | 0명 | 0.00 |
| A s2 | 1.89M | 없음 | 0, 0, .65, .6, .7 | A2 1명 | 0.20 |
| B s1 | 0.97M | 1.13M | 2.25–4.2 | A3 1명 | 2.05 |
| B s2 | 없음 | 없음 | 0 | 0명 | 0.02 |
| (비교) E1-F IPPO 개인 보상 | 1.48M | 없음(1.70M 중단) | 0.05–0.85 | A3 1명 | 0.64 |
| (비교) E1 IPPO 개인 보상 | 1.69M | 없음(2M까지) | 0–0.95 | A1·A2 | 0.38 |

- **공통 패턴(모든 run, 시점은 E44에서 정정: 정체는 0.1–0.2M에 시작)**: 0–0.5M에는 무작위 pickup이 있다(pickup 성공률 .31–.65). 0.5–1.5M에는 pickup을 거의 하지 않는다(요청 선반 pickup .002–.05/ep, pickup 성공률 .02–.10, 에피소드 끝 운반 .02–.15). 그 뒤 한 agent가 다시 발견하면 그 agent만 배운다.
- **해석**
  - 팀 보상(A)이 개인 보상(E1-F)보다 빠르지 않다. A s1은 2M까지 배송이 없었다.
  - B는 seed에 따라 1.13M 시작과 무배송으로 갈렸다. seed 2개로는 A와 B를 가를 수 없다.
  - 모든 성공 run에서 한 agent만 배운다. 팀 보상으로 팀원도 보상을 받지만, 행동은 따라오지 않는다.
  - 가설: 초반 무작위 pickup이 shaping 감점(비요청 pickup −.02, 다른 팀 요청 −.05, 조기 내려놓기 −.02 등)을 받아 정책이 pickup 자체를 피하게 되고, 이것이 긴 정체 구간과 "발견 운"을 만든다.
- **한계**: seed 2개, 2M까지만 봤다. 가설은 감점 항목별 분해로 검증하지 않았다.

## E44. pickup 회피 정체의 원인 분해 — 2026-10-21 Claude Code, 읽기 전용

- **코드·산출물**: `output/pickup_stall_20261021/`(pickup_stall.py, results.json, action_values.py, action_values.json). 학습은 하지 않았다. checkpoint 가중치 hash는 불변이다.
- **로그에는 감점 항목별 값이 없다.** shaping 합계와 toggle 통계만 있다. 그래서 (1) 로그의 toggle 시계열과 (2) 고정 정책·무작위 정책 롤아웃의 항목별 보상 귀속(pickup 보상 함수 래핑 + 이벤트 × 계수 + 잔차)으로 분해했다.
- **(1) 시점: 정체는 0.5M이 아니라 0.1–0.2M에 시작한다.** episode당 TOGGLE 수는 다음과 같다(0–0.1M → 0.1–0.2M → 0.4–0.5M). 요청 선반 pickup도 0.24–1.1/ep에서 0.0–0.02/ep로 떨어진다.
  - A s1: 17.6 → 0.86 → 0.13
  - A s2: 18.1 → 1.4 → 0.12
  - B s1: 31.1 → 9.0 → 0.52
  - B s2: 37.9 → 2.9 → 0.36
  - E1-F: 14.6 → 0.88 → 0.10
  - E1: 15.7 → 0.99 → 0.13
  - 고정 정책 롤아웃에서 0.25M checkpoint의 pickup 구간은 agent-episode당 0.08–0.29개다. 무작위 정책은 8.8–10.1개다.
- **(2) 무작위 정책(초기 탐색) 아래 행동별 즉시 보상**(A 환경, 20×500, 4 agent)
  - FORWARD −.0002, LEFT/RIGHT −.0022, NOOP −.0053, **TOGGLE −.0183**. TOGGLE이 가장 손해인 행동이다.
  - TOGGLE 결과 비율: 요청되지 않은 선반 pickup 45%(−.02), 내려놓기 48%(−.02, 조기 내려놓기 감점), 다른 팀 요청 pickup 4%(−.05), 자기 팀 요청 pickup 3.6%(평균 +.064; 같은 선반을 다시 집으면 0).
  - pickup 이후 손에서 놓을 때까지의 보상 합(carry 구간):
    - 요청되지 않은 선반: −.074(15 step).
    - 다른 팀 요청 선반: −.082.
    - **자기 팀 요청 선반: +.002**. +.15 보너스가 이후 감점으로 거의 상쇄되고, 무작위 이동으로는 배송에 도달하지 못한다.
    - episode 끝까지 들고 다닌 구간: −0.80 ~ −1.14(약 270–430 step).
  - E1-F(개인 보상) 환경도 같은 값이다. 첫 배송 전에는 팀 보상과 개인 보상의 차이가 보상에 나타나지 않는다. 그래서 A가 E1-F보다 빠르지 않은 것과 일치한다.
- **해석(인과 검증 전)**
  - 초기 탐색에서 TOGGLE의 기대 즉시 보상이 음수이고, pickup 이후 구간도 자기 팀 요청 선반조차 0 근처이거나 음수다. 그래서 PPO가 0.1–0.2M 안에 TOGGLE을 거의 버린다.
  - 그 뒤로는 요청 선반을 집는 경험 자체가 거의 생기지 않는다. 정체가 길어지고, 엔트로피로 남은 드문 TOGGLE이 우연히 요청 선반에 맞은 agent만 배운다. 이것이 "한 agent만 배움"과 "발견 운"의 기전이다.
  - 주된 감점은 요청되지 않은 pickup(−.02)과 조기 내려놓기(−.02)다. 둘이 TOGGLE 결과의 93%에 걸린다.
- **한계**
  - 관찰과 기대값 계산일 뿐이다. 감점을 바꾼 pilot으로 인과를 확인해야 한다.
  - 무작위 정책 기준이다. 실제 초기 정책도 거의 균등해(초기 entropy .91–.92) 근사로 본다.
  - 롤아웃은 run당 6 episode다.

## E45. pickup 감점 제거 pilot 준비(A·B _nopen, IPPO 1M × seed 2) — 2026-10-21 Claude Code, 구현·smoke만

- **사용자 요청**: `unrequested_shelf_pickup_penalty`와 `premature_drop_penalty`를 0으로 두는 후보 환경 옵션을 추가하고, IPPO 1M × seed 2 pilot을 smoke까지 준비하라. 실행 명령도 정리하라. 학습은 하지 않는다.
- **코드**
  - `env_candidates.py`에 `a_e1f4_team_nopen`, `b_e1f6_comm3_nopen`을 추가했다(기존 A·B와 두 감점만 0).
  - `run_candidate_pilot.ps1`의 후보 목록에 두 변형을 넣었다.
  - 테스트에 감점 값 확인을 추가했다. pytest 191 passed.
  - 환경·trainer 파일은 수정하지 않았다.
- **검증**
  - smoke `runs/smoke_nopen_20261021/`(두 변형 × seed 1·2, IPPO, 64 step)가 모두 complete다. checkpoint의 env에서 두 감점이 0이고, 다른 팀 요청 감점 .05와 요청 pickup 보상 .15는 그대로다.
  - 본 실행 계획(`runs/plancheck_nopen_*`, 학습 없음)을 기존 pilot(`runs/pilot_20261019_180044_*`)과 비교했다. config 차이는 이름·저장 경로·W&B 표기·`total_timesteps`(1M 대 2M), 환경 차이는 두 감점(.02 → 0)뿐이다.
  - trainer에 학습률 일정이 없으므로(E12 추가), 기존 2M pilot의 첫 1M과 같은 step에서 바로 비교할 수 있다.
  - 초기 가중치 hash는 smoke(CPU 설정)와 본 실행이 원래 다르다. 기존 pilot A s1의 초기 hash가 E1·E1-F seed 1과 같은 f78ec390…이었으므로, 본 실행도 기존 pilot과 같은 초기화일 것으로 예상한다. 실행 후 `initial_weights_sha256.txt`로 확인한다.
- **실행 명령**(Claude 앱 밖 PowerShell, 프로젝트 폴더에서):
  `powershell -ExecutionPolicy Bypass -File .\latent_team\run_candidate_pilot.ps1 -Candidates b_e1f6_comm3_nopen,a_e1f4_team_nopen -TotalTimesteps 1000000 -Track`
  - 4개 job이다. B만 돌리려면 `-Candidates b_e1f6_comm3_nopen`.
- **판정 지표(사전 등록)**: 기존 pilot의 같은 seed·같은 step과 비교한다.
  1. 0.1M 구간별 TOGGLE/ep(0.2M 이후 유지 여부)
  2. 요청 선반 pickup/ep
  3. 첫 평가 배송과 지속 배송 시점
  4. 배송 agent 수
  5. 다른 팀 선반을 옮기는 부작용(다른 팀 요청 pickup, 잘못된 반납)
- **학습은 하지 않았다.**

## E46. pickup 감점 제거 pilot 결과(A·B _nopen, IPPO 1M × seed 1·2) — 2026-10-22 Claude Code 분석, 실행은 사용자

- 실행: 사용자가 Claude 앱 밖 PowerShell에서 2026-10-21 13:36 KST 시작(`runs/pilot_20261021_133652_{a_e1f4_team,b_e1f6_comm3}_nopen/`, job 4개 모두 complete, 1,000,448 step). W&B project `rware-from-scratch`.
- 초기 가중치 sha256: 4개 모두 기존 pilot(`runs/pilot_20261019_180044_*`) 같은 seed와 일치.
- 비교: 기존 pilot(E43)의 같은 seed 첫 1M. 지표는 T089/E45에서 미리 정한 다섯 가지. 평가는 결정적 정책, 20 episode, 500 step.
- 결과(0.1M 구간 평균, 학습 로그는 확률적 정책):
  1. TOGGLE/episode: 기존은 0.1–0.2M부터 0.03–9로 붕괴. nopen은 1M 내내 A 100–230, B 180–310 유지. P(TOGGLE | pickup 가능) 기존 0.000 → nopen 0.11–0.28.
  2. 요청 선반 pickup/episode: 기존 0.2M 이후 0–0.5 → nopen A 1.4–3.8, B 3.9–18.2.
  3. 평가 배송: 네 run 모두 1M 안에 평가 배송 ≥1에 도달하지 않음(기존도 없음). B s1 nopen 마지막 평가 5회 평균 0.37(기존 0.05), 0.87M부터 0.55. 학습 중 배송/episode B s1 nopen 0.42→1.32(마지막 0.1M), 기존 0.23. A nopen은 0.12–0.30에서 정체.
  4. 배송하는 agent: 학습 중 0.9–1M 최다 agent 비율 A nopen 0.40/0.40, B s1 0.59, B s2 0.32(기존은 배송이 있으면 1.0). 평가에서는 B s1 nopen agent 6이 76%.
  5. 부작용: 요청되지 않은 선반 pickup/episode A 52–113, B 88–151(기존 0.03–3). 잘못된 반납 A 0.01–0.19, B 0.12–0.99(기존 0).
- 판정: 1·2 통과, 3 실패, 4 학습 중에는 통과에 가까우나 평가에서는 B s1이 여전히 한 agent 위주, 5 실패.
- 해석: 두 감점이 초기 pickup 붕괴의 원인이라는 E44의 기대값 분석은 인과적으로 확인됨. 그러나 감점이 없으면 아무 선반이나 집고 내려놓는 비용이 0이라 TOGGLE 남발이 생기고, 1M 안에 결정적 정책의 배송으로는 이어지지 않음.
- 한계: seed 2개, 1M. 학습 로그 per-agent 배송 수는 합계와 단위가 달라 비율로만 해석. 평가 per-agent 수는 20 episode 합이라 20으로 나눔.
- 근거: `output/nopen_pilot_20261022/compare.py`, `results.json`. 재현: `python output/nopen_pilot_20261022/compare.py`(RWARE_MARL env).

## E47. pickup 감점 크기 sweep(무작위 정책 즉시 보상) — 2026-10-22 Claude Code, 읽기 전용

- 요청: 감점 0, .005, .01, .015에서 무작위 정책의 행동별 즉시 보상 계산, 학습 없음. 비교용으로 .02(기존)도 포함.
- 방법: E44 `action_values.py`의 `run`을 그대로 쓰고 `unrequested_shelf_pickup_penalty`와 `premature_drop_penalty`만 p로 바꿈. 유효 행동 균등 무작위 정책, 20 episode × 500 step, seed 8100001부터. 무작위 정책은 보상에 영향받지 않으므로 모든 p가 같은 궤적을 본다. 환경: 기존 pilot A s1·B s1 checkpoint_00250의 env kwargs.
- 결과(평균 즉시 보상): FORWARD −0.0002, LEFT/RIGHT −0.002, NOOP −0.005(A −0.0053, B −0.0050)는 p와 무관. TOGGLE은 p=0 +0.0002, .005 −0.0044/−0.0043, .01 −0.0090/−0.0088, .015 −0.0137/−0.0132, .02 −0.0183/−0.0177(A/B). 대략 TOGGLE ≈ 0.0002 − 0.93p.
- TOGGLE이 가장 손해인 행동이 아니게 되는 경계: NOOP과 교차하는 p ≈ .0056(B)–.0059(A). 격자에서는 p ≤ .005.
- 해석: p=0에서는 TOGGLE이 FORWARD와 같은 최상위가 되어 E46의 TOGGLE 남발과 맞는다. p=.005는 TOGGLE이 최하위는 아니지만 여전히 평균보다 낮아, PPO가 TOGGLE을 줄이는 압력이 .02의 약 1/4로 남는다. 즉 남발 억제와 정체 회피 사이의 중간값 후보다.
- 한계: 즉시 보상만 본 것이고 이후 step의 결과(운반 구간 합계)는 참고용이다. "최하위가 아님" 기준은 사전에 정한 단순 기준이며 학습 결과를 보장하지 않는다. .005는 교차점 바로 아래라 여유가 작다.
- 근거: `output/penalty_sweep_20261022/penalty_sweep.py`, `results.json`.

## E48. 감점 .005 pilot 준비(B _pen005, IPPO 1M × seed 2) — 2026-10-22 Claude Code, 구현·smoke만

- 요청: p=.005로 후보 B 변형 추가, IPPO 1M × seed 2 pilot을 smoke test까지만, 실행 명령 정리, 학습 없음.
- 구현: `latent_team/env_candidates.py`에 `b_e1f6_comm3_pen005`(`SMALL_PICKUP_PENALTY`: `unrequested_shelf_pickup_penalty`=`premature_drop_penalty`=.005)를 추가했다. `run_candidate_pilot.ps1`의 ValidateSet과 `tests/test_latent_team.py`의 후보 계획 테스트(감점 값 .02/0/.005 확인)도 확장했다. 기존 후보와 기본값은 그대로다.
- 검증:
  - 테스트 191개 중 190개 통과. `tests/test_env.py::test_inactivity_1`이 한 번 실패했지만, 단독으로 3회 다시 돌리면 모두 통과했다. `rware/` 변경은 없다(무작위성에 따른 흔들림으로 판단).
  - smoke `runs/smoke_pen005_20261022_b_e1f6_comm3_pen005/`: seed 1·2 모두 complete. checkpoint의 두 감점 .005, 다른 팀 선반 감점 .05, 6 agent, 통신 3, 팀 보상 확인.
  - 계획 확인 `runs/plancheck_pen005_b_e1f6_comm3_pen005/`(학습 없음): 기존 B pilot과는 두 감점(.02→.005)과 학습 길이(2M→1M)만 다르다. B nopen과는 두 감점(0→.005)만 다르다.
- 알려진 표시 문제: W&B tag와 `experiment.json`에 `individual`로 남는다(CLI 기본값). 실제 env는 `team_reward_mode=team`이다.
- 학습은 하지 않았다.

## E49. 보상 결합 probe 대 정책 probe 팀 AUROC(B nopen) — 2026-10-22 Claude Code, 읽기 전용

- 요청(T100 직전): B nopen 1M checkpoint 롤아웃으로 보상 결합 probe와 정책 probe의 팀 AUROC 비교, 학습 없음.
- 모델: `runs/pilot_20261021_133652_b_e1f6_comm3_nopen/b_e1f6_comm3_nopen_ippo_s{1,2}/{checkpoint_00750,final}.pt` (0.77M, 1.00M). 팀 보상, 6 agent(3+3), goal (0,10)/(9,10). 가중치 hash 전후 동일. agent 학습 없음. 보상 모델은 평균 테이블이라 경사 학습도 없음.
- 롤아웃: 자기 학습 환경, 확률적 masked 정책, 80 episode × 500 step, reset seed 9,100,001부터.
- 신호
  - 보상 결합 probe: 지도에서 정한 공통 사건 "goal 칸 g에서 배송". agent i가 시야(Chebyshev ≤ 5) 안에서 남의 그 사건을 본 step의 자기 보상 평균 − 기저 보상. 자기 배송·반납 step은 제외. 쌍 거리 = 둘 다 3회 이상 본 probe의 |차| 평균. 팀 필드(team_id, rewarded_agents)는 읽지 않음.
  - 비교용: 같은 probe의 전역 관측판(상한), 직접 결합 증거(j의 사건 → i의 보상, D48 유보), 정책 probe(PGCT gate 규칙: zero hidden, unmasked, mean JS; run의 canonical 48개, E1-F v2 gate 48개, 롤아웃 방문 상태 256개).
- 결과(AUROC, 쌍 15개)
  - 보상 결합 probe: 1.00 / 1.00 / 1.00(13쌍) / 1.00. episode bootstrap 200회 95% 구간 1.00–1.00. 응답은 거의 정확히 1(자기 팀 goal) 또는 0.
  - 직접 결합: 1.00 / 1.00 / 0.97 / 1.00(10–12쌍). 전역판은 s1 0.77M에서 0.83(carrier와 보상받는 팀이 다른 배송 때문).
  - 정책 canonical .78 / .67 / .74 / .59. v2 gate .56 / .76 / .76 / .41. 방문 상태 .78 / .44 / .74 / .56. (순서: s1 0.77M, s1 1M, s2 0.77M, s2 1M)
  - 관측 수: agent당 시야 안 배송 0.77M 1–12회, 1M 3–37회(80 episode). 반납 0–7회.
- 해석: 팀 보상이 즉시·무잡음으로 팀 전원에 들어가고 goal 칸이 팀에 1:1이라, 이 신호는 거의 팀의 정의다. 식별 가능성 조건 (나)의 기준 사례로는 의미가 있지만, 방법으로는 자명하다는 비판을 받기 쉽다. 교환값이 사실상 목표 정보라 Du et al.과의 차이가 좁다. 사건을 보기 전에는 신호가 없다.
- 한계: 환경 하나(B nopen), seed 2개, checkpoint 2개, 쌍 15개(AUROC 단위 1/54). 확률적 정책 롤아웃. 시점별 식별 속도는 재지 않았다.
- 근거: `output/reward_coupling_20261022/reward_coupling.py`, `results.json`, `stdout.txt`.

## E50. 보상 결합 probe의 온라인 식별 시점(B nopen) — 2026-10-22 Claude Code, 읽기 전용

- 요청(T102): 식별 시점 측정. 학습 없음.
- 방법: `output/reward_coupling_20261022/identification_time.py`. B nopen s1·s2, 무작위 정책(유효 행동 균등, 학습 시작 대용)과 checkpoint 0.26/0.51/0.77/1.00M(확률적 정책). reset seed 9,200,001부터 80 episode를 순서대로 진행하며 온라인 갱신. 쌍 판정은 통신 범위(Chebyshev ≤ 3) 안에서 만났고 둘 다 같은 goal 배송 사건을 K번 이상 봤을 때만 한다(|응답 차| < 0.5 → 같은 팀). 판정 전에는 전이 없음. 가중치 hash 불변.
- 결과(15쌍 모두 맞힌 env step; K=1 / K=3)
  - s1: 무작위 15,283 / 4만 안에 미달(10/15). 0.26M 36,472 / 미달(11/15). 0.51M 12,944 / 32,785. 0.77M 19,551 / 35,624. 1M 11,034 / 22,200.
  - s2: 무작위 미달(10/15) / 미달(6/15). 0.26M 10,987 / 19,289. 0.51M 6,106 / 33,749. 0.77M 9,922 / 미달(13/15). 1M 20,501 / 미달(14/15).
  - 잘못된 판정: 20개 조건 × K 2개 중 s2 1M K=1에서 1쌍(30,623 step)뿐. K=3은 0.
  - 사건 수(80 episode): 무작위 20–21, checkpoint 29–144.
- 해석: K=1이면 대개 1–4만 step(1M 학습의 1–4%)에 팀을 다 찾는다. 무작위 정책도 가끔 배송해 s1에서는 1.5만 step에 끝났다. E49 답변의 "초기 발견에는 쓸 수 없다"는 과하게 비관적이었다.
- 한계: nopen은 TOGGLE 남발로 우연한 배송이 잦다. 감점 환경(.02, pen005)에서는 사건이 더 드물 수 있다. 무작위 정책은 실제 0 step 정책과 다르다. 쌍 15개, seed 2개.
- 근거: `identification_time.json`, `identification_stdout.txt`.

## E51. 2×2 새 칸 3개, 대기열 launcher, 감시·판정 스크립트 준비 — 2026-10-22 Claude Code, 구현·smoke만

- 요청(T106): (a) 감시·판정·보고 스크립트, (b) 대기열 launcher, (c) 같은 기반의 새 칸 3개. smoke와 설정 대조까지, 학습 없음.
- 구현
  - `latent_team/env_candidates.py`: `b6_corner_indiv_pen005`, `b6_adjacent_team_pen005`, `b6_adjacent_indiv_pen005`와 `LAYOUT`(goal 배치·보상)을 추가했다. `env_overrides`가 LAYOUT을 읽고, 목록에 없는 후보는 기존대로 모서리·팀이다. 인접 goal은 (width/2−1, 아래), (width/2, 아래)이고 짝수 seed에서 바뀐다.
  - `latent_team/run_queue.ps1`: 후보별 계획을 만들고, `train_from_scratch.py --run-job` 프로세스 수(외부 실행 포함)가 MaxConcurrent 미만일 때만 다음 job을 띄운다. 출력 폴더가 이미 있는 job은 건너뛴다.
  - `latent_team/pilot_verdict.py`: job이 모두 complete가 될 때까지 기다렸다가 다섯 기준을 판정하고 `output/pilot_verdicts/<plan>/`에 쓴다. 학습은 시작하지 않는다. 4번은 마지막 0.1M 배송 20건 미만이면 보류.
  - `run_candidate_pilot.ps1` ValidateSet, `tests/test_latent_team.py`(보상·goal 기대값), README 갱신.
- 검증
  - 테스트 `tests/test_latent_team.py` 20개 통과.
  - smoke `runs/smoke_queue_20261022_*` 6개 complete. checkpoint 확인: 보상 individual/team, goal (4,10)/(5,10) 또는 (0,10)/(9,10)(짝수 seed 반대), 6 agent, 통신 3, 감점 .005.
  - 대기열: 최대 2개 + 외부 2개 실행 중이면 45초 동안 새 job을 띄우지 않음(`smoke_queue_limit_20261022`, 계획만 남김). 같은 이름 재실행은 계획 단계에서 중단(기존 폴더 보호).
  - 설정 대조 `runs/plancheck_2x2_*`(학습 없음): B pen005 대비 config 차이 없음. env 차이는 `team_reward_mode`와(또는) `team_goal_positions`뿐이다.
  - 판정 스크립트: nopen에 `--once` → overall fail(3번 실패), E46과 일치. 진행 중인 pen005에 `--once`로 시험한 0.2M 시점 보고서는 의미 없음(밤에 덮어써질 예정).
- 학습은 하지 않았다.

## E52. pen005(모서리·팀)와 모서리·개인 칸 1M 분석 — 2026-10-23 Claude Code, 읽기 전용

- 대상: `runs/pilot_20261022_112521_b_e1f6_comm3_pen005/`(사용자 실행, 10-23 03:00 complete), `runs/queue_20261022_140114_b6_corner_indiv_pen005/`(대기열, 10-23 07:20–07:28 complete). 초기 가중치는 기존 B pilot과 같음(E48 확인).
- 판정(`latent_team/pilot_verdict.py`): 두 칸 모두 overall fail, 4번(배송 분산)만 실패. 모서리·팀은 1·2·3·5 통과. 모서리·개인은 5번 skipped.
- 학습(1M, 0.9–1M 구간): 학습 배송/episode 팀 1.17/1.01, 개인 1.16/1.35(기존 .02 B 0.23/0.005). 평가 마지막 5회 팀 0.81/0.38, 개인 0.99/0.83. 개인 s1 평가 onset 0.87M. 최다 agent 비율 팀 0.91/0.82, 개인 0.99/0.97.
- 독주 패턴: 0.5–0.6M까지 여섯 agent가 고르게 배송(최다 0.2–0.5), 0.6M 무렵 한 agent가 치고 나감. 독주 agent의 팀만 배송, 다른 팀 거의 0. 비독주 agent는 팀 보상에서는 pickup은 하지만 배송 실패(예: 팀 s2 agent 1 pickup 14.6/ep), 개인 보상에서는 pickup 자체가 거의 없음(≤ 1.7/ep). 잘못된 반납이 마지막 구간 0.3–1.2/ep로 증가.
- 팀 식별(학습 없음, `output/pen005_analysis_20261023/probes_pen005.py`, E49·E50 스크립트 재사용):
  - 보상 결합 probe AUROC: 팀 1.00(네 checkpoint). 개인 0.83/0.38(s1), 0.62/0.88(s2).
  - 정책 probe AUROC: 팀 0.46–0.74, 개인 0.43–0.69.
  - 온라인 식별(K=1): 팀은 대부분 12,122–36,458 step에 15쌍 정답, 오판정 0(s1 1M은 4만 step에 14/15). 개인은 15쌍 중 5–9쌍 오판정(반응이 모두 0 근처 → 거리 작음 → '같은 팀').
- 해석: pickup 정체는 해결, 남은 병목은 한 agent 독주. 정책 probe 실패는 비독주 agent의 정책이 goal 지향이 아니어서 조건 (가)의 반박으로 보기 어렵다. 보상 결합은 개인 보상에서 판정 보류 규칙이 없으면 오판정을 낸다.
- 한계: seed 2개, 1M, 칸 2개. 인접 칸 2개는 진행 중.
- 근거: `output/pen005_analysis_20261023/results.json`, `results_corner_indiv.json`, `stdout*.txt`, `output/pilot_verdicts/`.
- 메인 1단계 계획 확인 `runs/plancheck_main_corner_team_b_e1f6_comm3_pen005/`(12 job, 학습 없음).

## E53. 독주 원인 진단, 배운 agent 간 정책 probe, 절대 배송 수 — 2026-10-23 Claude Code, 읽기 전용

- 요청(T118 후속): D76의 학습 없는 진단 세 가지.
- 방법: `output/pen005_analysis_20261023/dominance_diag.py`. 대상은 모서리·팀(pen005) s1·s2, 모서리·개인 s1·s2의 1M checkpoint. 가중치 hash 불변.
  1. 제거 평가: 20 episode × 500 step, reset seed 9,300,001부터, 결정적(trainer 평가와 같은 `net.act(deterministic=True)`)·확률적. 조건은 정상 / 독주 agent 항상 NOOP / reset 직후 독주 agent를 위쪽 모서리 칸으로 옮긴 뒤 NOOP. 위치 이동은 평가용 사본에서 `_recalc_grid`로만 했고, 환경 코드 변경은 없다. 배송은 carrier별로 셌다.
  2. 배운 agent 간 정책 probe: 각 run의 독주 agent 4명(goal (9,10)×3, (0,10)×1). v2 gate bank와 네 run의 canonical bank, PGCT gate 규칙.
  3. 학습 로그의 0.1M 구간별 agent 배송 수 = 학습 배송/episode × 200 × agent 비율.
- 결과
  1. 정상일 때 독주 agent는 0.85–1.55/ep, 나머지는 0–0.15. 독주 agent를 NOOP으로 두든 옮기든 나머지는 0–0.3/ep(결정적 평가는 모두 0)이다. NOOP과 이동의 결과가 같아 통로 차단 효과도 없다. → 경쟁·차단이 아니라 나머지 agent가 배우지 못한 것이다.
  2. 같은 goal 쌍과 다른 goal 쌍의 평균 JS: v2 0.095 대 0.091, canonical 0.110–0.111 대 0.119–0.121. AUROC 0.56. 가장 가까운 쌍은 goal이 반대인 모서리·팀 s1과 모서리·개인 s1이었다(v2 JS 0.047, argmax 불일치 2%). 24개 agent 전체 평균 JS는 같은 run 0.149, 같은 seed 0.146, 다른 seed 0.141로, seed 계보 구조가 없다.
  3. 0–0.1M에는 agent마다 구간당 9–25건을 배송한다. 이후 독주 agent 외에는 늘지 않거나 줄어든다(예: 개인 s1은 0.6M 이후 0–9건). 팀 s2의 agent 1·2·4는 0.9M에도 11–14건이지만 결정적 평가에서는 0이다.
- 해석: Oracle-team 실험의 전제(학습 신호 문제)가 맞다. 배운 정책끼리도 goal로 갈리지 않는다. 가설: 팀 기준 'my goal' 채널과 GRU 기억 때문에 goal 위치가 가중치에 남지 않고, h=0·goal 비가시 probe에는 팀 차이가 나타나지 않는다. 이 가설이 맞다면 정책 probe의 실패는 학습 부족이 아니라 구조적 문제다.
- 한계: 배운 agent 4명(쌍 6개), 20 episode. 0.26M 비교는 라벨 오류로 계산되지 않았다.
- 근거: `output/pen005_analysis_20261023/dominance_diag.json`, `dominance_stdout.txt`.

## E54. 원리상 팀을 가르는 gate probe 13개로 정책 probe 재측정, goal 방향 축 통계 — 2026-10-23 Claude Code, 읽기 전용

- 요청(T126): 최적 행동이 goal 쪽에 따라 다른 gate probe 13개만으로 정책 probe AUROC를 기존 checkpoint에서 다시 잰다. 학습 없음.
- probe 선택: E1-F v2 gate bank 48개 중, 지도 BFS(운반 중 선반 칸 진입 불가)로 계산한 왼쪽 goal(0,10)·오른쪽 goal(9,10)의 최단 경로 첫 행동 집합이 겹치지 않는 13개 [1,5,6,12,15,16,23,25,27,37,41,42,47]. 인접 goal에서는 3개뿐.
- 대상: 모서리 goal checkpoint 28개. pen005 팀·개인(s1·s2 × 0.51/0.77/1M), nopen(s1·s2 × 0.77/1M), B .02(s1·s2 × 1M/2M), A .02(s1·s2 × 1M/2M), E1-F 1.54M(ippo/pgct/oracle/all). 가중치 hash 불변.
- 결과 1 (JS): 평균 팀 AUROC 48개 0.512, 13개 0.493, 35개 0.524. 13개가 나은 경우 6개, 나쁜 경우 12개. 개선 없음.
- 결과 2 (goal 방향 축): agent마다 s_i = 13개 평균 [p(왼쪽 최적 행동) − p(오른쪽 최적 행동)]. s가 자기 goal 쪽을 가리키는 정도(AUROC): pen005 팀 0.89, pen005 개인 0.96, nopen 0.89, B .02 0.92, A 0.44, E1-F 0.56. 쌍 AUROC |s_i−s_j|: 0.70/0.73/0.72/0.57/0.53/0.41(같은 13개의 JS는 0.53/0.52/0.54/0.47/0.31/0.56). 부호 규칙(부호가 같으면 팀원) 정확도: 0.76/0.83/0.83/0.57/0.58/0.33. 28개 중 축 > JS13이 21개.
- 해석: 6-agent B 칸의 정책은 goal 방향을 담고 있다. 이는 seed 1·2에서 goal 쪽이 바뀌어도 유지된다. 그러나 다섯 행동 전체의 JS에서는 무관한 차이에 묻힌다. 보상 결합이 실패하는 개인 보상 칸에서 처음으로 쓸 만한 팀 신호(부호 규칙 0.83)가 나왔다. 4-agent run에서는 나타나지 않았다(배송이 거의 없어 덜 학습되었을 가능성, 미검증).
- 한계: checkpoint 28개, 6-agent는 seed 2개. goal 후보 칸(지도)을 안다고 가정하고, 두 팀·두 goal 기준이다.
- 근거: `output/probe_discriminative_20261023/discriminative_probes.py`, `goal_axis_gate.py`, `results.json`, `goal_axis_results.json`, `stdout.txt`, `goal_axis_stdout.txt`.

## E55. 팀당 요청 수(3/6/9)에 따른 초기 탐색 신호(무작위 정책) — 2026-10-23 Claude Code, 읽기 전용

- 계기(T130): 사용자 의견 '개인 보상의 팀원 경쟁은 요청 선반이 너무 적어서 생긴 문제 아닌가, 물량이 많은 편이 낫다'.
- 방법: `output/request_sweep_20261023/request_sweep.py`. E44 `action_values.run`을 재사용했다. 모서리·개인 칸(6 agent, 감점 .005, corner_indiv s1 checkpoint_00250의 env kwargs)에서 `request_queue_size_per_team`(3/6/9)과 `request_queue_size`(2배)만 바꿨다. 유효 행동 균등 무작위 정책, 20 episode × 500 step.
- 결과: agent당 자기 팀 요청 선반 pickup/episode 1.02 / 1.96 / 2.73. TOGGLE 중 자기 팀 요청 비율 5.0% / 9.7% / 13.5%. TOGGLE 평균 즉시 보상 −0.0043 / −0.0020 / −0.0013(NOOP −0.0053).
- 해석: 요청을 늘리면 초기 정책이 보상받는 pickup을 2–3배 자주 경험하고 팀원 간 경쟁도 줄어든다. E53의 '나머지가 배울 기회가 부족했다'와도 맞는다. 팀당 6개(전체 12, agent 6)는 RWARE의 easy 난이도 비율(요청 = 2 × agent 수)과 같다.
- 한계: 무작위 정책 계산이며, 학습에서 독주가 풀리는지는 미검증. 쉬워지면 IPPO도 잘 배워 전이의 여지가 줄 수 있다.

## 새 실험 기록 양식

ID / 실제 실행 날짜·작성 도구 / 상태(계획·smoke·진행·완료·실패) / 초기 모델 / 환경·training seed·예산 / 평가 seed·정책·horizon / 결과 / 비교 기준·한계 / 원본 경로 / 재현 명령.

## 2026-10-23 Codex — A+C 구현 검증 (본 실험 아님)

- 상태: 코드 검증 완료, PPO/신경망 학습 미실행. 초기 학습 모델/checkpoint 없음. runner 통합 검사는 PPO update를 stub으로 바꾼다.
- 조건: tests/test_reward_probe_ac.py 11개 + tests/test_latent_team.py 20개. 실제 환경 검사는 기존 reference manifest의 환경 설정을 읽고 seed 1, 개인 보상, goal (0,10)/(9,10), semantic 관측으로 제어된 1-step 사건을 구성했다. training seed 실험이나 평가 episode 성능 수치가 아니다.
- 결과: 31 passed. 가상 local NOOP/turn/가능한 FORWARD의 public post-observation이 실제 환경과 일치. 두 팀 운반자를 같은 물리 배송에 오프라인으로 배치하면 shaping을 제외한 배송 성분은 [[1,0],[0,1]]. 숨은 상태 사용은 이 정답 검사에만 있다.
- 합성 충분 경험에서는 같은 보상 벡터 agent만 donor로 선택하며, 미지/모호/모두 0/범위 밖은 보류. 개인 보상 벡터를 합산하지 않는지, 평가 경험을 메모리에 넣지 않는지, checkpoint와 actual_allocation 연결도 확인.
- 명령: C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe -m pytest tests/test_reward_probe_ac.py tests/test_latent_team.py -q -p no:cacheprovider --basetemp output/ac_validation_20261023_saved_state_tmp
- 근거: tests/test_reward_probe_ac.py, tests/test_latent_team.py, output/ac_validation_20261023_saved_state_tmp/, output/ac_verification_20261023/summary.json. 표준 gym Box 정밀도 및 multi-agent list reward 경고가 있었고 실패는 없다.
- 한계: 경험 조건부 평균 예측기이며 CNN 일반화는 구현하지 않았다. 실제 학습된 정책으로 수집한 배송 경험에서의 보상 예측 정확도/AUROC/전이 향상은 미검증. 기존 결과와 로그는 보존.

## E56. A+C v1 본 학습 진행 확인(모서리·개인, 좌표 마스킹) — 2026-10-26 Claude Code, 읽기 전용

- 실행: 사용자가 10-23 18:55 앱 밖 PowerShell에서 시작(PID 18880, Codex가 준 명령). `runs/ac_20261023_185556/`, `--reward-probe-ac --team-reward individual --modes ippo pgct oracle all --seeds 1 2 --timesteps 5000000 --track`. 8 job을 한 프로세스에서 순차 실행한다(job당 약 42시간, 약 34 step/s).
- 설정 확인(`experiment.json`, checkpoint `effective_env_kwargs`): `team_reward_mode=individual`, goal 팀0 (0,10)/팀1 (9,10)(s1; s2는 좌우 반대), 감점 .005, 요청 팀당 3, λ .01, γ .99, GAE λ .95. 모든 mode에서 self_x/self_y를 0으로 마스킹. PGCT는 `graph_mode=unrestricted` + A+C allocation override. 팀 구성은 agent 0·2·4 = 팀0, 1·3·5 = 팀1(`_balanced_assignments`).
- IPPO s1(complete, 5,000,192 step): 평가 배송 24.1/ep(4.75–5M 평균), agent별 0 / 4.64 / 5.38 / 5.05 / 4.20 / 4.85. 6명 중 5명이 배웠고 agent 1(팀0)은 0. 1M 시점은 2.58/ep.
- PGCT s1(running): 10-26 11:44에 2.87M, 13:22에 3.01M.
  - 2.87M까지: 이웃 쌍 비교 11,275회 중 지지 부족 보류 6,022, 거리·신호 미달 5,248, 승인 5(전부 agent 4–6 쌍, 같은 팀). 실제 전이 update 2,807개 중 3개. 신호 ≥ .2인 비교 5,253회 중 5회. 가상 질의(A) 지지 0.
  - 3.01M까지: update 2,943개 중 전이 5개, 승인 행 20개.
  - 2.75M checkpoint 예측기: agent별 지지 장면 16–165개, 보상 큰 지지 장면(|평균| ≥ .2) 1–2개, LRU 4,096칸이 가득 찬 상태.
- 비트 단위 일치(`output/ac_offline_20261026/divergence.py`): 첫 실제 전이(2,821,120 step, update 2755, 4↔6) 전 2,754 update에서 return·value loss·policy loss·entropy가 IPPO s1과 완전히 같다. value loss는 update 2755에서 처음 다르고, 평가는 2.87M에서 처음 다르다(10.9 대 12.35). 원인: peer loss가 0이면 IPPO와 같은 연산만 하고, A+C gate는 난수를 쓰지 않는다. 옛 전이 mode는 warm-up 직후부터 증류해 처음부터 달랐다(E24의 분기는 같은 mode 재실행의 GPU 비결정성).
- W&B `probe/pgct_gate_arcs`·`probe/same_team_gate`·`probe/cross_team_gate`는 trainer 자체(unrestricted) gate다. 2,826개 로그 전부에서 `update/physical_candidate_arcs`와 같다. 실제 전이는 `update/actual_transfer_arcs`(0이 아닌 로그 3개).
- 한계: seed 1만 진행. PGCT s1은 사실상 IPPO 복제라 Q1 정보가 거의 없다. 판정은 5M 이후.
- 근거: `runs/ac_20261023_185556/*/{job_status.json,metrics.jsonl,transfer.jsonl}`, `comparison.md`.

## E57. A+C gate 오프라인 진단(저장된 checkpoint, 학습 없음) — 2026-10-26 Claude Code

- 방법: `output/ac_offline_20261026/collect.py`. A+C IPPO s1 checkpoint 7개(0.256M, 0.768M, 1.536M, 2.56M, 3.584M, 4.864M, final 5.0M) × 8 episode, 확률 마스킹 정책(정책 입력 좌표 마스킹), reset seed 9,300,001부터. agent 전이 168,000개의 공용 관측(v1 `public_view`, 반경 2)·행동·자기 보상과 채점용 숨은 라벨(팀, 위치, 배송자, goal 소유)을 저장. 팀 라벨은 채점에만 썼다.
- 조건 1(관측 가능성): 두 goal 배송의 공용 관측(자기 상태 + 벽·goal 채널)이 겹치지 않는다(각 2종, 공통 0).
- 조건 2(음성 예시): 다른 팀 goal에 선반을 들고 들어간 전이가 agent마다 31–136번(agent 1은 0), 평균 보상 ≈ 0. 이전 답변의 '개인 보상에서 상대 goal에 가지 않는다'는 틀렸다.
- v1 정확 일치 예측기: 같은 장면이 3 episode 이상에서 8번 이상 반복되는 전이는 agent 경험의 5–27%. 배송 key 278개 중 표본이 충분한 17개의 보상 비율 중앙값 .45, std .45, std ≤ .10은 1개(숨은 요청 여부 때문에 같은 공용 전이에 +1/0이 섞임). 쌍별 공통 보상 사건 질의 0. 다른 agent를 지우거나 3×3으로 줄이면 지지는 62–77%로 늘지만 분산 필터가 여전히 사건을 막고, 필터를 빼면 다른 팀 승인 6건.
- 일반화 예측기(agent별 MLP 567→128→64→1, 자기 보상 MSE, episode 80% 학습): 사건 probe(배송·반납 1,068개) AUROC 1.00(같은 팀 평균 거리 .26, 다른 팀 .54), 좌우 대칭 probe .91(agent 1 제외 1.00), 무작위 경험 probe .81. 1.5M까지 경험만 쓰면 .67/.57. 팀별 응답이 반대로 나온다: 예) agent 3(팀0) 팀0 goal 배송 .89 / 팀1 goal .00 / 팀0 배송의 대칭 .00 / 팀1 배송의 대칭 .87. 거리 행렬에서 모든 agent의 가장 가까운 두 명이 팀원(같은 팀 최대 .34 < 다른 팀 최소 .39).
- 한계: seed 1 run 하나, 15쌍(같은 6·다른 9). 실제 학습 경험의 근사(checkpoint 롤아웃). 사건 probe 일부는 배송한 agent의 학습 표본이다. agent 1은 예측기가 모든 곳에서 ≈ 0이라 팀원과 가까운 것은 약한 증거다.
- 근거: `output/ac_offline_20261026/{README.md,collect.py,analyze.py,results.json,details.py,details.json,divergence.py}`, `rollouts.npz`.
- 재현: `python output/ac_offline_20261026/collect.py` → `analyze.py` → `details.py`.

## E58. Gate v2 구현·검증(본 학습 없음) — 2026-10-26 Claude Code, 사용자 요청

- 사용자 요청(T139): 예측기 일반화, 사건 probe, 문턱 재보정, A+C gate W&B 기록을 새 옵션으로 추가하고 기본값을 유지하며 smoke test까지. 학습은 사용자가 시작한다.
- 구현: `latent_team/reward_probe_ac_v2.py`, `train_from_scratch.py --ac-gate {v1,v2}`(기본 v1, v2 job 이름 `*_acv2_*`, manifest `reward_probe_ac_v2`). 설정 `ACV2Config` 기본값: 자기 전이 버퍼 20,000, 사건 버퍼 512(|r| ≥ .5), 사건 과표집 25%, MLP hidden 128, lr 1e-3, round마다 50 step × batch 256(P=5 update, warm-up 중에도 학습), 성숙 = 자기 사건 ≥ 20 + 자기 사건 평균 예측 ≥ .3, probe = 쌍 각자 최근 사건 32개 + 좌우 대칭, 거리 Σ|a−b|/Σ(|a|+|b|), EMA β .35, 3 round 이상, 문턱 .5, 대칭 gate, 별도 RNG. W&B·metrics `update/ac_v2_*`와 사후 감사 `update/ac_v2_audit_*`(env info 팀 ID, gate 입력 아님). checkpoint `reward_probe_ac_v2`. 문서 `docs/REWARD_PROBE_AC.md` v2 절.
- 검증: `tests/test_reward_probe_ac_v2.py` 7개 + 기존 v1 11개 = 18 passed, `tests/test_latent_team.py` 20 passed. 실제 trainer smoke(`--smoke-test`, 6 agent, 64 step, CPU, scratchpad 출력, W&B 없음): `update/ac_v2_*`가 `metrics.jsonl`에 기록, checkpoint 저장, 보상 사건 0이라 전부 보류, IPPO·PGCT 초기 hash(`caaef33f…`)와 eval 일치.
- 실행 중인 A+C v1 학습(PID 18880)은 시작 시 코드를 이미 읽었고 job을 같은 프로세스에서 돌리므로 영향이 없다.
- 한계: 본 학습 성능과 시점별 온라인 정확도는 미검증. 비대칭 donor 규칙은 넣지 않았다.

## E59. Gate v1 대 v2 오프라인 판정 비교(기존 모델 경험 재생, 학습 없음) — 2026-10-26 Claude Code

- 방법: `output/ac_offline_20261026/gate_eval.py`. 롤아웃 경험을 수집 순서대로 실제 `ACController`(v1)와 `ACV2Controller`(v2)에 넣고 1,000 env step마다 판정 round. 모든 쌍이 매 round 접촉한다고 가정(접촉 빈도는 별개 문제). v1 가상 질의는 전체 관측이 필요해 재생하지 않았다.
- 데이터: E57 롤아웃(A+C IPPO s1 0.26–5M, 28 round) + 옛 1M IPPO(좌표 있음, 정책 입력에 좌표 유지) `runs/queue_20261022_140114_b6_{corner,adjacent}_indiv_pen005/*_ippo_s{1,2}` checkpoint 00250·00500·00750·final × 8 episode(16 round).
- 결과(정확도 = (같은 팀 승인 + 다른 팀 거부)/결정한 쌍, 15쌍):

| 데이터 | v1 정확도 (결정 쌍) | v1 승인 같은/다른 | v2 정확도 (결정 쌍) | v2 승인 같은/다른 |
|---|---|---|---|---|
| A+C IPPO s1 0.26–5M | .63 (8) | 1 / 0 | 1.00 (10) | 4 / 0 |
| 모서리 1M s1 | .25 (8) | 0 / 0 | 전부 보류 | 0 / 0 |
| 모서리 1M s2 | .50 (12) | 0 / 0 | 전부 보류 | 0 / 0 |
| 인접 1M s1 | .60 (15) | 0 / 0 | 전부 보류 | 0 / 0 |
| 인접 1M s2 | .56 (9) | 0 / 0 | 전부 보류 | 0 / 0 |

- v2(A+C 5M): 같은 팀 거리 .14–.18, 다른 팀 .58–.70. 보류 5쌍은 모두 agent 1이 낀 쌍. 첫 승인은 2.56M checkpoint 경험 구간(round 15), 다른 팀 승인은 모든 round에서 0. 좌우 대칭을 끈 v2도 같은 결과.
- v1: 1M 모델에서 거리 AUROC .30–.42. v1의 '거부'에는 문턱 미달 외에 신호·round 부족도 섞여 있다.
- 한계: 5M run은 seed 1 하나, 실제 경험의 약 1/180 재생, 접촉 가정. 1M 모델은 배송 agent가 0–2명이라 v2 성숙 agent가 없다. 인접 배치는 배운 모델이 없어 조건 (가)의 음성 사례를 검증하지 못했다.
- 근거: `output/ac_offline_20261026/{gate_eval.py,gate_eval.json,gate_eval_stdout.txt}`, `rollouts_*_1M_s*.npz`.

## E60. 보상 성분별 식별 가능성 감사와 실제 경험량 초기 식별 곡선(학습 없음) — 2026-10-26 Claude Code, 사용자 요청(T142)

- 요청: D87의 최소 실험 1번(성분 감사)과 2번(실제 경험량에 맞춘 초기 식별 곡선, V2 절대 차이 대 대조 부호). 학습 금지.
- 방법: `output/ac_early_id_20261026/stream_eval.py`. checkpoint 0.256/0.512/0.768/1.0M가 각 구간을 대표하도록 구간 길이만큼(구간당 약 25만 step) 확률 마스킹 정책 재생, 데이터당 약 100만 env step. 보상을 배송·반납·pickup·goal 진척·나머지로 분해(환경 내부 사건과 진척 식을 재현, 감사 전용). 같은 경험을 실제 `ACV2Controller`(기본 설정)에 학습 박자(5 update마다 round, warm-up 0.1M, 통신 범위 안 물리 이웃)로 입력. 4 round마다 15쌍을 V2 절대 차이와 대조 c = r̂(q) − r̂(대칭 q)의 상관·부호로 채점. 감사는 `audit.py`(성분 × 특징 묶음 × 구간 조건부 평균, 10개 3+3 분할 중 실제 팀 순위, agent 쌍 AUROC).
- 데이터: A+C IPPO s1(좌표 마스킹), 옛 1M IPPO 모서리·개인 s1·s2, 인접·개인 s1(정책 입력에 좌표 있음, gate 입력에는 없음).
- 결과 1(≤0.26M): goal 진척(방향 특징)은 모서리 3개에서 분할 순위 1/10, agent AUROC .87–1.00. 인접에서는 순위 6/10, .35. 배송(방향+지형)은 네 데이터 모두 team_gap 1.00, 순위 1/10(인접도 goal 근처 지형으로 두 goal이 구분된다). pickup은 순위 1–10 혼재, agent AUROC .30–.50으로 관측상 식별 불가. 나머지(감점 등)도 대체로 불가.
- 결과 2: V2 gate 첫 같은 팀 승인 acs1 0.27M, 모서리 s1 0.32M, s2 0.26M, 인접 0.86M. 다른 팀 승인은 네 데이터 모두 0(195–200 round). AUROC ≥ .9 첫 시점: V2 절대 차이 0.16 / 0.27 / 0.02 / 0.78M, 대조 상관 0.04 / 0.20 / 0.10 / 0.06M. 대조 corr>0 판정 정확도(0.1/0.2/0.3M): acs1 .93/1.00/1.00, 모서리 s1 .40/.73/1.00, s2 1.00/1.00/1.00, 인접 .67/.93/.80.
- 해석: E59의 'v2 첫 승인 2.56M'은 재생량(1/180) 때문이었다. 실제 경험량에서는 모서리 0.26–0.32M. 초기 팀 신호는 goal 진척 방향과 goal 근처 배송이고, pickup은 아니다(D87 예상 확인). 대조는 순위를 더 일찍 맞추지만, 0으로 바로 판정하면 0.1–0.2M에 오판정이 있어 보류 규칙이 필요하다. 인접 배치는 방향 신호에서는 음성 사례지만 배송 지형으로 결국 식별된다.
- 한계: 각 구간을 구간 끝 정책으로 대신했다(재생 초기 배송은 학습 로그보다 적음). run마다 seed 하나, 15쌍. 같은 팀의 행동 유사성 혼입을 완전히 배제하지 못했다. AUROC는 모든 쌍 접촉 가정. 대조 방식은 분석 스크립트에만 있다.
- 근거: `output/ac_early_id_20261026/{README.md,*_curve.json,audit.json,summary.json,*_stdout.txt}`, `*_audit.npz`.
- 실행 중이던 A+C v1 학습(PID 18880)은 건드리지 않았다. 처음 띄운 분석 프로세스 4개는 메모리 사용이 커질 것으로 보여 Claude가 직접 멈추고 배열을 미리 할당하도록 고쳐 다시 실행했다(학습 프로세스와 무관).

## E61. Oracle s1이 IPPO보다 빠르지 않은 이유 — 읽기 전용 분석(학습 없음) — 2026-10-29 Claude Code, 사용자 요청(T147)

- 대상: `runs/ac_20261023_185556`(A+C v1, 모서리·개인 보상, 좌표 마스킹, seed 1). IPPO·Oracle·PGCT는 complete(5M). All s1은 10:40 기준 1.6M에서 진행 중이다. checkpoint 정책 추론만 했다(실제 상태 분석은 mode·시점마다 3 episode).
- 팀 평가 배송/episode at 1M / 2M / 3M / 4.5M:
  - IPPO 2.4 / 8.1 / 11.6 / 21.7
  - Oracle 0.4 / 7.4 / 10.6 / 21.5
  - PGCT v1 2.4 / 8.1 / 11.7 / 20.5
  - All(1M) 1.5
  - Oracle은 0.5–1M에서만 뚜렷이 느리다.
- 증류 효과(같은 팀 (V_i − V_j)²):
  - probe 위: IPPO .060 / .870 / 1.205, Oracle .0015 / .0060 / .0048 (1M / 2M / 4.75M). 증류는 probe를 150–250배 맞췄다.
  - 실제 방문 상태: IPPO .080 / .674 / 1.194, Oracle .023 / .115 / 1.223. 후반에는 효과가 이어지지 않는다.
- 팀원 V의 쓸모(IPPO 실제 상태, 내 할인 return 대비 MSE): 2M에 자기 critic .18, 팀원 critic .74. 4.75M에 자기 critic .83, 팀원 critic 1.79. 같은 팀 critic 차이(2M .67)가 다른 팀(.44)보다 작지 않다.
- 1M의 probe 위 |V|: IPPO .140, Oracle .064. 첫 발견자의 V가 평균되어 눌렸다는 가설과 맞지만, 결과일 수도 있다.
- 수준 도달 뒤 +5 배송까지 걸린 step(M):
  - 5→10: IPPO 1.13, Oracle 1.43, PGCT 1.13
  - 10→15: IPPO .92, Oracle .92, PGCT 1.18
  - 15→20: IPPO .87, Oracle .46, PGCT .56
  - PGCT v1은 2.82M까지 IPPO와 같았으므로, IPPO와 PGCT의 차이가 seed 하나에서 생기는 혼돈 수준이다.
- 정책 JS(probe, IPPO, 같은 팀 / 다른 팀): 0.25M .038 / .056, 1M .185 / .189, 4.75M .217 / .248. KL(π‖U)은 1M에 배송한 agent(.79·1.34·.60)와 못 한 agent(.50·.48·.56)를 잘 가르지 못한다.
- `probe/cross_team_gate`(PGCT 평균 .277)는 다른 팀 물리 이웃 비율(4.98/18)과 같다. A+C PGCT의 trainer gate가 `unrestricted`이기 때문이다. 실제 v1 allocation은 같은 팀 48 arc, 다른 팀 0 arc.
- 한계: seed 하나. 실제 상태는 3 episode이고 mode마다 방문 분포가 다르다.
- 근거: `output/oracle_why_20261029/{README.md,oracle_why.py,oracle_vgap.py,oracle_realgap.py,oracle_js.py}`.

## E62. 전이 재설계 전제 점검: 팀원 critic의 팀 특이성과 max 상향(학습 없음) — 2026-10-29 Claude Code, 사용자 요청(T148)

- 대상: A+C IPPO s1 checkpoint 1M·2M·3M. 시점마다 3 episode를 확률 정책으로 굴리고, 받는 agent의 실제 방문 상태에서 6개 critic을 모두 recurrent로 돌렸다. 할인 return은 γ .99이고, episode 끝 100 step은 제외했다. 추론만 했다.
- 뒤처진 agent(그 rollout 배송 0) / 앞선 agent별 상향폭 mean relu(max_j V_j − V_i), 같은 팀 대 다른 팀 critic:
  - 1M: 뒤처진 .35 / .34, 앞선 .03 / .15
  - 2M: 뒤처진 1.19 / .72, 앞선 .13 / .56
  - 3M: 뒤처진 1.07 / 1.12, 앞선 .18 / .25
- 운반 중이고 자기 goal이 시야 밖인 상태에서 각 critic과 내 return의 상관(자기 / 같은 팀 / 다른 팀):
  - 1M: .56 / .65 / .56
  - 2M: .59 / .66 / .53
  - 3M: .39 / .48 / .50
  - 전체 상태: 자기 .49–.56, 같은 팀 .34–.41, 다른 팀 .27–.31
- 내 return 대비 편향(자기 V / max(자기, 같은 팀) / max(자기, 다른 팀)):
  - 1M: +.03 / +.25 / +.29
  - 2M: +.27 / +.76 / +.89
  - 3M: +.13 / +.45 / +.52
- 해석:
  - max 결합은 뒤처진 agent만 크게 올리고 앞선 agent는 거의 바꾸지 않는다. 방향이 저절로 정해진다.
  - 다른 팀 critic은 앞선 agent에게도 큰 상향을 준다. gate가 필요하다.
  - 팀 특이성은 작다. 관측 ch4·ch12가 팀 기준이기 때문이다. 과대추정 위험이 있다.
- 한계: seed 하나, 시점마다 3 episode, IPPO 정책만 봤다. 상향의 방향이 맞는지(행동 선택 개선)는 측정하지 않았다.
- 근거: `output/oracle_why_20261029/{transfer_premise.py,transfer_premise_stdout.txt}`.

## E63. max 결합 advantage(D91) 반사실 점검: 팀원 정책을 내 자리에서 실행(학습 없음) — 2026-10-29 Claude Code, 사용자 요청(T149)

- 데이터: A+C IPPO s1(좌표 마스킹) 1M·2M, 옛 모서리 1M IPPO s1·s2 final(좌표 있음). 모두 개인 보상이다. V2 성숙은 E60 replay에서 해당 시점의 판정을 쓴다. 성숙한 agent만 받는 쪽·주는 쪽으로 넣었다. 뒤처진 agent는 그 episode에서 배송이 0인 성숙 agent다.
- 방법:
  - env를 deepcopy해 상태를 복원했다(재현 확인).
  - 받는 agent i의 자리와 보상은 그대로 두고, i를 자기 정책 또는 팀원 j의 정책으로 실행한다. j의 actor는 i의 관측 이력으로 데운다. 나머지 agent는 자기 정책이다.
  - dMC = G(π_j) − G(π_i)를 critic 신호 dV = V_j(h_i) − V_i(h_i)와 비교한다.
  - 공통 난수, N=8, H=100. 데이터당 3 episode, τ = 80·180·280.
  - 행동 수준: 유효 행동마다 one-step A_self와 A_max를 MC Q(H=60, 자기 정책 또는 앞선 팀원 정책으로 이어 감)와 Spearman으로 비교했다.
  - 정책·critic 추론만 했다.
- 상태 수준, 뒤처진 agent이고 dV > .2인 경우. 각 항목은 n / 평균 dMC / dMC>0 / 유의 이득 / 유의 손실:
  - acs1 1M: 같은 팀 20 / +.28 / 1.00 / .40 / .00, 다른 팀 20 / +.08 / 1.00
  - acs1 2M: 같은 팀 9 / +.48 / 1.00 / .89 / .00, 다른 팀 9 / +.08
  - corner s1: 같은 팀 9 / +.58 / 1.00 / .78 / .00, 다른 팀은 전체 dMC −.01이고 유의 손실 .67
  - corner s2: 같은 팀 18 / +.28 / .67 / .50 / .28, 다른 팀 18 / +.14 / .89
- 앞선 agent:
  - 같은 팀 dV는 대체로 음수(−.15 ~ −.52)이고 실제 dMC도 음수(−.02 ~ −.31)다. corr(dV, dMC)는 +.07 ~ +.85.
  - 같은 팀 dV > .2인 경우(acs1 2M, n=14): dMC +.08, 양수 .71, 유의 손실 .14.
  - 다른 팀 dV > .2인 경우(n=11·27): dMC +.00 / −.06, 유의 손실 .09 / .22.
- 크기 보정: dV는 무한 horizon, dMC는 H=100이다(대략 0.63배가 기대값). acs1 2M에서 dV 1.21 대 dMC .48로 약 1.6배 과대이고, 나머지는 비슷하거나 과소다.
- 행동 수준(뒤처진 상태 9–39개), Spearman:
  - vs 자기 정책 이어 간 Q: A_self +.19 ~ +.43, A_max −.27 ~ +.38. A_max가 나은 데이터가 없다.
  - vs 앞선 팀원 정책 이어 간 Q: A_self −.01 ~ +.28, A_max +.18 ~ +.30. A_max가 조금 낫다.
  - Q_own과 Q_lead의 일치도 +.20 ~ +.49로 MC 잡음이 크다.
- gate, 뒤처진 agent:
  - 범위 안 성숙 팀원이 있는 step 비율: .46 / .20 / .28 / .31
  - uplift > .05 step 비율, gate 대 팀 전체: .42 대 .96, .20 대 1.00, .28 대 1.00, .10 대 .46
  - 평균 uplift, gate 대 팀 전체: .14 대 .36, .28 대 1.28, .18 대 .63, .06 대 .30
- h=0 질의:
  - donor 값 corr: h=0 .55–.86, L8 window .88–1.00
  - 같은 팀 uplift corr: h=0 .33–.81, L8 .52–1.00
  - 자기 대 donor 선택이 뒤집히는 비율: 뒤처진 agent .00–.02, 앞선 agent .00–.18
- 상관 지표(`max_advantage_check.py`, 앞선 팀원의 최선 행동에 대한 정규화 advantage 차이): own .03–.13 → max .13–.19. 다른 팀 앞선 agent의 최선 행동에 대해서도 비슷하게 오른다. 앞선 agent의 corr(A_max, A_own)은 .92–1.00.
- 해석:
  1. 뒤처진 agent에게 같은 팀 critic이 높게 나오면 대체로 실제로 더 나은 정책이 있다. 다른 팀 critic의 높은 값은 대부분 실제 이득이 작다. 정책 가치 수준의 팀 특이성은 E62 critic 상관보다 크다.
  2. "max advantage가 더 좋은 행동을 강화한다"는 미입증이다. 내 정책 기준으로는 A_self보다 낫지 않고, 팀원 정책 기준으로 조금 낫다(MAMBA 이론과 일치).
  3. 통신 범위와 성숙 제한으로 실제 기회는 팀 전체 대비 21–40%다.
  4. 질의는 h=0이 아니라 짧은 window(L8)가 필요하다.
- 한계: 훈련 seed 3개(acs1 s1, 옛 s1·s2), episode 3, τ 3, 일부 집단 n=9. MC 잡음이 크다(행동 수준). H=100 절단. v2 승인 절차(EMA·round)는 모사하지 않았고, 성숙과 범위만 썼다.
- 근거: `output/oracle_why_20261029/{counterfactual_check.py,cf_*.json,cf_*_stdout.txt,summarize_cf.py,cf_summary.json,cf_summary_stdout.txt,max_advantage_check.py,max_advantage_check_stdout.txt}`.

## E64. Max-GAE(D) 구현, smoke·동등성 검증, 1M 분기 규모 측정 — 2026-11-02 Claude Code, 사용자 요청(T154)

- 구현(기본값·기존 모드 불변):
  - `latent_team/max_gae.py`: 새 파일.
  - `latent_team/train_from_scratch.py`:
    - 모드 `oracle_max`, `oracle_maxd1`, `all_max`, `allk_max`.
    - 옵션 `--max-gae-eta`, `--max-gae-scale`, `--max-gae-lambda`, `--max-gae-cap`, `--init-checkpoint`, `--transfer-warmup`.
  - `tests/test_max_gae.py`: 11개.
  - `docs/MAX_GAE.md`.
  - trainer와 `rware/`는 수정하지 않았다.
- 계산:
  - 받는 agent의 rollout 위에서 donor critic을 recurrent로 돌린다. 시뮬레이션에서는 이것으로 critic 가중치 공유를 대신한다.
  - b = clip(max_j V_j − V_i, 0, cap), f = V_i + b, A_max = GAE(f). timeout은 donor 값으로 포착하고, done 처리는 trainer와 같다.
  - actor = A_self + ηΔ. value는 자기 return으로 학습한다.
  - donor는 rollout 중 한 번이라도 Chebyshev 3 안에서 만난 agent다. update 단위로 고정한다.
  - donor가 없으면 trainer의 `update_ippo`를 그대로 호출한다. donor가 있으면 그 안의 `compute_gae` 호출 한 곳에 advantage를 주입한다.
- 검증:
  - pytest: 새 11개 + 기존 A+C·latent_team 38개, 총 49개 통과.
  - 실제 trainer smoke(64 step): donor를 끈 `oracle_max`가 IPPO와 최종 가중치까지 비트 단위로 같다. 처음부터와 1M 분기 둘 다 확인했다.
  - 1M 분기 smoke: 모든 모드의 초기 hash가 `fc4a533b…`로 같다. oracle donor는 같은 팀뿐이다.
  - plan-only: 분기 12 job 계획을 확인했다(`--plan-only`, scratchpad).
- 규모 측정: `runs/ac_20261023_185556/..._ippo_s1/checkpoint_01000.pt`에서 1,024 step × 4 update. `oracle_max`·`all_max`, raw η=.5. 측정용 짧은 실행이고 scratchpad에 두었다.
  - 접촉: rollout 안에서 모든 쌍이 만났다(30/30). rollout 단위 접촉 규칙에서는 통신 제한이 사실상 없다.
  - 뒤처진 agent(0-index 0·1·4):
    - b > 0인 step 99%, 평균 b .30–.53.
    - raw 변화량 |A_actor − A_self| / std(A_self) = 2.6–12.
  - 앞선 agent(2·3·5): 변화량 0–.4.
  - 속도: 25.6 sps(원래 IPPO 1M 무렵 26.1, 약 2% 감소).
- 해석:
  - 방향은 의도대로다. 앞선 agent는 거의 영향을 받지 않는다.
  - 다만 raw Δ가 자기 advantage보다 몇 배 커서, PPO의 agent별 정규화 뒤에는 raw D2(η=.5)와 D1이 거의 같은 update가 된다.
  - 그래서 D2 기본값을 `relative`로 바꿨다. Δ를 std(A_self)로 다시 맞춘 뒤 η를 곱한다. 이 설정의 smoke 변화량은 .14–.33이다.
- 한계: 규모 측정은 seed 1, 4 update, 시점 하나(1M)다. 학습 효과는 측정하지 않았다.
- 근거: `latent_team/max_gae.py`, `tests/test_max_gae.py`, `docs/MAX_GAE.md`. smoke·측정 폴더는 scratchpad(`smoke_eq*`, `smoke_br*`, `scale_br`, `plancheck_br`)로, 프로젝트에는 남기지 않았다.

## E65. A+C r̂와 희소 보상: 보상 밀도, v2 사건 누적·성숙·거리 시간표(학습 로그 읽기 전용) — 2026-11-02 Claude Code, 사용자 질문(T155)

- 질문(T155): 보상이 일부 행동에만 있으면 대부분의 전이에서 r̂ = 0일 텐데, 보상 기반 식별이 가능한가.
- 대상: `runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1`(A+C v2 PGCT, 모서리·개인 보상, 좌표 마스킹, seed 1, 5M, complete). `experiment.json`, `metrics.jsonl`, `transfer.jsonl`만 읽었다. 학습·추론은 하지 않았다.
- 보상 설정(`experiment.json`):
  - 큰 사건: 배송 1.0, 반납 1.0
  - 요청 선반 집기 .15
  - 진척 shaping: 반납 .02, goal .005, 요청 선반 .002
  - 감점: 교착 .02, 막힘 .01, 정지 반복 .006, 잘못된 선반 집기 .05, 잘못된 반납 .005, 실패 전진 .005, 요청 외 집기 .005, 조기 내림 .005, noop .003, 무효 집기 .002
  - 표준 RWARE(`rware/warehouse.py`)는 배송 +1뿐이다(TWO_STAGE에서만 반납 .5).
- 보상 밀도(학습 로그 구간 평균, 0이 아닌 agent-step 비율): 0–0.1M .37, 0.1–0.25M .25, 1–2M .41, 4–5M .55. 양수 .08–.24, 대부분 감점이다.
- v2 사건(r ≥ .5, 배송·반납)은 희소하다. agent당 누적 사건(구간 끝)은 0.1M 11, 0.25M 30, 0.5M 53, 1M 252, 2M 2,439, 5M 21,945다. 전이 대비 .01%(0.1M)–.44%(5M)다.
- v2 판정 시간표(구간 평균, 문턱 .5):

| 구간 | 성숙 agent | 같은 팀 거리 | 다른 팀 거리 | 다른 팀 승인/update |
|---|---|---|---|---|
| 0–0.1M | 0 | – | – | 0 |
| 0.1–0.25M | 3.3(끝 5) | .25 | .99 | 0 |
| 0.25–0.5M | 5 | .23 | .99 | 0 |
| 0.5–1M | 5 | .19 | .99 | 0 |
| 1–2M | 5.8(끝 6) | .11 | .95 | 0 |
| 2–3M | 6 | .11 | .88 | 0 |
| 3–4M | 6 | .09 | .77 | 0 |
| 4–5M | 6 | .08 | .64 | .64 |

- 실제 전이 arc(`transfer.jsonl`): 같은 팀 5,252(0–1M 546, 1–2M 1,072, 2–3M 1,310, 3–4M 1,198, 4–5M 1,126). 다른 팀 352는 모두 4.28M 이후다(4–4.5M 104, 4.5–5M 248).
- 해석:
  - 희소한 사건으로도 agent당 수십 번 쌓이면(0.25M 무렵) 같은 팀과 다른 팀이 갈린다.
  - 성숙 전 보류 때문에 0.1–0.25M 이전에는 판정이 없다. 0.25–1M에는 한 agent가 보류 상태였다.
  - 다른 팀 거리는 학습 후반에 계속 내려와 4.28M부터 문턱 아래 쌍이 생긴다. 원인은 분석하지 않았다.
- 한계: seed 하나. 감사 거리는 그 update에서 비교한 쌍의 EMA 평균이다. 보상 밀도는 학습 rollout 기준이다.
- 근거: `output/ac_sparsity_20261102/{README.md,reward_settings.py,reward_sparsity.py,*_stdout.txt}`.

## E66. PGCT-D2 검토용 점검: Max-GAE 척도·가림의 합성 수치, 1M 무렵 agent별 보상 사건(학습 없음) — 2026-11-02 Claude Code, 사용자 요청(T163)

- 합성 점검: `latent_team/max_gae.py`의 `blended_advantages`·`max_targets`를 합성 텐서(1,024 step, γ .99, λ .95, η .5)에 적용했다. 환경·학습 없음. 스크립트는 scratchpad(`d2_review_check.py`)이고 프로젝트에 남기지 않았다.
  - 지표 mean|A_actor − A_self| / std(A_self) (괄호는 최대), raw / relative:
    - b ~ U(0, 1e-6): .0000 / .43 (1.38)
    - b ~ U(0, 1e-2): .004 / .44
    - b ~ U(0, .3): .11 / .44
    - b ~ U(0, 1): .38 (1.20) / .44 (1.41)
    - 1% step에만 b = 1e-3: .00003 (.0014) / .089 (5.03)
  - 가림: V_i .2, donor critic 1.0(실제 b .8 일정), window 8, D1. Δ[0..6] = +.46 → +.67, Δ[7..] ≈ −.13(정상 상태 −.12).
- 로그 읽기: `runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/metrics.jsonl`의 `train/agent_k_delivery_count`·`return_count` 기록 평균(배송/반납, 0-index).
  - 0.95–1.06M(108개): a0 0/0, a1 .06/0, a2 2.96/.94, a3 2.27/.50, a4 0/0, a5 1.63/.40. 같은 구간 평가(2개)에서 a0·a1·a4는 배송 0.
  - 1.4–1.5M(97개): a0 0/0, a1 1.01/.12, a2 5.62/2.12, a3 3.48/.85, a4 0/0, a5 2.92/1.06.
- 해석:
  - relative는 b 크기와 무관하게 거의 같은 비중으로 advantage를 바꾼다. raw는 b에 비례한다.
  - 가림은 매 rollout 첫 7 step에 체계적인 양의 편향을 만든다.
  - a0·a4는 분기 구간(1.0–1.5M)에서 v2 사건(r ≥ .5)이 사실상 없어 성숙하기 어렵다.
- 한계: 합성 텐서라 실제 크기는 다르다. 로그 지표의 단위(episode당/rollout당)는 확인하지 않았고 0이라는 사실만 쓴다. seed 하나.
- 근거: 위 run 경로, D98.

## E67. D98 필수 수정 1–4 검증: pytest와 실제 trainer smoke(64 step) — 2026-11-02 Claude Code, 사용자 요청(T164)

- 대상 코드: `latent_team/max_gae.py`, `latent_team/train_from_scratch.py`, `tests/test_max_gae.py` (D99).
- pytest: `tests/test_max_gae.py` 19개(새 8개)와 `test_latent_team.py`·`test_reward_probe_ac.py`·`test_reward_probe_ac_v2.py`, 모두 57개 통과.
- smoke(모서리·개인, A+C, `--smoke-test`: 32 step rollout × 2 update, scratchpad에 두고 프로젝트에는 남기지 않음):
  - `smoke_eq`: 처음부터, `ippo`·`oracle_max`, `--transfer-warmup 100`(donor 보류). 최종 가중치가 비트 단위로 같다.
  - `smoke_br`: A+C IPPO s1 `checkpoint_01000.pt` 분기, seed 1, `oracle_max`(shrink, η .5)·`oracle_maxd1`.
    - donor arc 6(같은 팀).
    - shrink에서 뒤처진 a0·a1의 η_eff .04–.08, 변화량 .41–.59. D1은 변화량 5.3–14.7.
    - 앞선 a2·a3는 변화량 0. a5는 update 2에서 η_eff .5(줄이지 않음), 변화량 .35.
    - critic EV −26.6–.27(32 step rollout이라 의미 없음).
  - `smoke_v2_135`: 권고 명령 형태(`--ac-gate v2`, `ippo`·`oracle_max`·`oracle_maxd1`·`allk_max` × seed 1·3·5). 12 job이 모두 complete이고 초기 가중치가 같다.
  - `--seeds 1 2` 분기 plan-only는 seed 2의 goal 배치 불일치로 거부되고, 폴더를 만들지 않았다.
- 한계: smoke는 학습 결과가 아니다. 분기 실험 자체는 아직 실행하지 않았다.
- 근거: D99, `docs/MAX_GAE.md`.

## E68. PGCT-D2 필수 5–8 단위·회귀·실제 trainer smoke 검증 — 2026-11-02 KST, Codex (사용자 요청 T166)

- 코드: `latent_team/reward_probe_ac_v2.py`, `latent_team/max_gae.py`, `latent_team/train_from_scratch.py`; 새 테스트 `tests/test_pgct_max.py`. trainer·환경 소스 수정 없음. D100 참조.
- 실제 수행: 새 15개 + 기존 Max-GAE/A+C/latent_team 57개 = **72 passed**(33.69초). XML `output/pgct_d2_validation_20261102/pytest_final.xml`. 첫 실행의 임시 상위 폴더 오류(17 setup errors)와 다음 실행의 테스트 fixture 오류 2개를 수정한 뒤 최종 성공했다. 학습 구현 오류로 분류하지 않는다. 첫 smoke 산출물도 보존했다.
- 실제 trainer smoke: `b6_corner_indiv_pen005`, 개인 보상, A+C 좌표 마스킹, seed 1, CPU, 32-step rollout × 2 update = 64 step/job, 총 9 job. 초기 모델 `runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt`(1,024,000 step), 새 optimizer·v2 상태. W&B off. 최종 평가는 smoke용 1 episode × 16 step이며 성능 근거가 아니다.
- 산출물 기준 폴더 `output/pgct_d2_validation_20261102/pytest_final/test_real_trainer_closed_gate_0/`:
  - `plan/`: IPPO, `pgct_max`, `oracle_mature_max`, `oracle_max` 4 job. v2가 실제 refit(version=2)하되 min_events=1000000으로 승인을 막는 테스트 조건에서 앞의 세 모드 최종 모든 가중치 tensor 비트 동등.
  - `injected/`: 테스트 monkeypatch로 정답 승인·현재 성숙을 주입한 두 새 모드. 두 모드 모두 `oracle_max`와 최종 모든 tensor 비트 동등. 실제 donor와 actor 변화 있음; cross-team donor 0; 승인 범위 12/12.
  - `filtered/`: 같은 주입에 EV floor=1e9(테스트용 값)로 두 모드의 donor 전부 차단. 둘 다 IPPO와 비트 동등.
  - `held/`: `pgct_max`, warm-up=1. update 1 donor 0, update 2 donor 존재.
- v2 refit 테스트 조건: interval=1, batch_size=16, train_steps=2. 연구용 기본값 변경이 아니다. 자연스러운 온라인 팀 승인을 만들거나 검증한 smoke가 아니다.
- 합성 테스트: 접촉 상실 후 승인 보존·대칭·복사본·현재 성숙 회귀·round/거리 경계, gate 판정의 라벨 독립, contact/EV 교집합, oracle_mature의 양쪽 보상 성숙, 잘못된 입력 거부, v2 아닌 계획 거부, 자기 복제 critic의 actor/self 차이 ≤1e-6(raw/shrink).
- 한계: 장기 학습·독립 training seed 결과 아님. D2 실제 성능·EV 임계값·보상 사건 없는 agent·E65 후반 오승인은 미결. PGCT-D2 새 본 학습 미실행, 커밋 없음. 기존 분기는 읽기만 했다.
- 자세한 명령·근거: `output/pgct_d2_validation_20261102/README.md`, `docs/MAX_GAE.md`.

## E69. 최종 PGCT 논문·발표 제작용 기존 로그 읽기와 산출물 검증 — 2026-11-02 KST, Codex (T167, 새 학습·실험 없음)

- 목적: 영어/한국어 논문과 17장 발표에 동일한 실제 근거를 넣는다. 기존 학습·평가·probe를 다시 실행하지 않았다.
- 읽은 로그:
  - `runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1/{transfer,metrics}.jsonl`: 모서리·개인·좌표 마스킹·seed 1, 최종 5,000,192 step. same/cross directed allocation 누적 5,252/352, 비중 93.71877230549608%, 첫 cross 4,280,320 step. 그림은 100k-step 구간 평균의 실제 EMA와 보상 예측기 성숙이다. 기존 접촉/전이 규칙의 발견 구성요소 진단이며, 최종 PGCT 학습 결과로 재명명하지 않았다.
  - `runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/metrics.jsonl`: 독립 PPO 기준선. deterministic 평가 20 episode × 500 step, seed 1. 추출한 평가점 97개, 마지막 4,966,400 step의 단일 평가 배송 25.65/ep. 과거 E61의 평균이나 최종 5회 평균과 같은 통계라고 설명하지 않는다.
  - `runs/maxgae_br1M_20261102_165216/experiment.json` 및 상태/마지막 로그: 문서 근거를 추출할 때 독립 PPO s1이 추가 35,840 step, 나머지 계획 상태인 스냅샷. 최종 성능 비교에는 넣지 않았다.
- 공통 초기 모델 설명: 기존 1,024,000-step 독립 PPO s1 checkpoint에서 0.5M 추가 분기, seed 1·3·5. 처음부터 독립 training seed라고 세지 않았다. 최종 PGCT, mature-oracle, count-matched 비교 결과는 대기다.
- 실제 수행: 파일 읽기, 표/CSV/과학 그래프 생성, 새 구조도 3개 제작, LaTeX/PPTX/대본 생성, PDF 컴파일·렌더링·내용 대응 확인만 했다. `E68`의 72 tests/64-step smoke는 기존 검증 인용이며 재실행하지 않았다.
- 산출물: `paper/english/main.{tex,pdf}`, `paper/korean/main.{tex,pdf}` 각 7쪽; `presentation/pgct_beamer.{tex,pdf}`와 `pgct_presentation.pptx` 각 17장; 한국어 대본. PPTX 본문·도식·표3·그래프2는 native, 수식만 8·10·12·13장에서 이미지. PowerPoint 앱 자체 검사는 하지 않았으며 Artifact Tool 렌더링과 OOXML 검사를 했다.
- 검증: 한영 공통 수식 11개, 인용 7개, label 일치; 17장 제목/대본 순서 일치; PDF 3개 컴파일 성공, overfull·누락 glyph 오류 없음; PPTX 구조/geometry finding 0. 연구 코드 SHA-256는 읽기 시작과 같음.
- 참고자료: 사용자 ICCAS PDF 전체16장 구성·스타일만 참고. 음성 M4A는 812.9793초 메타데이터만 확인, 전사/청취내용 미확인. 다른 연구 수치·식·그림을 가져오지 않았다.
- 근거·재생성: `output/pgct_authoring_20261102/{collect_evidence.py,evidence.json,quality_report.json}`, `paper/evidence_manifest.json`, `presentation/reference_analysis_ko.md`, `PGCT_DELIVERABLES_README.md`. TeX 빌드는 Tectonic 0.17.0(XeTeX 기반). 공식 목표 연도 IJCAI 양식 적용은 별도다.
- 연구 코드·보상·gate 기준·실험 실행 변경 없음. 새 학습·중단·재시작·커밋 없음. 문서 컴파일용 도구·캐시는 새 제작 폴더에만 저장했다.

## E70. Codex PGCT-D2 구현 재검증: 테스트 재실행과 자연 승인 경로 smoke(학습 없음) — 2026-11-02 Claude Code, 사용자 요청(T168)

- pytest: `test_pgct_max.py`·`test_max_gae.py`·`test_latent_team.py`·`test_reward_probe_ac.py`·`test_reward_probe_ac_v2.py` 72개 통과. 실제 trainer 테스트는 scratchpad의 긴 basetemp에서 `oracle_mature_max` job 경로가 260자를 넘어(261–262자) FileNotFoundError로 실패했고, 짧은 경로(`AppData\Local\Temp\pgt`)에서 통과했다.
- 자연 경로 smoke: `pgct_max`, A+C IPPO s1 `checkpoint_01000.pt` 분기, seed 1, `--smoke-test`(32 step × 2 update). v2 설정은 배선 확인용으로 느슨하게 줬다(interval 1, batch 16, train_steps 2, min_events 1, min_event_response −1, event_reward .1, min_rounds 1, distance_threshold 1.0, min_probes 1). 임시 폴더는 지웠다.
  - update 1: 성숙 1명(a2), 비교 0, donor 0.
  - update 2: 성숙 2명(a2·a3). rollout 접촉 10쌍, update 직전 이웃 4쌍. 비교된 1쌍은 update 직전 이웃이 아니었다. 승인 arc 2, donor arc 2(다른 팀, 문턱 1.0 때문), donor = 접촉 ∧ 승인 성립, adv_shift .088.
  - 두 update 모두 `transfer.jsonl` 실제 allocation 0.
- 해석: rollout 접촉으로 비교하는 경로가 실제로 동작한다. 보상 사건이 없는 a0·a1·a4는 느슨한 설정에서도 성숙하지 않았다.
- 한계: 설정이 연구 기본값이 아니고 2 update뿐이라 승인 정확도와 성능 근거가 아니다.

## E71. D103 rollout 시작 스냅샷 비교 검증: pytest와 실제 trainer 자연 승인 경로 — 2026-11-02 Claude Code, 사용자 요청(T169)

- pytest: `test_pgct_max.py`(새 5개)·`test_max_gae.py`·`test_latent_team.py`·`test_reward_probe_ac.py`·`test_reward_probe_ac_v2.py` 77개 통과(basetemp `AppData\Local\Temp\pgt`, 지움).
  - 단위: 스냅샷 비교 거리가 rollout 시작 상태로 직접 계산한 값과 같고(비교 버전 (1, 1)), 비교 뒤에만 버전 2로 재학습된다. 접촉 뒤 경험과 재학습 예측기로 계산하면 다른 값이 나온다. 스냅샷 뒤 재학습은 RuntimeError.
  - 계획: 플래그는 `pgct_max`·`oracle_mature_max`에만 True, `pgct`는 False. 플래그 없는 옛 `pgct_max` job은 `run_job`에서 거부.
- 실제 trainer 자연 경로(`test_real_trainer_natural_approval`): A+C IPPO s1 `checkpoint_01000.pt` 분기, seed 1, `pgct_max`, `--smoke-test`(32 step × 2 update). 배선 확인용 설정: interval 1, batch 16, train_steps 2, min_events 1, min_event_response −1, event_reward .001, min_rounds 1, distance_threshold 1.0, min_probes 1.
  - update 1: rollout 시작 시 6명 모두 미성숙, 비교 0, donor 0.
  - update 2: rollout 시작 시 6명 성숙, 접촉 10쌍 모두 비교(거리 .31–.99), donor 20 arc = 접촉 ∧ 승인.
  - final 상태: 비교 버전 모두 (1, 1), 예측기 버전 모두 2, `transfer.jsonl` allocation 0.
- 한계: 연구 기본값이 아닌 배선 확인용 설정, 2 update. 승인 정확도·지연·성능 근거가 아니다.

## E72. 논문·발표자료용 근거 재계산과 문서 검증(학습 없음) — 2026-11-02 Claude Code, 사용자 요청(T171)

- 근거 재계산: `output/pgct_docs_20261102/evidence_plots.py`가 원본 로그를 읽어 `evidence.json`, CSV 3개, 논문·발표 그래프를 만든다.
  - 발견 로그 `runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1`: 같은 팀 5,252 / 다른 팀 352 arc(93.72%), 첫 다른 팀 4,280,320 step, 100k 구간 51개. E65·Codex 값과 같다.
  - IPPO 기준선 `runs/ac_20261023_185556` seed 1·2: 마지막 평가(4.97M) 25.65 / 24.05 배송/episode(20 episode × 500 step).
- 문서 검증: 영어·한국어 논문(각 8쪽), Beamer 17장 모두 Tectonic 0.17.0으로 컴파일 오류·Overfull·누락 글자 0. PDF를 그림으로 렌더링해 전 페이지를 확인했다. `check_consistency.py`: 한영 수식·인용·라벨·참조·그림·수치 일치, Beamer·PPTX·대본 17장 제목 순서 일치. `preview_pptx.py` 근사 미리보기에서 PPTX 글자 넘침 0(PowerPoint로 직접 열어 보지는 않음).
- 참고 자료: ICCAS PDF 16장을 렌더링해 구성 확인. M4A는 머리 정보로 812.98초만 확인, 전사 못 함.
- 실행 상태(읽기 전용): v1 run `runs/ac_20261023_185556`의 `pgct_s2`는 `job_status.json`이 failed(KeyboardInterrupt, 2026-11-02 18:01:52)로 바뀌었다. 약 3.19M에서 사용자가 중단한 것으로 보인다(T170 뒤). 1단계 분기 `runs/maxgae_br1M_20261102_165216`는 진행 중이며(작업 중 확인 시 `ippo_s1` 0.12M), 부분 결과는 문서에 넣지 않았다.
- 한계: 그래프는 seed 1(발견)과 seed 2개(기준선)뿐이다. 통합 PGCT 학습 결과는 없다.

## E73. v2 gate 다른 팀 거리 하락·후반 승인 원인 분석과 분기 실험 중간 현황(학습 없음) — 2026-11-04 Claude Code, 사용자 질문(T174)

- 대상: `runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1`(모서리·개인, seed 1, 5M, v2 reward probe gate, update 끝 접촉, mirror 사용, 통신 3). 팀 {0,2,4} vs {1,3,5}.
- 방법(읽기 전용): `output/ac_drift_20261104/drift_analysis.py`가 checkpoint 1000·2000·3000·3500·4000·4250·4500·4750(1.02–4.86M)의 저장 예측기·사건 버퍼로 probe(최근 32개 + mirror)를 다시 만들고, 거리를 소유자·사건 종류(배송=FORWARD 사건, 반납=운반 1→0인 TOGGLE 사건)·원본/mirror로 나눴다. `transfer.jsonl`의 실제 allocation으로 다른 팀 arc를 셌다. `split_distance_check.py`는 공개 행동별 거리 최대값(그룹 4개 이상), `no_mirror_check.py`는 mirror 없는 거리를 같은 예측기로 계산했다. 출력은 각 `*_stdout.txt`.
- 결과:
  - 최근 32개 중 반납: 1.02M 1–6, 2.05M 6–10, 3.07M 10–12, 4.35M 13–15(agent 1·2·3·5). agent 0은 사건 64개·반납 2개, agent 4는 반납 7–10개.
  - 다른 팀 거리 (1,2)/(2,3)/(2,5): 1.02M .99/1.00/1.00, 2.05M .75/.71/.78, 3.07M .62/.63/.58, 4.10M .53/.53/.56, 4.35M .46/.48/.40, 4.86M .49/.47/.47. agent 0 pair .89–.95, agent 4 pair .60–.70. 같은 팀 최대 .15–.23.
  - 공유 비중(1−거리) (2,5): .00→.22→.42→.60(1.02/2.05/3.07/4.35M). 상위 기여는 모두 반납 probe(원본·mirror).
  - EMA (1,2)/(2,3)/(2,5): 4.10M .52/.51/.55, 4.35M .48/.50/.49, 4.61M .47/.49/.45.
  - 다른 팀 arc: (1,2) 120 첫 4.28M, (2,3) 118 첫 4.33M, (2,5) 114 첫 4.35M, 합계 352. 같은 팀 5,252(4M 이후 1,126).
  - mirror 없음: 다른 팀 거리 거의 같음(4.35M (2,5) .40→.39), 2M 이후 같은 팀 최대 .09–.15(mirror 있을 때 .15–.23).
  - 공개 행동별 최대: 같은 팀 최대 / 다른 팀 최소 = 1.02M .98/1.00, 2.05M .77/.99, 3.07M .79/.99, 3.58M 이후 .65–.71/.98–.99.
- 분기 실험 현황(2026-11-04 10:42 KST): 모두 v1 A+C IPPO s1 checkpoint_01000(1.02M) 가중치에서 0.5M(500,736 step). 1단계 `runs/maxgae_br1M_20261102_165216`(PID 28892) 8/12 완료, `ippo_s5` 59,392 step, 남은 4 job은 11-05 새벽 무렵 종료 추정. 2단계 `runs/pgctd2_br1M_20261102_180212` 6/6 완료. 결정적 평가 20 episode × 500 step 마지막 3회 평균 배송/episode: ippo 6.6/5.6, allk_max 6.5/6.4, oracle_max 5.9/6.0, oracle_maxd1 6.1/6.0(s1/s3), oracle_mature_max 6.3/6.7/6.9, pgct_max 6.0/5.7/5.9(s1/s3/s5). agent 0은 모두 0, agent 1·4는 0–0.75. per-agent 지표 `eval/agent_k_delivery_count`는 20 episode 합계(합을 20으로 나누면 `eval/deliveries`와 일치).
- 한계: seed 1 run 하나. 예측기는 checkpoint 시점 고정이며 오프라인 재계산이다. 사건 종류 분류는 행동·운반 비트 기준의 근사다. mirror 없는 거리와 행동별 최대는 gate를 다시 돌린 결과가 아니다. 분기 결과는 seed 2–3개의 중간값이다.

## E74. 행동별 최대 거리 gate 후보의 오프라인 검증(학습 없음) — 2026-11-04 Claude Code, 사용자 질문(T175)

- 스크립트: `output/ac_drift_20261104/stratified_gate_check.py`, 출력 `stratified_gate_check_stdout.txt`.
- 데이터: `runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1` checkpoint 19개(0.26–4.86M), `runs/pgctd2_br1M_20261102_180212/*_pgct_max_s{1,3,5}`의 checkpoint_00250·final(각 0.26M·0.50M, 1.02M IPPO s1 가중치에서 분기, 예측기는 분기 안에서 새로 학습). 팀은 agent 번호 % 2(`_balanced_assignments`).
- 방법: 저장 예측기·사건 버퍼로 gate와 같은 probe(각 agent 최근 32개, mirror 있음/없음)를 만들고, gate 성숙 조건(사건 20개 이상, 자기 사건 평균 예측 .3 이상)을 먼저 적용했다. 합산 거리, 행동별 최대(그룹 4개 이상), 경험 조건 m=2·4·8(두 agent 모두 그 행동의 자기 사건 m개 이상)을 비교했다.
- 결과(25개 checkpoint 전체, 같은 팀 최대 / 다른 팀 최소):
  - mirror 없음: 합산 .28 / .39, 행동별 최대 .96 / .98, m2 .96 / .98, m4 .86 / .98, **m8 .28 / .98**.
  - mirror 있음: 합산 .29 / .40, 행동별 최대 .95 / .97, m4 .80 / .97, **m8 .33 / .97**.
  - m8에서도 모든 성숙 pair가 비교 가능한 행동을 하나 이상 가졌다(비교 pair 수가 합산과 같음).
- 한계: checkpoint 스냅샷, EMA·접촉·round 미재현. seed 1 5M 하나와 같은 출발점의 짧은 분기. 홀수 seed·r_c=3·모서리·개인만.

## E75. action_max gate 구현 검증: 단위 테스트, 전체 테스트, 저장 checkpoint 대조(학습 없음) — 2026-11-04 Claude Code, 사용자 승인(T176)

- 단위 테스트 `tests/test_gate_action_max.py` 7개 통과: 가중평균 항등식과 최댓값, 경험 조건, 반납 공유 합성 예제(pooled는 다른 팀 승인, action_max는 거절·같은 팀 승인; contact_snapshot 켬/끔), 공통 행동 없음 처리, 기본값·옛 설정·잘못된 mode, 실행 계획 이름·기록·v1 거부.
- 전체 `tests/` 255개 통과(`--basetemp C:/Users/95101/AppData/Local/Temp/pgt_all`, `-p no:cacheprovider`).
- 대조 `output/ac_drift_20261104/implementation_crosscheck.py`(출력 `implementation_crosscheck_stdout.txt`): E74와 같은 25개 checkpoint의 예측기·사건 버퍼를 새 `ACV2Controller`에 복원하고 `compare()`로 성숙 pair 299개를 계산했다. 오프라인 m8 결과와 거리 차이 0, 선택 행동 동일, 성숙 판정 동일. 같은 팀 최대 / 다른 팀 최소: mirror 없음 .281 / .975, mirror 있음 .328 / .970.
- 한계: 실제 학습 중 EMA·접촉 궤적은 확인하지 않았다. E74의 데이터 범위 한계가 그대로 적용된다.

## E76. 본 실험 준비 검증: r_c=5 후보, 실제 trainer action_max, CLI smoke, 5M plan, 처리 속도(학습 결과 아님) — 2026-11-04 Claude Code, 사용자 승인(T178)

- 테스트: `tests/test_comm5_candidate.py` 2개(plan 설정 차이가 communication_range뿐, IPPO 300 step 같은 행동에서 r_c 3·5의 관측·보상 비트 동일·접촉 수만 증가), `tests/test_gate_action_max.py::test_real_trainer_action_max_on_comm5`(1M checkpoint 분기, 허용적 gate 값으로 배선만 확인: action_max 비교 발생, donor = 접촉 ∩ 승인, checkpoint에 action_max·mirror 없음·contact_snapshot·r_c 5 기록), `test_env_candidates_plan`에 새 후보 추가. 전체 `tests/` 258개 통과.
- CLI smoke(`--smoke-test`, 처음부터, `C:/Users/95101/AppData/Local/Temp/pgsmoke_main`, 프로젝트 밖): `--env-variant b6_corner_indiv_comm5_pen005 --reward-probe-ac --ac-gate v2 --ac-distance action_max --ac-config latent_team/acv2_nomirror.json`, mode pgct_max·ippo·oracle_max·allk_max·oracle_mature_max × seed 1·2·3, 15 job 모두 complete. 모두 r_c 5, seed 2 goal 좌우 반전, gate job은 action_max·mirror 없음·contact_snapshot, seed별 초기 가중치 동일(s1 `caae…`, s2 `30e4…`, s3 `8acd…`).
- 5M plan-only(임시 폴더): job당 5,000,192 step, 평가 50 update마다 20 episode, Max-GAE donor 보류 100 update. queue plan-only `runs/plancheck_main_amax_20261104_b6_corner_indiv_comm5_pen005`(계획·bank만): `-ModeMajor` 순서 pgct_max s1–3 → ippo → oracle_max → allk_max.
- 처리 속도(checkpoint 파일 시각): 단독 약 32 step/s, 2개 동시 job당 25–26, 4개 동시(10-22 queue, IPPO) job당 약 16.
- `latent_team/main_check.py`를 2단계 분기에 시험 적용: pgct_max s1·s3·s5 같은 팀 승인 arc 1,920·1,350·1,510, 다른 팀 0, 첫 같은 팀 승인 0.02–0.03M.
- 한계: smoke는 학습 결과가 아니다. 4개 동시 속도는 IPPO job 기준이며 gate job은 더 느릴 수 있다.

## E77. 난이도 후보 무작위 정책 비교, 후보 smoke, W&B 영상 확인(학습 결과 아님) — 2026-11-04 Claude Code, 사용자 요청(T180)

- 무작위 정책(`output/difficulty_20261104/difficulty_random.py`, 유효 행동 균등, 20 episode × 500 step, seed 1 배치, reset seed 8,200,001부터, 출력 `stdout.txt`·`results.json`): agent·episode당 자기 팀 요청 pickup / 보상 사건(r≥.5) / 선반–goal 평균 거리. 기준 .883 / .025 / 10.0, 넓은 지도 .258 / 0 / 13.0, 요청 2개(실제 [2,2] 확인) .775 / .025 / 10.0. 사건 수는 적어 잡음이 크다.
- 첫 smoke에서 요청 2개 후보가 trainer의 자동 상향으로 [3,3]이 되는 문제를 발견해 `CONFIG`에 `auto_scale_request_queue=False`를 넣었다. 재 smoke에서 [2,2].
- smoke(`--smoke-test`, 임시 폴더): 두 후보 × pgct_max·ippo·oracle_max·allk_max × seed 1·2 모두 complete. 넓은 지도 goal (0,10)/(15,10), seed 2 반전.
- 영상: smoke는 영상을 끈다(`train_from_scratch.py` smoke 설정). 1 update 일반 실행(`--save-video --track --wandb-mode offline`, 임시 폴더, 약 4.7분)에서 로컬 `videos/final.mp4`(약 0.88MB)와 W&B media 영상 생성. 5M plan에는 save_eval_video·wandb_log_eval_video True, wandb_video_interval 5, eval_interval 50.
- 테스트: `tests/test_comm5_candidate.py::test_harder_variants_change_one_thing_each` 추가, 전체 259개 통과. 중간 한 번 `tests/test_env.py::test_inactivity_1` 실패, 단독 3회·전체 재실행 통과(수정하지 않은 환경 테스트, 불안정으로 판단).
- 한계: 무작위 정책은 학습 가능성을 말하지 않는다. 넓은 지도의 5M 학습 가능성은 미확인.

## E78. 기존 지도 학습 이정표와 넓은 지도 초반 비교(읽기 전용) — 2026-11-04 Claude Code, 사용자 질문(T184)

- 스크립트 `output/difficulty_20261104/learning_milestones.py`, 출력 `learning_milestones_stdout.txt`.
- 기존 지도(11×10, 모서리·개인, 처음부터 5M, 결정적 평가 20 ep): 평가 배송/ep 0.5·5·10·20 첫 도달 — IPPO s1 0.72/1.28/2.51/4.25M, IPPO s2 0.82/1.43/2.46/4.40M, PGCT v2 s1 0.77/1.38/2.46/4.51M. 5M 시점 25.6/24.1/19.8, agent별(≥0.5/ep) 학습 시작 0.77–2.97M, 각 run 6명 중 5명.
- 학습 초반(0.026M까지) train 배송/ep: 기존 지도 .48–.57, 넓은 지도 본 실험(`runs/main_amax_20261104_115922_*`) .14–.30.
- 해석(가정): 단순 비례로 2.5–3.5배 늦으면 넓은 지도 배송 시작 약 2–3M, 5/ep 약 3.5–5M, 10/ep 약 6–9M. 비례 가정은 검증되지 않았다.

## E79. 수렴 판정·queue 재개 옵션 검증과 기존 지도 5M 수렴 여부(학습 없음) — 2026-11-04 Claude Code, 사용자 결정(T185)

- `tests/test_main_check.py` 4개(평탄 수렴·상승 미수렴, 늦게 배우는 agent 1명이면 미수렴, 시점 미도달·평가 부족이면 판정 없음, 작은 잡음은 수렴). 전체 263개 통과.
- 같은 규칙을 기존 지도 5M run에 적용(시점 2·3·4·4.95M): IPPO s1 4.95M 19.8→21.8, IPPO s2 18.2→21.6, PGCT v2 s1 16.2→19.9(배송/ep, A→B), agent 최대 변화 .59–1.09. 어느 시점도 수렴 아님.
- `run_queue.ps1 -ResumePlan`(기존 plan 재사용, 기존 PID 기록 유지)을 `-PlanOnly`로만 확인: 계획을 다시 만들지 않고 12 job을 mode 순서로 인식. 확인용 로그는 지웠다.
- 현재 본 실험 queue는 VS Code 터미널(PID 7124) 안에서 실행 중이며 4 job(부모 PID 7124)이 돌고 있다.


## E80. 51,200 step 종료 원인 조사·격리 경로 저장 검증(학습 없음) — 2026-11-04 Codex (T186)

- 대상 plan: runs/main_amax_20261104_115922_b6_wide_corner_indiv_comm5_pen005. 처음부터·넓은 지도·개인 보상·r_c=5·action_max·mirror 없음·10M·W&B online·영상 저장.
- 14:31 KST 관측: pgct_max s1–3, ippo s1–3, oracle_max s1–2 failed at 51,200. oracle_max s3 47,104, allk_max s1/s2/s3 45,056/44,032/43,008; 상태 running 및 metrics 증가 확인(CIM 조회는 접근 제한).
- 결과: 넓은 지도 10M 실행의 첫 평가(51,200 step)에서 W&B 미디어 목적지 길이 262–268자로 파일 이동 실패. LongPathsEnabled=0. 짧은 경로 성공·260자 일반 경로 FileNotFoundError·확장 경로 성공을 격리 검증했다. 로컬 eval_00050.mp4는 저장됨. 예외 위치는 trainer log_to_wandb → W&B Video.bind_to_run → shutil.move/open(dst).
- 실제 수행: 기존 로그·manifest·설치 SDK 열람, 레지스트리 조회, 별도 output 폴더의 표준 라이브러리 파일 저장 검증. W&B Video 추가 재현은 의존성 로딩 중 결과 없이 진단 프로세스만 중단. 장기 학습·새 평가 실행 없음.
- 코드 관측: --save-video이면 첫 평가부터 eval_video가 생겨 wandb_video_interval=5와 무관하게 업로드 경로가 호출된다(trainer 5192, 5228–5233).
- 한계: 현재 프로세스에 수정 주입 안 함. 새 경로에서의 실제 W&B 영상 업로드·학습 재실행 검증은 안 함.
- 근거: output/wandb_path_diagnosis_20261104/REPORT.md, diagnose.py, diagnosis.json, path_probe.py, path_probe.json; 각 job_status.json·metrics.jsonl·err.log.


## E81. 새 프로젝트·영상 10개 보존 실행기 검증(본 학습 아님) — 2026-11-04 Codex (T187)

- 실제 수행: 관련 23개 및 전체 tests/ 271개 통과(53.15초), PowerShell AST 구문 검사, 새 launcher -PlanOnly로 12 job 계획 생성.
- plan: runs/verify_am2_1104_b6_wide_corner_indiv_comm5_pen005/experiment.json. 프로젝트 rware-main-amax-v2, 요청 10M(실제 rollout 정렬 10,000,384), keep_last 10, 넓은 지도·개인·r_c5·action_max·mirror False, pgct_max/ippo/oracle_max/allk_max × s1–3 확인. 계획만이며 본 학습 미실행.
- 실제 영상 smoke: output/video_retention_20261104/main_amax_20261104_144148_b6_wide_corner_indiv_comm5_pen005/b6_wide_corner_indiv_comm5_pen005_ippo_s1/final.pt. 처음부터 IPPO s1·CPU·넓은 지도, 32×12=384 step, 결정적 평가 1ep×2step, W&B offline. 13개 영상 생성 후 로컬10·W&B offline media10개, final 포함, 최장 W&B 경로177자, complete. 성능 결과/독립 seed 실험으로 세지 않는다.
- 처음 sandbox 시도는 W&B backend 시작에 진전이 없어 해당 검증 프로세스만 중단. 새 폴더에서 sandbox 밖 offline 검증 완료. 실제 본 실험 프로세스 조작 없음.
- 새 파일: latent_team/video_retention.py, run_main_amax_v2.ps1, tests/test_video_retention.py. 수정: train_from_scratch.py 옵션·설치 연결, README, .gitignore.
- 한계: 온라인 업로드·본 학습 미실행. 종료 시 영상 업로드하며 강제 종료 시 로컬 파일만 남을 수 있음. 기존 결과 삭제 없음; 새 run의 오래된 영상만 사용자 지시대로 순환 삭제.
- 근거: output/video_retention_20261104/REPORT.md, offline_result.json, verify_offline.py; 새 plan manifest.

## E82. am2 재실행 초기 점검과 Codex 수정 검증(읽기 전용) — 2026-11-04 15:27 KST Claude Code (T188)

- 실행: `runs/am2_1104_152110_b6_wide_corner_indiv_comm5_pen005`(사용자가 15:21 `run_main_amax_v2.ps1`로 시작, W&B `rware-main-amax-v2`, 10M, keep_last 10). pgct_max s1–3(PID 34424·23532·38028), ippo s1(PID 35324), 부모 PID 7124(VS Code 터미널). 15:25 기준 0.00–0.01M, 오류 로그 오류 0. 이전 실행의 프로세스 없음.
- 경로: 가장 긴 로컬 영상 경로(oracle_max_s1 `videos/eval_00050.mp4`) 184자. W&B 목적지는 run 이름과 무관한 `wandb_data/<ID>` 아래(E81 최장 177자).
- 테스트: 전체 271개 통과.
- 51,200 step 통과 여부는 백그라운드 감시(읽기 전용)로 확인 중. 결과는 이 항목 뒤에 추가한다.
- 후속(17:58 KST): 감시 대상 `am2_1104_152110`은 사용자가 약 1만 step에서 멈추고(D119 절차) seed별 순서로 `runs/am2_1104_153231_b6_wide_corner_indiv_comm5_pen005`를 15:32에 다시 시작했다(감시는 시간 초과로 끝남). 새 실행 seed 1의 pgct_max·ippo·oracle_max·allk_max(PID 32108·20432·28100·29944)가 0.14M에서 running, 평가 2회(51,200·102,400 step)와 `eval_00050`·`eval_00100.mp4` 저장, 오류 로그 오류 0, job당 약 16 step/s. W&B 영상은 설계대로 종료 시 업로드 대기(`upload_status` pending). 51,200 step 실패는 재발하지 않았다.

## E83. 넓은 지도 본 실험 학습 붕괴 확인(읽기 전용) — 2026-11-05 11:41 KST Claude Code (T193)

- 실행: `runs/am2_1104_153231_b6_wide_corner_indiv_comm5_pen005` seed 1(pgct_max·ippo 1.17M, oracle_max·allk_max 1.08–1.09M), job당 15–16 step/s, 오류 0.
- 결과: 마지막 평가(1.08–1.13M) 배송/ep 4 mode 모두 0.00, agent별도 모두 0. 학습 중 배송/ep(0.2M 구간별) 0.15→0.02→0.01→0.01→0→0(4 mode 거의 같음). ippo_s1 학습 중 요청 선반 집기/ep 0.775→0.053→0.028→0.008→0.012→0.006, agent_1 집기 횟수 6.33→0.59→0.07→0.03→0.015→0.036, episode return −0.36→−0.04(감점 회피로 상승). pgct_max 성숙 0/6, `ac_v2_own_events_mean` 8.3.
- 해석: 보상이 멀어 감점만 경험하고 TOGGLE을 회피하는 정책으로 수렴(pickup 정체). 방법 차이가 아니라 환경 문제.

## E84. 불일치 개수 검정 오프라인 확인(행동 구분 없음, 학습 없음) — 2026-11-05 Claude Code (사용자 질문 T210)

- 질문: 공유 보상과 팀별 보상이 같은 행동 안에 섞이면 행동별 최대가 다시 희석된다. 행동으로 나누지 않고 |p−q| > δ인 probe 개수로 판정하면 되는가.
- 방법: `output/ac_drift_20261104/disagreement_count_check.py`. v2 발견 run(seed 1, 5M) checkpoint 19개, 성숙 쌍, 64 probe(최근 32개씩, 대칭 사본 없음, 행동 구분 없음). δ = .3, .5.
- 결과(δ .5): 같은 팀 최대 개수 / 다른 팀 최소 개수 — 0.26M 27/48, 1.02M 16/62, 2.05M 13/49, 3.07M 9/43, 4.35M 17/34, 4.86M 13/36. 전체 같은 팀 ≤ 27, 다른 팀 ≥ 33으로 모든 시점에서 분리되나 여유가 작다(행동별 최대는 .28 대 .97). 비율로는 같은 팀 ≤ .47, 다른 팀 ≥ .52.
- 해석: 행동으로 나눌 수 없을 때의 대안이 될 수 있으나 문턱 개수가 필요하고, 다른 팀 개수도 팀별 사건 수가 줄면 함께 준다. 구현하지 않았다.
- 출력: 같은 폴더 `disagreement_count_check_stdout.txt`, `.json`.

## E85. 팀 차이가 드문 경우 오프라인 모의(학습 없음) — 2026-11-05 Claude Code (사용자 질문 T211)

- 방법: `output/ac_drift_20261104/rare_difference_check.py`. v2 발견 run checkpoint_04250(4.35M)의 예측기와 512 사건 버퍼에서, 각 agent의 probe를 팀별 사건(배송, FORWARD) k개 + 공유 사건(반납, TOGGLE) 32−k개로 재구성(대칭 사본 없음). 두 종류가 32개 이상인 성숙 agent 1–5.
- 결과(같은 팀 최대 / 다른 팀 최소): 합친 거리 k=8 .16/.18, k=16 .12/.36, k=32 .01/.99. 행동별 최대 n_min 8: k ≤ 6 다른 팀 .03(분리 실패), k ≥ 8 .98 이상. n_min 2: k ≥ 2 다른 팀 .98, 같은 팀 ≤ .21. 불일치 개수(|p−q|>.5): k=0 18/0, k=8 14/16, k=16 12/32.
- 해석: 드문 차이에서 실패하는 원인은 거리 식이 아니라 "최근 32개 중 n_min개" 경험 조건이다. 행동별 probe 수집(행동마다 최근 사건 따로 보관) 또는 n_min 낮추기가 대안이며, 후자는 경험 불균형 쌍의 같은 팀 거리를 키울 수 있다. 구현하지 않았다.

## E86. 저장소 분리(D131) 회귀 검증과 Overcooked smoke·sanity — 2026-11-05 Claude Code (T216)

- 실행 여부: 실제 실행. 장기 학습 아님(smoke·sanity만).
- 환경: 이 PC `C:\PGCT_Overcooked\robotic-warehouse`(브랜치 `feature/overcooked`, 시작 커밋 122615f), conda `marl-policy-sharing`(Python 3.11.15, torch 2.11.0+cu128, numpy 2.4.6, gymnasium 1.3.0), CPU. 문서의 `RWARE_MARL` 환경은 이 PC에 없고 `pyglet`도 없다.
- 로컬 참조 실험: 이 PC에는 `runs/sp_20261008_120638_983201/experiment.json`이 없어, 저장된 plan(`output/ac_validation_20261023_final_tmp/test_from_scratch_plan_is_rand0/plan/experiment.json`의 `e1_pgct_s3`)으로 대체본을 만들었다(`_reconstructed` 표시, git 무시, 커밋 안 함). 원래 파일이 아니다.
- RWARE 테스트(`RWARE/tests`, 이 PC):
  - 변경 전: 264 통과, 7 건너뜀, 1 실패. 실패는 `test_policy_probe_consensus.py::test_rware_probe_map_builder_saves_visual_manifest`(pyglet 없음, 렌더링)로 분리와 무관하다. 대체본 없이 돌리면 240 통과·31 건너뜀·같은 1 실패.
  - 구조 이동 직후: 264 통과, 7 건너뜀, 같은 1 실패.
  - 알고리즘 추출 후: 289 통과, 8 건너뜀, 같은 1 실패(기존 264 + 등가성 25; 추가 건너뜀은 agent checkpoint가 아닌 probe bank 파일).
- 등가성 테스트(`RWARE/tests/test_pgct_core_equivalence.py`): 분리 전 동결 사본(`RWARE/tests/legacy/`)과 새 코드를 같은 입력에서 비교, 허용 오차 0. gate 8 update 기록(pooled/action_max × mirror × snapshot × view/원관측), 설정 기본값, 거리, GAE, Max-GAE 목표·혼합, donor 규칙 5종, core 네트워크·PPO 업데이트 대 trainer `update_ippo`(가중치 비트 동일), 기존 checkpoint 적재(MLP 1, CNN 1), 실제 trainer `pgct_max`·`oracle_max` 옛 모듈 대 새 모듈. 모두 통과.
- trainer 지문(`RWARE/tests/characterize_pgct_d2.py`, `b6_corner_indiv_comm5_pen005`, action_max, seed 1, 64 step smoke, 6 mode × 3 설정): 추출 전후 18/18 동일(최종 가중치·gate 상태·max_gae.jsonl). gate 닫힘이면 pgct_max = ippo, oracle 주입이면 pgct_max = oracle_max, 허용 설정에서는 실제 비교 10회·donor 16 arc.
- 공유 코어 테스트(`PGCT/tests`): 12 통과(toy 환경에서 6개 방법 학습 루프 포함).
- Overcooked 테스트(`Overcooked/tests`): 14 통과. 스크립트 경험(각 agent가 두 수프를 모두 가끔 배달)에서 RWARE D2 값 그대로의 gate가 숨은 팀 2쌍만 승인(AUROC 1, F1 1, 같은 팀 거리 < .2, 다른 팀 > .8). gate 닫힘이면 pgct_d2·pgct_id_only·random_gate 가중치 = ippo, oracle 승인 주입이면 pgct_d2 = oracle, oracle 외 방법은 라벨을 읽지 않음.
- Overcooked smoke: 6 방법 × seed 1, 256 step(`Overcooked/runs/oc_smoke_20261105`, `oc_smoke_parallel_20261105`, PowerShell 실행기·집계 포함) 오류 없음. 첫 시도에서 병렬 job의 `plan.json` 동시 쓰기 경쟁으로 1 job이 실패해 원자적 교체·재시도로 고쳤고, 6개 동시 실행으로 재확인했다.
- Overcooked sanity(주 설정, 51,200 step, warm-up 10, ippo·pgct_d2 × seed 1, scratchpad): 두 job 모두 정상 종료(각 약 20분, 동시 실행에서 약 42 step/s). 51,200 step 동안 배달 0회였고 자기 재료 투입은 있었으나, gate 보상 사건(r ≥ .5)이 0개라 모든 agent가 미성숙이고 donor도 없었다. 그 결과 pgct_d2 최종 가중치가 ippo와 비트 단위로 같았다(gate가 학습 난수 흐름을 건드리지 않음을 실제 설정에서 재확인). 결정적 평가 수익 0. 학습 신호가 생기는지는 이 길이로 판단할 수 없다.
- 한계: 학습·식별 성능 결과가 아니다. 1M checkpoint가 필요한 RWARE 실제 trainer 테스트 3개는 이 PC에서 건너뛰었다. 출력 폴더 중 scratchpad 것은 세션 임시 폴더라 남지 않는다.
- 근거: `docs/MIGRATION.md`, `RWARE/tests/test_pgct_core_equivalence.py`, `RWARE/tests/characterize_pgct_d2.py`, `Overcooked/tests/`, `Overcooked/docs/EXPERIMENT_DESIGN.md`.

## E87. 최신 논문·현재 RWARE 2M run·Overcooked 재현 점검 — 2026-11-05 18:42 KST Codex (T218)

- 첨부: `C:/Users/95101/Downloads/PGCT_fin.pdf`(504,471 bytes, 11-05 18:28). SHA-256 `A3707649928E90CFC232D5C9E9FFD518EAE217216003079F55FB0ACD8ED71190`로 저장소 `MainPaper/PGCT_fin.pdf`와 동일. 본문 전체를 text 추출하고 LaTeX·코드와 대조했다.
- 논문 대 코드: 개인 reward MLP(공개 before/action one-hot/after, 128-64), event ≥.5, 최근 event 32/agent, action-max n_min=8, EMA .35, 3 rounds, threshold .5, 5-update refit, rollout-start snapshot, contact donor, Max-GAE η=.5/shrink/window 8, critic own-return target가 일치한다. 논문의 현재 11×16 run 문장은 E83 이후 오래된 상태이며 현재 run은 11×10이다.
- W&B 읽기 전용 확인(project `t98339-hanyang-university/rware-main-amax-v2`, 18:36 KST): IPPO `yidd96p3` running, heartbeat 18:36:46, 617,472 step, 평가 배송 0. PGCT `57ebqe5l` running, heartbeat 18:36:40, 580,608 step, 평가 배송 0, mature 5, same-team approved arcs 8, cross-team 0, coverage 2/3. 이 PC에는 Python 프로세스가 없어 실행은 다른 호스트다.
- 정확 계획 재생성: `RWARE/runs/verify_day_base_1105_exact/experiment.json`. scratch, seed 1, 6 agent/2×3, 11×10 corner, 개인 보상, r_c=5, 2M(실제 2,000,896), PPO 1024/64·4 epoch·8 minibatch·lr 2e-4·entropy .05, gate action_max/no mirror/snapshot, Max-GAE η=.5/shrink/warm-up 100. 기록·논문과 일치.
- RWARE 실제 smoke: 같은 후보와 두 mode, 64 step. 둘 다 `final.pt` 저장, 초기 hash `caaef33f...abcfae` 일치. 소프트웨어 경로 정상.
- 테스트: `PGCT/tests` 12 통과; `RWARE/tests/test_pgct_core_equivalence.py` 27 통과·3 건너뜀; `Overcooked/tests` 14 통과.
- Overcooked 6-method smoke: ippo/oracle/all/pgct_d2/pgct_id_only/random_gate × seed 1, 256 step, 모두 `status=complete`, `final.pt` 6개. wiring 확인용이라 return 0은 성능 결과가 아니다.
- 한계: E86의 주 설정 51,200-step sanity에서는 delivery/event/maturity가 0이었다. 5M 본 실험의 과학적 유효성은 미확인이다. Overcooked runner는 W&B가 아니라 `metrics.jsonl`, `gate.jsonl`, `eval.jsonl`만 쓴다. 본 실험 전에 0.5–1M pilot을 권한다.
- 실행 장애: `RWARE/latent_team/run_queue.ps1` 기본 Python은 이 PC에 없는 `RWARE_MARL`; 설치된 호환 환경은 `marl-policy-sharing`. `run_main_amax_v2.ps1`은 Python 인자를 전달하지 않으므로 이 PC에서는 direct Python entry 또는 `run_queue.ps1 -Python <경로>`가 필요하다. PowerShell script는 execution-policy bypass도 필요하다.

## E88. Overcooked W&B 연동·RWARE 출력 정리 후 회귀 검증 — 2026-11-05 19:15 KST Codex (T221)

- 실행 여부: 학습 성능 실험은 새로 시작하지 않았다. W&B offline wiring smoke 1개(IPPO, seed 1, 256 step)와 tests/plan-only만 실행했다. 진행 중 `oc_pilot_500k`는 중단·재시작하지 않았다.
- W&B offline smoke: 실제 SDK `wandb 0.30.0`, project `pgct-overcooked-smoke`, run ID `amqxujbc`, 4 update/256 step 완료, metrics·eval·summary와 `final.pt` 생성. sandbox 사용자가 사용자 홈 W&B 설정/로그 경로를 읽지 못한다는 경고가 있었으나 job exit 0, offline run 파일 저장 완료. 실제 사용자 PowerShell의 권한 상태를 뜻하지 않는다.
- launcher plan-only: `verify_wandb_plan_1105`, IPPO seed 1, 128 step, `tracking.enabled=true`, project `pgct-overcooked`, mode `online` 기록. 학습은 실행하지 않았다.
- 테스트: `Overcooked/tests` 16 통과(새 tracking 2 포함), `PGCT/tests` 12 통과, `RWARE/tests/test_pgct_core_equivalence.py` 27 통과·3 건너뜀. 알고리즘 수치 경로 변경 없음.
- RWARE cleanup: `RWARE/output` 상위 53개 디렉터리, 1,445개 파일, 171,382,902 bytes 삭제. 보존 8개는 `ac_drift_20261104`, `ac_early_id_20261026`, `ac_offline_20261026`, `fig2_real_20261105`, `fig_data_20261105`, `paper_revision_20261105`, `pgct_d2_validation_20261102`, `pgct_docs_20261102`. `RWARE/runs`와 source/checkpoint는 삭제하지 않았다. 상세 목록 `docs/RWARE_OUTPUT_CLEANUP_20261105.md`.
- 현재 pilot 관측(19:15): IPPO 102,400 step(최근 episode return mean .233), PGCT-D2 100,352 step(.213), 오류 로그 비어 있음. 아직 warm-up 경계이며 PGCT mature/approval/donor 0. 기존 프로세스에는 새 W&B callback이 반영되지 않는다.

## E89. RWARE day_base 재현성(현 저장소 대 실행 중 W&B run)과 Overcooked 500k·2M 학습·gate 진단, 재생 영상 — 2026-11-05 Claude Code (T224)

- 실행 여부: 실제 실행(읽기 전용 W&B 조회, 짧은 재실행, checkpoint 분석). 장기 학습 없음. 실행 중인 학습(`oc_2m_pair_v1`, 다른 PC의 RWARE 2 run)은 건드리지 않았다.
- RWARE 설정 대조: 다른 PC(DESKTOP-03ASKH8, RTX 5070 Ti, 분리 전 코드)에서 실행 중인 `b6_corner_indiv_comm5_pen005_acv2_amax_{ippo,pgct_max}_s1`(W&B `rware-main-amax-v2`, `yidd96p3`·`57ebqe5l`)의 W&B config와 현 저장소 계획(`RWARE/runs/verify_day_base_1105_exact`)이 trainer 87개·환경 40개 항목 모두 같다(예산 2,000,896, warm-up 100).
- RWARE 짧은 재실행: 같은 계획을 3 update(3,072 step)로 이 PC(DJKLSFJ, RTX 3060, CUDA)에서 실행(`RWARE/runs/verify_gpu_3upd_1105`, W&B 끔). 첫 update에서 롤아웃의 이산 결과(배송·집기·에피소드 수익·행동 수 등 train 지표 99/114)가 W&B 기록과 정확히 같고, 행동 확률·hidden norm·loss는 상대 1e-5 수준으로 다르다(GPU 부동소수점). 2번째 update부터 경로가 갈라진다. 다른 GPU에서 비트 단위 연속은 기대할 수 없다. 같은 이 PC GPU에서도 ippo와 pgct_max의 첫 loss가 마지막 자리에서 달라 CUDA 연산 비결정성이 있다(원격은 두 run이 같았다).
- 분리 전후 동작: Codex W&B 콜백과 이번 eval_hook 추가 후 `RWARE/tests/test_pgct_core_equivalence.py` 28 통과·4 건너뜀(건너뜀은 모두 probe bank 파일), trainer 지문 18/18이 분리 전 기록(E86)과 동일.
- Overcooked 학습(`oc_pilot_500k`, seed 1, 500,736 step, CPU, job당 약 135–150 step/s): 에피소드당 자기 레시피 배달 0(0–100 update) → 약 25(451–489 update), 결정적 평가 수익 agent당 8.475(자기 배달 21). `oc_2m_pair_v1`(같은 seed·설정, W&B 추적)의 448k까지 지표는 pilot과 숫자까지 같다(CPU 결정적).
- Overcooked gate: pgct_d2가 처음부터 끝까지 승인 0, donor 0이라 IPPO와 가중치·지표가 동일. 500k pilot `final.pt` 진단(`reward_gate` 상태): 팀 [0,0,1,1]. agent 0(양파 팀)은 배달 사건 2회로 미성숙(요리 담당). agent 2·3(토마토 팀)은 각각 서쪽·동쪽 창구에서만 배달해 서로의 배달 probe에 대한 예측이 .04(자기 1.0), 쌍 거리 EMA .92. 다른 팀 쌍 1-2 .82, 1-3 .999. 2M run 감사 거리(435k) 같은 팀 .919, 다른 팀 .917. 원인: 공개 view가 자기 중심이지만 회전 정규화가 없어 좌우 대칭 창구의 같은 사건이 다른 입력이 됨 + 역할 분화로 한 팀원이 보상 사건을 갖지 못함.
- 재생 영상: `Overcooked/runs/oc_pilot_500k/ippo_s1/videos/final_seed5000001_det.mp4`(400 step, 수익 8.475로 평가와 일치), W&B run `el1xz1u1`(project `pgct-overcooked`, group `oc_pilot_500k`). 학습 중 영상 기록 옵션 추가(테스트로 학습 가중치 불변 확인).
- 테스트: Overcooked 19 통과(영상 3개 추가), 공유 코어 12 통과.
- 한계: Overcooked는 seed 1 하나, 0.5M까지만 관찰. 방향 정규화 가설은 아직 검증하지 않았다.

## E90. Overcooked probe v2 사전 검증: 스크립트 반대 창구 시험과 500k 정책 재생 오프라인 비교 — 2026-11-05 Claude Code (T225)

- 실행 여부: 실제 실행. 학습 없음(테스트, 오프라인 재생, 256 step smoke). 실행 중인 `oc_2m_pair_v1`은 건드리지 않았다.
- 스크립트 시험(`Overcooked/tests/test_probe_v2.py`, 팀 고정 [0,0,1,1], 각 팀 한 명은 서쪽·한 명은 동쪽 창구에서만 배달, 수프 종류 무작위, D2 gate 값, 8 update): 같은 팀 EMA v1 .528/.537(미승인) → v2 .317/.265(정확히 팀원 2쌍 승인). 다른 팀은 v1 .92–1.0, v2 .92–.99.
- 오프라인 재생(`python -m pgct_overcooked.offline_gate_check --job Overcooked/runs/oc_pilot_500k/ippo_s1 --updates 40`): 500k pilot 최종 정책(행동 샘플링, torch seed 12345)으로 40 update(40,960 step) 경험을 만들어, 같은 경험을 새 gate에 v1·v2로 각각 넣었다. 팀 [0,0,1,1], 자기 배달 [0, 999, 962, 665], agent 0(요리 담당)은 미성숙.
  - v1: 2–3(같은 팀, 반대 창구) .76 미승인, 1–2(다른 팀, 같은 서쪽 창구) .054 오승인(20,480 step), 1–3 .924. F1 0.
  - v2: 2–3 .358 승인(35,840 step), 1–2 .067 오승인, 1–3 .547. precision .5, recall .5, F1 .5.
- 해석: v2는 반대 창구 팀원 문제를 해결한다. 그러나 학습된 정책은 상대 수프를 거의 배달하지 않아, 그 경험만으로 새로 학습한 예측기는 수프 종류를 구분하지 못하고 같은 창구의 다른 팀을 승인한다(v1·v2 공통). 실제 학습에서는 초반 탐색의 상대 수프 배달이 섞여 pilot gate의 1–2 거리가 .82였으므로, 이 재생은 다른 팀 분리에 비관적인 조건이다. v2는 1–3 거리도 .92→.55로 낮춰 다른 팀 여유가 줄었다.
- 기타 검증: Overcooked 테스트 28 통과(v2 9개 추가, v1·v2 설정에서 IPPO 비트 동일), RWARE gate 관련 60 통과·7 건너뜀, trainer 지문 18/18 분리 전과 동일, v2 smoke(pgct_d2·pgct_id_only·random_gate, 256 step, 영상 포함) 정상, `Overcooked/runs/oc_pilot_500k_probe_v2` 계획 생성(pgct_d2·oracle, 학습 안 함).
- 한계: seed 1, 고정된 최종 정책 재생이라 학습 중 경험 분포와 다르다. 실제 판정은 0.5M 학습 pilot에서 한다.

## E91. day_base RWARE 2 run(진행 중)과 이 PC Overcooked 학습 분석 — 2026-11-06 KST Claude Code (T227)

- 실행 여부: 읽기 전용 분석(W&B API `run.history`, 브라우저 W&B 화면, 로컬 JSONL). 학습 시작·중단 없음. 제가 띄운 W&B 전체 다운로드(느려서)만 중단했다.
- RWARE `rware-main-amax-v2`(DESKTOP-03ASKH8, RTX 5070 Ti, 분리 전 코드, seed 1, 11×10 모서리·개인 보상·r_c 5, 2M 예정): 01시 무렵 IPPO 1,257,472 step, PGCT-D2 1,169,408 step, 12.8시간, 약 27 step/s. 팀 [0,1,0,1,0,1].
  - 식별(PGCT-D2): 성숙 agent 0.16M부터 늘어 0.23M에 5/6(agent 1은 끝까지 미성숙, 보상 사건 없음). 승인 같은 팀 arc 8/12(성숙 agent 사이 같은 팀 쌍 전부), 다른 팀 0, donor 8개 모두 같은 팀, 0.25M 이후 변동 없음. 감사 거리 같은 팀 .25→.07, 다른 팀 .99–1.00.
  - 학습(창 평균): 학습 배송/에피소드 IPPO 0.02–0.18 대 PGCT 0.37(0.6M)→1.06(0.8M)→1.9–2.5(0.9–1.2M). 학습 수익 IPPO −0.01–0.01 대 PGCT 0.16→0.47. 결정적 평가 배송 IPPO 1.23M까지 24회 모두 0, PGCT 0.768M 0.25 → 1.024M 2.30 → 1.18M 1.45. 배송은 agent 0(팀 0)과 agent 5(팀 1) 두 명만 한다. 이 둘의 critic EV .73/.69, 나머지 −.22–.11.
  - Max-GAE: 수신자 5, select_frac .48→.63, mean_b .02→.29, adv_shift .15→.35, eta_eff .44→.14(shrink).
  - 이상 신호: PGCT의 요청 선반 집기 8–14/에피소드, 잘못된 반납 4.6–6.3/에피소드, 1.126M 평가에서 잘못된 반납 39.75·집기 42.5. 집기 보상은 agent·선반마다 한 번만 지급(`_pickup_shaping_reward`)되므로 보상 악용은 아니고, 반납 단계 미학습에 따른 반복 동작으로 보인다(영상 미확인).
  - 한계: seed 1 하나, IPPO 대 PGCT-D2만 있어 이득을 critic 접근 때문이라고 단정할 수 없다(oracle_max·allk_max·추가 seed 필요). 2M 판정 전.
- Overcooked(이 PC, seed 1):
  - `oc_2m_pair_v1`(v1): 2026-11-05 21:08 두 job 모두 KeyboardInterrupt로 중단(IPPO 526,336, PGCT 512,000 step). 0.5M pilot과 숫자까지 같은 구간이라 새 정보 없음.
  - `oc_pilot_500k_probe_v2` pgct_d2: 0.497M(거의 완료), 승인 0. 같은 팀 2–3 EMA .578(v1 .919), 다른 팀 1–2 .815, 1–3 1.0, agent 0 미성숙. 학습 지표는 v1 pilot과 486 update 동안 동일(배달 0→약 25/에피소드). 처리량 36 step/s로 v1 pilot의 약 1/3.5, 롤아웃·PPO·gate가 고르게 느려져 CPU 클럭 저하(전원·발열 설정)로 보인다. oracle job은 아직 시작 전.
- 해석: RWARE에서는 식별이 깨끗하고(오승인 0, 같은 팀 거리 .07–.25) PGCT-D2만 수집 붕괴에서 벗어나는 중이다. Overcooked는 v2로 같은 팀 거리가 .92→.58로 줄었지만 임계 .5를 넘지 못한다. 남은 차이는 회전 후에도 창구 앞 양 대각선 칸(접시 디스펜서/카운터)이 좌우로 다르기 때문으로 보인다. 두 환경 모두 보상 사건이 없는 agent는 묶이지 않는다.

## E92. Overcooked v3 사전 검증과 v1·v2·v3 정책 재생 비교, 승인 임계 수치 검증 — 2026-11-06 Claude Code (T228)

- 실행 여부: 실제 실행(테스트, 오프라인 재생, 수치 계산, 그림). 학습 없음. 실행 중인 `oc_pilot_500k_probe_v2/oracle_s1`은 건드리지 않았다.
- v2 pilot 결과(학습): `oc_pilot_500k_probe_v2/pgct_d2_s1` 완료(500,736 step, 3.9시간, 약 36 step/s). 승인 0, 같은 팀 2–3 EMA .578, 다른 팀 1–2 .815, 1–3 1.0, agent 0 미성숙. 학습 지표는 v1과 동일.
- v3 테스트(`Overcooked/tests/test_probe_v2.py`): 서쪽·동쪽 창구 배달의 v3 view가 완전히 같다. 스크립트 반대 창구 팀원 EMA .001/.004, 다른 팀 .956–.972(v1 .53/.54, v2 .32/.27). Overcooked 테스트 30 통과.
- 그림: `Overcooked/docs/img/probe_views_v1_v2_v3.png`(실제 projection 출력에서 그림, `Overcooked/docs/make_probe_view_figure.py`).
- 정책 재생(`offline_gate_check --schedule checkpoint_00250.pt:20 final.pt:20`, 500k pilot IPPO 정책, 같은 경험): 자기 배달 [0, 555, 485, 369], 상대 수프 배달 [0, 6, 2, 164]. 같은 팀 2–3은 세 프레임 모두 비교 0회였다. agent 2가 후반에야 성숙해 비교 기회가 3번이었고, 최종 정책 구간의 2–3 접촉은 롤아웃의 40%였다(중간 구간은 100%). 다른 팀 EMA 1–2: v1 .113 / v2 .134 / v3 .146(모두 오승인), 1–3: .941 / .723 / .297(v3 오승인).
- 해석: v1이 다른 팀 1–3을 떼어 놓은 것은 수프 종류가 아니라 창구 위치였다. v3는 위치를 지워 수프 종류만 남긴다. 재생에서 새로 학습한 agent 1 예측기는 상대 수프 경험이 6번뿐이라 모든 배달에 1로 답했고, agent 3은 구분했다. 한쪽만 구분하면 pooled 거리는 n_i/(n_i + 2n_j) = 1/3(같은 probe 수)이 되어 τ = .5에서 승인된다(수치 확인). 실제 학습의 v1·v2 pilot에서는 같은 서쪽 창구의 1–2가 .82로 떨어져 있어, 학습 궤적에서는 예측기가 수프 종류를 배웠다. 재생은 이 점에서 비관적이며 v3의 실제 효과는 판정하지 못했다.
- 임계값 수치 검증(`threshold_math.py`): 이상적 같은 팀 0, 다른 팀 1(probe 수와 무관). 공유 사건 비율 s의 pooled 다른 팀 거리 (1 − s)/(1 + s). 응답 비율 ρ: D = (1 − ρ)/(1 + ρ), τ = .5 ⇔ ρ* = 1/3. 균일 오차 ε: 같은 팀 ≤ ε, 다른 팀 ≥ 1 − 2ε(격자 탐색으로 tight 확인), τ = .5 보장 ε < 1/4, 최적 τ = 1/3(ε < 1/3).
- 한계: v3의 실제 식별은 학습 궤적에서만 확인할 수 있다. `pgct_id_only` v3 0.5M(IPPO 궤적 재현, gate 판단만 기록)을 권장한다.

## E93. Overcooked 해결책 오프라인 시험: 사건 기준 낮추기와 probe 주인별 거리 — 2026-11-06 Claude Code (T229)

- 실행 여부: 실제 실행(정책 재생, 학습 없음). 500k pilot IPPO 정책을 checkpoint_00250 20 update → final 20 update로 재생, torch seed 12345, v3 프레임, 같은 경험.
- pilot 역할 분석(학습 기록, 에피소드당): 자기 재료 넣기 400–489 update [12.3, 28.8, 28.0, 23.9](agent 0 요리 담당 포함), 자기 배달 [0, 9.2, 8.5, 5.5], 상대 재료 넣기 0.0–0.01(초반 0–100 update 0.10–0.25), 상대 수프 배달 [0, 0.15, 0.19, 2.68].
- A(사건 .5): 사건 [0, 555, 485, 369], agent 0 미성숙. 같은 팀 2–3 비교 0회(최종 예측기 간 거리 .012). 다른 팀 1–2 .146, 1–3 .297 오승인.
- B(사건 .1, min_event_response .06): 사건 [1230, 2844, 2973, 2335], 네 명 모두 성숙. 같은 팀 0–1 .307, 2–3 .258 승인(재현율 1.0, 첫 정답 승인 20,480 step). 다른 팀 1–2 .259, 1–3 .320 오승인, 0–2 .524, 0–3 .708 거부. precision .5, F1 .67.
- probe 주인별 거리(분석만, gate 규칙 아님): 다른 팀 1–2 A .083/.225, B .127/.231. 1–3 A .391/.229, B .437/.232. 양쪽 모두 낮아 주인별 최댓값으로도 거부되지 않는다(양쪽 예측기 모두 구분 못 함). 같은 팀 B 0–1 .014/.533은 최댓값 .53으로 오히려 같은 팀을 막는다. 이 경우 주인별 최댓값은 해결책이 아니다.
- 해석: 사건 기준을 환경 보상 단위로 맞추면 성숙·같은 팀 승인 문제는 풀린다. 남은 다른 팀 오승인은 대조 경험 문제다. 재생은 새 예측기가 짧은 최신 경험만 보는 비관적 조건이고, 실제 v1 pilot 학습에서는 같은 서쪽 창구의 다른 팀 1–2가 .82였다.
- 근거: `Overcooked/pgct_overcooked/offline_gate_check.py`(`--gate`, owner_split 출력), 결과는 scratchpad `sol_A.json`, `sol_B.json`(세션 임시).

## E94. 상대 보상 사건 규칙 구현 검증과 Overcooked 재생 — 2026-11-06 Claude Code (T231)

- 실행 여부: 실제 실행(테스트, 지문, 계획 생성, smoke, 정책 재생). 학습 없음.
- 공유 코어 테스트 15 통과(추가 3: 사건 선택·기준 상승 시 정리·상수배 불변, 기본 절대 규칙 불변, 보상을 1/20로 줄인 toy에서 상대 규칙으로 성숙과 정확한 팀 승인).
- RWARE: gate 관련 테스트 60 통과·7 건너뜀. trainer 지문 18/18이 분리 전 기록과 동일하다(지문은 분리 후 추가된 설정 키 3개를 제외하고 비교). 상대 설정 계획 job 이름 `b6_corner_indiv_comm5_pen005_acv2_amax_rel_{pgct_max,ippo}_s1`, manifest에 event_threshold relative·event_scale .1·ratio .3·mirror false·action_max 기록, 학습 없음.
- Overcooked: 테스트 30 통과, `main_4p2t_probe_v3_rel.json` pgct_d2 smoke(256 step) 완료.
- 정책 재생(500k pilot, checkpoint_00250 20 + final 20, seed 12345, v3): 상대 규칙(κ .1, ρ .3) 결과가 절대 .1(E93 B)과 모든 수치에서 같았다. 사건 [1230, 2844, 2973, 2335], 네 명 성숙. 같은 팀 0–1 .307·2–3 .258 승인. 다른 팀 1–2 .259·1–3 .320 오승인, 0–2 .524·0–3 .708 거부. precision .5, recall 1.0. 배달하지 않는 agent 0의 M은 .2라 기준 .02에서 재료 넣기(.1)가 사건이 된다.
- 해석: 환경별 튜닝 없이 같은 규칙으로 Overcooked의 사건 범위 문제가 풀린다. 다른 팀 오승인(대조 경험 부족)은 그대로다.

## E95. RWARE PGCT leading/lagging agent 학습 상승량 중간 분석 — 2026-11-06 02:45 KST Codex

- 실행 여부: W&B API 읽기 전용 분석. 코드·설정·실행 중인 학습을 변경하지 않았다.
- 대상: W&B project `t98339-hanyang-university/rware-main-amax-v2`, seed 1, 2M 예정. PGCT-D2 run `57ebqe5l`은 1,319,936 step, IPPO run `yidd96p3`은 1,423,360 step에서 실행 중이었다. 비교는 가장 최근 공통 평가점 1,280,000 step을 사용했다.
- 평가 조건: 결정적 정책, 20 episode × 500 step. `eval/agent_i_delivery_count`는 20 episode 전체 배송 수다. 팀은 기본 balanced assignment `[0,1,0,1,0,1]`, 즉 team 0 = A1·A3·A5, team 1 = A2·A4·A6이다.
- 1.28M 결과: PGCT `[30,0,13,17,0,40]`, IPPO `[0,0,0,0,0,0]`. 경험적 leader는 A1·A6이고, 배송이 나타난 lagging receiver는 A3·A4다. A3는 PGCT +13회(+0.65/episode), leader A1의 43.3%; A4는 +17회(+0.85/episode), leader A6의 42.5%다. A3 첫 배송은 1.28M, A4는 1.2288M이며 leader A1·A6은 0.768M부터 배송했다. A5는 donor access가 있으나 아직 배송 0, A2는 reward predictor 미성숙으로 donor access가 없고 배송 0이다.
- 전달 기작: 0.768–1.28M 구간에 donor arc 8개가 모두 같은 팀이고 cross-team arc는 0, receiver는 5명이다. donor-derived max term 선택 비율은 평균 .643, 자기 advantage 대비 정규화 평균 변화량은 .338이다. A2만 reward predictor 미성숙이므로 team 1에서 A4의 유일한 donor 후보는 A6이다. team 0의 A3는 A1과 A5 모두를 후보로 가지므로 A1 단독 기여는 분리할 수 없다.
- 한계: seed 1 중간 결과이며 IPPO가 공통 평가점에서 0이라 상대 개선율(%)은 정의할 수 없어 절대 배송 증가량과 leader 대비 비율을 보고했다. W&B에는 receiver별 실제 선택 donor와 agent별 `select_frac`이 없고 aggregate만 있다. 따라서 A4←A6은 강한 정황이지만 엄밀한 인과 효과는 아니며, 정확한 edge별 사용량은 실행 호스트의 `max_gae.jsonl`과 critic-access ablation/추가 seed가 필요하다.

## E96. LBF·VMAS 환경 구현 검증과 짧은 IPPO 시험 학습 — 2026-11-06 03:00 KST 전후 Claude Code (T236)

- 실행 여부: 테스트, 원본 비교, 스크립트 정책 식별 시험, smoke, 짧은 IPPO 시험 학습만 했다. 본 실험(수백만 step)은 하지 않았다. 환경: 이 PC, `marl-policy-sharing`(Python 3.11.15, torch 2.11 CPU 실행, numpy 2.4.6, gymnasium 1.3.0).
- 테스트: PGCT 15, Overcooked 30, LBF 22(식별 시험 포함), VMAS 12 통과. `pgct_core`는 바꾸지 않았으므로 RWARE 동결 사본 비교는 다시 돌리지 않았다.
- LBF 원본 일치(`LBF/tests/test_lbforaging_equivalence.py`): 팀 1개 설정에서 lbforaging 2.0.0과 비교했다. 6개 설정(크기 6–12, agent 2–4, 먹이 1–4, 시야 1–8, coop 포함) × 25 episode, 무작위 행동(LOAD 30%)으로 7,406 step을 비교했다. 그중 적재 성공 step 44, 실패 step 755다. 위치·먹이·보상·종료·격자 관측이 모두 같았다. lbforaging 2.0.0은 gymnasium 1.x에서 격자 관측 공간 생성이 실패한다(Box shape 불일치). 그래서 원본을 벡터 모드로 만든 뒤 원본의 격자 관측 코드 경로로 전환해 비교했다.
- LBF 스크립트 정책 식별(`LBF/tests/test_identification.py`):
  - 조건: 매 step 무작위 두 agent가 무작위 종류 먹이 옆에서 LOAD하고, 나머지는 무작위로 움직인다. 150 step × 8 update, main 설정의 gate 값(상대 사건 규칙, action_max, 거울 없음, snapshot).
  - 결과: 네 명 모두 성숙했고, 같은 팀 두 쌍만 승인됐다. AUROC 1.0, 같은 팀 거리 평균 .003, 다른 팀 .905(EMA .902–.908), 첫 올바른 승인 750 step, 오승인 없음.
  - 한계: 모든 agent가 두 종류 먹이를 다 겪는 이상적인 경험이다. 학습된 정책의 대조 경험 부족(Overcooked 조건 3)은 이 시험으로 확인되지 않는다.
- VMAS: 덮기 보상, 협동 변형, 난수 격리, 라벨 무관성, 학습 경로 테스트가 통과했다.
  - VMAS 난수 상태는 프로세스 전체가 공유한다(`Environment.vmas_random_state`). 어댑터마다 자기 사본을 쓰도록 고쳤고, 평가 환경을 끼워 넣어도 학습 환경 관측열이 같음을 테스트로 확인했다.
  - 속도: 무작위 정책 1 thread 기준 초당 141 step(LIDAR 12선)과 146 step(8선). cProfile상 LIDAR가 step 시간의 약 40%다.
- LBF 속도: 무작위 정책 기준 초당 약 22,600 env step.
- 스크립트 경로: `scripts/run_experiment.ps1 -Experiment LBF -Smoke -NoTrack`로 ippo·pgct_d2 2개 job을 병렬 실행했다(`LBF/runs/lbf_smoke_ps_1106`). 영상 3개와 `aggregate_experiment.ps1` 요약까지 생성됐다. smoke 값은 배선 확인용이라 수치 의미가 없다.
- LBF IPPO 시험 학습(`LBF/runs/lbf_pilot_ippo_200k/ippo_s1`, main_4p2t, seed 1, 숨은 팀 [0,0,1,1]):
  - 규모: 200,704 step, 976초(초당 약 206 step, 다른 시험·테스트와 CPU를 같이 썼다).
  - 학습 중 episode 평균(agent당 보상): update 1–10 .047, 91–100 .099, 187–196 .166. agent당 자기 팀 먹이 수집은 episode당 .19에서 .66으로 늘었고, 다른 팀 먹이 수집은 .17에서 .12로 줄었다.
  - 결정적 평가(20 episode, 최대 50 step), agent 평균 보상: 51k .019, 102k .071, 154k .130, 200k .143.
  - 200k 평가 세부: episode당 자기 팀 먹이 수집 합계 2.4개(전체 6개 중), 다른 팀 0.2개, 적재 실패 51.4회. 팀 보상(최대 1)은 team0 .253, team1 .321.
  - 해석: IPPO는 배우지만 200k에서 먹이의 약 40%만 모은다. 결정적 정책은 혼자 들 수 없는 먹이에 LOAD를 반복한다. 영상 `videos/final_seed5000001_det.mp4`에서 레벨 1 agent가 레벨 2 먹이 옆에서 반복 적재한다. 같은 팀 협력이 병목이므로 critic 전이(Q2)를 볼 여지가 있다. seed 1개, 200k라 결론은 아니다.
  - 비용: 5M이면 job당 약 6.7시간이다(초당 206 step 가정).
- VMAS IPPO 시험 학습(`VMAS/runs/vmas_pilot_ippo_100k/ippo_s1`, main_4p2t, seed 1, 숨은 팀 [0,0,1,1]):
  - 규모: 100,352 step, 1,199초(초당 약 84 step, 처음 몇 분은 LBF 시험과 CPU를 같이 썼다).
  - 학습 중 100 step episode의 agent당 보상: update 1–10 .311, 45–54 .372, 89–98 .468. 자기 종류 덮기는 agent당 .31에서 .47로 늘었다. 다른 종류 덮기는 .36에서 .43으로 줄지 않았는데, 다른 종류를 덮어도 비용이 없기 때문이다.
  - 결정적 평가(10 episode), agent 평균 보상: 51k .575, 100k .675. 100k에서 episode당 자기 종류 덮기 합계 2.7, 다른 종류 2.3.
  - 해석: 학습은 느리게 오른다. 다른 종류 목표도 계속 덮으므로 대조 경험은 LBF보다 풍부할 수 있다. seed 1개, 100k라 결론은 아니다.
  - 비용: 2M이면 job당 약 6.6시간이다.

## E97. LBF 학습 중 식별(1M)과 RWARE 본 실험 seed 1 최종(2M) 분석 — 2026-11-06 KST Claude Code (T238)

- 실행 여부: 읽기 전용 분석. 학습·코드는 바꾸지 않았다. 근거: 로컬 `LBF/runs/lbf_id_1m/*`(metrics, gate, eval), W&B `pgct-lbf` 두 run(84jyrm9w, 4a1scxhq), W&B `rware-main-amax-v2` 두 run(yidd96p3, 57ebqe5l). W&B 요약값은 로컬 로그와 같았다.
- **LBF `lbf_id_1m`** (사용자 실행):
  - 조건: main_4p2t, seed 1, 숨은 팀 [0,0,1,1], 1,000,448 step. 결정적 평가는 20 episode × 최대 50 step. 이 PC(DJKLSFJ).
  - IPPO와 PGCT-ID-only의 최종 agent 가중치는 완전히 같다. gate는 학습에 개입하지 않는다.
  - 식별(PGCT-ID-only):
    - 네 명 모두 warm-up 끝(102k) 전에 성숙했다. 첫 올바른 승인은 117,760 step이고 같은 팀 두 쌍이다.
    - 이후 1M까지 precision·recall 1.0을 유지했고 오승인은 0이다. AUROC 1.0이다.
    - 175회 비교 각각에서 같은 팀 D는 .06–.14(평균 .095), 다른 팀 D는 .94–1.00(평균 .985)이다. 임계값을 .14–.93 사이 어디에 두어도 결과가 같다.
    - 모든 쌍이 매 update 접촉했다. 비교는 전부 LOAD 층에서 64 probe로 했다.
  - 학습(IPPO), agent 평균 평가 보상: 51k .019, 154k .130, 256k .175, 563k .223. 이후 1M까지 .209–.231에서 정체했다.
  - 1M 평가 세부: 먹이 6개 중 3.75개를 모은다. 다른 팀 먹이는 .25개, 적재 실패는 58회/episode다. 팀 보상(최대 1)은 .44와 .49다. 같은 팀 협력이 여전히 병목이다.
  - 대조 경험: 1M 직전 학습 중에도 agent당 다른 팀 먹이 적재가 .08–.10회/episode, 적재 실패가 4–6회/episode 남아 있다. 모든 agent가 같은 상황에서 두 종류 먹이를 다 겪는다. Overcooked의 조건 3 실패가 LBF에서는 생기지 않았다.
  - 속도: 46 step/s, job당 6.1시간이다. 시험 학습(206 step/s)보다 rollout(8.6초 대 1.9초)과 PPO(12.5초 대 2.9초)가 똑같이 약 4.3배 느렸다. gate 비용은 update당 .2–.34초(약 1.5%)다. RWARE는 다른 PC(DESKTOP-03ASKH8)에서 돌았으므로 원인이 아니다. 원인은 확인하지 못했다(전원·팬 설정이나 다른 프로세스 가능성). 이 속도라면 5M은 job당 약 30시간이다.
- **RWARE `rware-main-amax-v2`** (b6_corner_indiv_comm5_pen005, PGCT-D2 = acv2 action_max, 거울 없음, Max-GAE):
  - 조건: seed 1, 2,000,896 step, DESKTOP-03ASKH8. 결정적 평가는 20 episode × 500 step. 팀은 team0 = A1·A3·A5, team1 = A2·A4·A6.
  - 최종 평가(eval/final): 배송/episode는 IPPO 6.85, PGCT 8.55(+25%)다.
    - agent별 20 episode 합계: IPPO [53,23,18,21,0,22], PGCT [54,0,32,28,0,57].
    - 팀 합계: IPPO 71·66, PGCT 86·85.
  - 마지막 주기 평가(1,996,800): IPPO 6.55, PGCT 9.45.
  - 첫 배송: PGCT 819,200(.35/episode), IPPO 1,433,600(1.1/episode).
  - 평가 39점 평균 배송/episode(학습곡선 면적 대용): IPPO 1.24, PGCT 3.19(2.6배). 1.5M 이후 10점 평균은 4.61 대 7.88.
  - IPPO가 막판에 따라붙어 격차가 줄고 있다.
  - gate(W&B update 기록 1,954개):
    - 다른 팀 승인과 다른 팀 donor arc는 전 구간 0이다.
    - 성숙 순서: A3 124,928 → A4 130,048 → A6 150,528 → A1 171,008 → A5 229,376. 첫 donor arc는 163,840이다.
    - 0.5M 이후 donor arc 8개(같은 팀 쌍 전부)와 receiver 5명을 유지했다. select_frac은 약 .62, adv_shift는 약 .32다.
    - A2는 끝까지 성숙하지 못했다(자기 보상 사건 없음). PGCT run에서 A2는 배송 0이고, IPPO run에서는 23이다. team1 요청을 A4·A6가 먼저 가져가는 경쟁 때문일 수 있으나 확인하지 않았다. A5는 두 run 모두 0이다.
  - 한계: seed 1개다. 판정에는 seed 2·3이 필요하다(D78).
- **Overcooked `oc_id_500k_v3_rel`**: 02:15에 KeyboardInterrupt로 멈췄다(489 update 중 16 update, 37 step/s). 결과가 없으므로 v3+상대 규칙의 학습 중 식별은 아직 검증되지 않았다.
