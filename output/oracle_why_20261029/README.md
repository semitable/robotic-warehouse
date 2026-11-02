# Oracle s1이 IPPO보다 빠르지 않은 이유 — 읽기 전용 분석 (2026-10-29, Claude Code, T147)

학습 없음. 대상: `runs/ac_20261023_185556`(A+C v1, 모서리·개인 보상, 좌표 마스킹, seed 1). IPPO·Oracle·PGCT는 complete(5M), All은 1.6M에서 진행 중.

| 스크립트 | 내용 |
|---|---|
| `oracle_why.py` | 평가 배송 곡선, 수준 도달 뒤 기울기, peer loss 대 value loss, `transfer.jsonl` 실제 allocation, `probe/cross_team_gate` |
| `oracle_vgap.py` | canonical probe 48개(좌표 0)에서 팀원 critic 차이(V_i − V_j)² |
| `oracle_realgap.py` | 각 mode의 정책으로 3 episode를 굴려, 실제 방문 상태에서 팀원 critic 차이와 내 return 대비 MSE 계산 |
| `oracle_js.py` | probe에서 정책 JS(같은 팀·다른 팀)와 KL(π‖U) |

## 결과 (seed 1)

팀 평가 배송/episode(20 episode, 평가 5회 이동평균)

| mode | 0.5M | 1M | 1.5M | 2M | 3M | 4M | 4.5M |
|---|---|---|---|---|---|---|---|
| IPPO | 0.0 | 2.4 | 5.7 | 8.1 | 11.6 | 17.0 | 21.7 |
| Oracle | 0.0 | 0.4 | 5.2 | 7.4 | 10.6 | 17.4 | 21.5 |
| PGCT v1 | 0.0 | 2.4 | 5.7 | 8.1 | 11.7 | 17.9 | 20.5 |
| All | 0.0 | 1.5 | 6.5 | – | – | – | – |

수준 L에 처음 도달한 뒤 L+5까지 걸린 step(M)

| L | IPPO | Oracle | PGCT v1 |
|---|---|---|---|
| 5 | 1.13 | 1.43 | 1.13 |
| 10 | 0.92 | 0.92 | 1.18 |
| 15 | 0.87 | 0.46 | 0.56 |

PGCT v1은 2.82M까지 IPPO와 같고, 그 뒤 전이 22 update만으로 갈라졌다. 따라서 IPPO와 PGCT의 차이(15→20에서 0.87 대 0.56M)가 seed 하나에서 나타나는 혼돈 수준의 차이다.

팀원 critic 차이 (같은 팀 쌍 평균 MSE)

| 시점 | probe: IPPO | probe: Oracle | 실제 상태: IPPO | 실제 상태: Oracle |
|---|---|---|---|---|
| 0.5M | .0100 | .0003 | – | – |
| 1M | .0596 | .0015 | .080 | .023 |
| 2M | .8698 | .0060 | .674 | .115 |
| 4.75M | 1.2053 | .0048 | 1.194 | 1.223 |

- probe 위 |V| 평균(1M): IPPO .140, Oracle .064.
- 실제 상태에서 내 할인 return에 대한 MSE(IPPO): 2M에 자기 critic .18, 팀원 critic .74. 4.75M에 자기 critic .83, 팀원 critic 1.79.
- IPPO에서 실제 상태의 같은 팀 critic 차이는 다른 팀 차이보다 작지 않다(2M에 .67 대 .44).
- peer loss×.01은 value loss×.5의 0.2–2%다. 다만 probe 차이가 이미 작게 유지되고 있어서 생기는 값이다(위 표).

정책 JS(probe, IPPO): 0.25M에 같은 팀 .038·다른 팀 .056, 1M에 .185·.189, 4.75M에 .217·.248. Oracle도 비슷하다. KL(π‖U)은 1M에 배송한 agent 3·4·6이 .79·1.34·.60, 못 한 1·2·5가 .50·.48·.56이다.

`probe/cross_team_gate`(PGCT, 평균 .277)는 통신 범위 안 다른 팀 쌍의 비율(4.98/18 = .277)과 같다. `same_team_gate` .338도 4.05/12와 같다. A+C PGCT는 trainer gate를 `unrestricted`로 두고 controller가 실제 allocation을 덮어쓰기 때문이다(`latent_team/train_from_scratch.py:171`). 실제 v1 allocation은 같은 팀 48 arc, 다른 팀 0 arc다.

## 전이 재설계 전제 점검 (T148, E62)

`transfer_premise.py` → `transfer_premise_stdout.txt`. IPPO s1의 1M·2M·3M에서 3 episode씩, 받는 agent의 실제 방문 상태에서 6개 critic을 모두 recurrent로 돌렸다.

- 상향폭 mean relu(max_j V_j − V_i), 같은 팀 / 다른 팀. 뒤처진 agent(배송 0): 1M .35/.34, 2M 1.19/.72, 3M 1.07/1.12. 앞선 agent: .03/.15, .13/.56, .18/.25.
- 운반 중이고 goal이 시야 밖일 때 critic과 내 return의 상관(자기 / 같은 팀 / 다른 팀): 1M .56/.65/.56, 2M .59/.66/.53, 3M .39/.48/.50.
- 내 return 대비 편향(자기 V / max(자기, 같은 팀)): +.03/+.25, +.27/+.76, +.13/+.45.

## D91 추가 점검 (T149, E63)

- `max_advantage_check.py` → `max_advantage_check_stdout.txt`: h=0·L8 질의 정확도, 앞선 팀원의 최선 행동에 대한 advantage 선호(상관 지표).
- `counterfactual_check.py` → `cf_*.json`, `cf_*_stdout.txt`: env deepcopy로 상태를 복원하고, 받는 agent 자리에서 팀원 정책을 실행한 실제 return 차이(dMC)를 critic 신호 dV = V_j(h_i) − V_i(h_i)와 비교. 행동별 one-step A_self·A_max를 MC Q와 비교. V2 성숙(E60 replay) + Chebyshev 3 통신 제한. CRN, N=8, H=100(상태)·60(행동), episode 3, τ = 80·180·280.
- `summarize_cf.py` → `cf_summary_stdout.txt`, `cf_summary.json`.
- 결과 요약은 `EXPERIMENTS.md` E63에 있다.

## 한계

seed 하나다. 실제 상태 분석은 mode·시점마다 3 episode이고, 정책이 mode마다 달라 방문 상태 분포도 다르다. Oracle의 1M |V|가 낮은 것은 원인일 수도 있고 결과(배송이 적음)일 수도 있다.
