# A+C gate 오프라인 진단 — 2026-10-26 Claude Code

학습 없음. 저장된 checkpoint를 확률 정책으로 굴려 얻은 경험만 사용했다. 팀 라벨은 채점에만 썼다.

## 파일

| 파일 | 내용 |
|---|---|
| `collect.py` | checkpoint 롤아웃 → agent별 공용 관측(before/after, 반경 2)·행동·자기 보상 + 채점용 숨은 라벨 |
| `rollouts.npz` | A+C IPPO s1(좌표 마스킹), 0.26M–5M checkpoint 7개 × 8 episode |
| `rollouts_{corner,adjacent}_indiv_1M_s{1,2}.npz` | 옛 1M IPPO(좌표 있음), checkpoint 4개 × 8 episode |
| `analyze.py` → `results.json` | 관측 가능성, 경험 커버리지, 정확 일치·거친 key 지지, 일반화 MLP 일괄 측정 |
| `details.py` → `details.json` | 배송 key의 보상 혼합, MLP 사건 probe 거리 행렬 |
| `divergence.py` | A+C PGCT s1과 IPPO s1이 처음 달라지는 update |
| `gate_eval.py` → `gate_eval.json` | 실제 v1·v2 gate 클래스에 경험을 시간 순서로 흘려 넣은 판정 비교 |

## 주요 결과

- 첫 전이(2,821,120 step, update 2755) 전 2,754 update에서 PGCT s1과 IPPO s1의 return·loss·entropy가 비트 단위로 같다.
- 모서리 배치에서 두 goal 배송 장면의 공용 관측은 겹치지 않는다. agent는 다른 팀 goal에 선반을 31–136번 들고 갔고 보상은 ≈ 0이다(agent 1 제외).
- v1 배송 key 278개 중 표본이 충분한 17개의 보상 비율 중앙값 .45, std .45. std ≤ .10은 1개.
- 일괄 MLP(모든 경험): 사건 probe AUROC 1.00, 대칭 probe .91, 1.5M까지 경험만 쓰면 .67.

### gate_eval (모든 쌍 접촉 가정, 1,000 env step마다 round)

| 데이터 | v1 판정 정확도 (결정/15) | v1 승인 같은/다른 | v2 판정 정확도 (결정/15) | v2 승인 같은/다른 |
|---|---|---|---|---|
| A+C IPPO s1 0.26–5M | .63 (8) | 1 / 0 | **1.00 (10)** | **4 / 0** |
| corner 1M s1 | .25 (8) | 0 / 0 | 보류 (0) | 0 / 0 |
| corner 1M s2 | .50 (12) | 0 / 0 | 보류 (0) | 0 / 0 |
| adjacent 1M s1 | .60 (15) | 0 / 0 | 보류 (0) | 0 / 0 |
| adjacent 1M s2 | .56 (9) | 0 / 0 | 보류 (0) | 0 / 0 |

- v2 거리(A+C 5M): 같은 팀 .14–.18, 다른 팀 .58–.70. 보류 5쌍은 모두 배송 2회인 agent 1이 낀 쌍이다.
- v2 첫 승인은 2.56M checkpoint 경험 구간(round 15). 다른 팀 승인은 모든 round에서 0.
- 대칭 probe를 끈 v2도 결과가 같다.
- 1M 모델은 배송 agent가 0–2명이라 v2가 성숙하지 않아 모두 보류한다. v1은 결정하지만 정확도 .25–.60, 거리 AUROC .30–.42다.

## 한계

seed 1 run 하나(5M), 15쌍. 실제 학습 경험의 약 1/180만 재생했다. 모든 쌍이 매 round 접촉한다고 가정했다. 인접 배치는 배운 모델이 없어 조건 (가)의 음성 사례를 검증하지 못했다. v1의 가상 질의(A)는 전체 관측이 필요해 재생하지 않았다.
