# From-scratch 핵심 비교(E17) 중단 시점 분석 — 약 2.5M step

2026-10-15 Claude Code. 읽기 전용 분석이며 학습·로그·checkpoint를 변경하지 않았다. 재현: `python output/scratch_review_20261015/analyze.py`. 수치: `analysis.json`, 그림: `curves.png`.

## 조건과 한계

- **run**: `runs/scratch_20261013_202352_stage2`(IPPO/PGCT/Oracle/All, 무작위 초기화, 같은 초기 가중치).
- **환경**: E1 11×10, home .5, individual reward, 4 agents/2 teams.
- **평가**: deterministic masked policy, seeds 5,000,001–5,000,020, 20×500 step, 51,200 step마다.
- **중단**: 2026-10-15 01:01 KST Windows Update 재부팅. 마지막 기록은 IPPO 2,487,296 / PGCT 2,502,656 / Oracle 2,471,936 / All 2,455,552 step.
- **공통 비교 구간**: 51,200 ~ 2,406,400 step(All의 마지막 평가). AUC는 이 구간의 사다리꼴 평균이다.
- **한계**: training seed가 1개이고 학습 초기 구간만 있다. 평가값의 인접 평가 간 차이 표준편차는 0.2–0.8 delivery다. 아래 차이는 **방향 관찰**이지 효과 검증이 아니다.

## 1. 평가 성능

| mode | 마지막 3회 평균 delivery | 요청 선반 pickup | return | full cycle 성공률 | delivery AUC/step | pickup AUC/step |
|---|---:|---:|---:|---:|---:|---:|
| IPPO | 3.10 | 20.68 | 1.12 | .58 | 0.41 | 4.25 |
| PGCT | 1.92 | 2.20 | 0.98 | .62 | 0.26 | 3.48 |
| **Oracle** | **9.98** | 10.42 | **9.43** | **1.00** | **1.28** | 2.51 |
| All | 3.45 | 15.15 | 1.37 | .65 | 0.69 | 3.89 |

처음 오른 시점(3점 이동평균 기준):

| mode | pickup ≥1 | pickup ≥10 | delivery ≥1 | delivery ≥5 |
|---|---:|---:|---:|---:|
| IPPO | 1.89M | 1.95M | 2.05M | 미도달 |
| PGCT | 1.89M | 2.00M | 2.15M | 미도달 |
| **Oracle** | **1.64M** | **1.69M** | **1.89M** | **2.25M** |
| All | 1.79M | 1.95M | 1.89M | 미도달 |

- 모든 mode에서 **0.5–1.5M은 보상이 거의 0인 정체 구간**이었다. train delivery는 0.002–0.006/episode였다. 학습은 1.6–1.9M부터 시작했다. 따라서 2M 시점 성능만으로 판단하면 안 된다는 기준이 맞았다.
- IPPO와 All은 pickup은 많지만(15–21) delivery로 이어지는 비율이 낮다. pickup conversion은 .11–.12다. Oracle은 pickup 10.4 → delivery 10.0 → return 9.4로 한 사이클을 완성했다.
- 참고(조건이 다름): 이전 from-scratch PGCT(home 1.0, complete graph)는 같은 2.4M에서 평가 delivery 26.7이었다. home .5가 더 어렵거나 학습이 느리다는 신호지만, 통신 그래프도 달라 단정하지 않는다.

## 2. 가장 중요한 관찰: 성능이 agent 1–2개에서만 나온다

마지막 3회 평가의 agent별 delivery/episode:

| mode | Agent1 (팀0) | Agent2 (팀1) | Agent3 (팀0) | Agent4 (팀1) |
|---|---:|---:|---:|---:|
| IPPO | 1.63 | 1.47 | 0 | 0 |
| PGCT | 1.92 | 0 | 0 | 0 |
| **Oracle** | 0 | **9.98** | 0 | 0 |
| All | 0 | 2.12 | 1.33 | 0 |

- Oracle의 우위는 **Agent2 한 명이 배송 사이클을 먼저 완성한 결과**다. 팀원 Agent4를 포함한 나머지 3명은 아직 배송하지 못한다.
- 한 agent가 언제 "깨치는지"는 seed와 우연의 영향이 크다. 따라서 이 우위를 Oracle consensus의 효과로 바로 귀속할 수 없다. 다른 seed에서 같은 패턴이 반복되는지 확인해야 한다.
- 가능한 연결고리(가설): Oracle은 Agent2와 Agent4가 critic을 공유한다. 앞으로 Agent4의 학습이 Agent2보다 빨라지는지가 같은 팀 공유 효과를 보는 직접적인 관찰점이 될 수 있다.

## 3. 실제 critic 전이

| mode | warm-up 후 전이가 있던 update 비율 | 같은 팀 arc-update | 다른 팀 arc-update | 평균 gate ratio |
|---|---:|---:|---:|---:|
| IPPO | 0% | 0 | 0 | 0 |
| PGCT | 77% | 3,496 | 5,590 | .59 |
| Oracle | 76% | 4,688 | **0** | .27 |
| All | 99% | 5,526 | 10,524 | .95 |

- 전이는 모두 물리 이웃 범위 안에서만 일어났다. 설계대로 Oracle은 같은 팀(1–3: 1,145, 2–4: 1,199 update)만, All은 모든 쌍에 전이했다.
- **PGCT의 gate는 같은 팀과 다른 팀을 거의 구분하지 않았다.**
  - 같은 팀 1–3: 826, 2–4: 922
  - 다른 팀 1–2: 808, 2–3: 886, 1–4: 601, 3–4: 500
  - 학습 초기에는 정책들이 비슷해서 JS gate가 넓게 열린다. 이는 실험 A의 "공통 입력 JS는 팀을 구분하지 못한다"는 결과와 일치한다.
- peer loss 크기(1.5M 이후 평균): PGCT .0010, All .0017, Oracle .0039.

## 4. 학습 지표

- **entropy**: 0.0–0.5M 약 .91–.92 → 1.5M 이후 IPPO .83, PGCT .83, Oracle .76, All .89. Oracle이 가장 빨리 결정적인 정책으로 수렴했다.
- **value loss**: 1.5M 이후 .002–.006 수준이다. 보상이 생긴 뒤 커졌고, Oracle이 가장 크다(.0062).
- **approx_kl**: .0028 근처로 모두 안정적이다.
- **속도**: job당 약 24 step/s(4개 병렬).

## 5. 결론 (현 시점)

1. 무작위 초기화에서도 네 mode 모두 1.6–1.9M부터 학습이 시작됐다. 학습 가능성은 확인됐다.
2. 2.4M 시점에는 Oracle이 가장 앞서 있다(delivery AUC 1.28 vs All .69, IPPO .41, PGCT .26). pickup과 delivery 시작도 가장 빠르다.
3. 그러나 그 우위는 **agent 한 명의 선행 학습**에서 나왔다. seed가 1개라 Oracle 효과로 결론 내릴 수 없다.
4. PGCT gate는 초기에 팀을 구분하지 않는다. 지금의 PGCT는 "잠재 팀 기반 공유"의 대리 지표가 아니다.
5. 해석 원칙은 유지한다. 이 실험은 critic 공유 효과의 검증이며, 팀 목표가 대칭인 환경이라는 점을 감안한다.
6. **다음 단계 제안**: 처음부터 재실행(15M)하고, 가능하면 seed 2·3을 함께 돌린다. agent별 학습 시작 시점(특히 Oracle에서 같은 팀원의 뒤따라오기)을 핵심 관찰 지표로 추가한다.
