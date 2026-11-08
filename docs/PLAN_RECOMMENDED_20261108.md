# 추천 실험 계획 (2026-11-08, Claude Code, T260, D146)

## 지금 주장할 수 있는 것과 약한 곳

| 질문 | 근거 | 강도 |
|---|---|---|
| Q1 식별: 라벨 없이 같은 팀을 찾는가 | RWARE seed 1(2M, 다른 팀 승인 0, 6명 중 5명 성숙). LBF 4명 seed 1·2(완벽). LBF 12명·3팀 seed 1(같은 팀 18쌍 전부, 다른 팀 48쌍 0). Overcooked는 분담형에서 실패(경계 사례) | **강함**(환경·규모가 달라도 성립) |
| Q2 전이: 팀원 critic을 빌리면 빨리 배우나 | RWARE seed 1만: 첫 배송 0.82M 대 IPPO 1.43M, 평가 평균 2.6배, 최종 +25% | **약함**: seed 1개, RWARE에 Oracle·무작위 donor 대조 없음 |
| 적용 조건 | LBF(4명·12명)는 팀원이 고르게 배워 Oracle도 IPPO와 같음 | 관찰만 있고 직접 시험은 없음 |

논문의 중심 주장은 Q2인데, 근거가 가장 약하다. 그래서 새 환경을 늘리기보다 **Q2를 확실하게 만드는 것**이 가장 가치 있다.

## 추천 1 (최우선): RWARE 본 비교 완성 — 다른 PC

- 설정: seed 1과 같은 `b6_corner_indiv_comm5_pen005`(11×10, 6명 = 2팀 × 3, 개인 보상, 통신 5), acv2 action_max, 거울 없음, 2M.
- 방법 4개:
  - `ippo`
  - `pgct_max`(PGCT-D2)
  - `oracle_max`: 정답 팀원 donor, 상한
  - `allk_max`: 만난 agent 중 무작위로 Oracle과 같은 수만큼 donor. "맞는 팀원이어서인지, critic을 더 받아서인지"를 가르는 대조
- seed 1–3. seed 1의 ippo·pgct_max는 이미 있으므로 **10 job**이다. 계산 여유가 있으면 ippo·pgct_max만 seed 4·5를 더한다. RWARE는 첫 배송 시점이 seed마다 크게 흔들려서 seed 수가 결론을 좌우한다.
- 명령(다른 PC, RWARE 폴더):
  - `run_main_amax_v2.ps1 -Candidate b6_corner_indiv_comm5_pen005 -Seeds 2,3 -Modes pgct_max,ippo,oracle_max,allk_max -TotalTimesteps 2000000`
  - seed 1은 `-Seeds 1 -Modes oracle_max,allk_max`
- 시간: RWARE 학습기는 job당 초당 약 25 step이다(다른 PC 실측. 이 PC는 LBF와 함께 돌릴 때 약 11 step). 2M이면 job당 약 22시간이고, 동시 4개면 약 3일이다.
- 사전 판정 기준:
  1. 학습곡선 면적(평가 배송 평균)에서 PGCT-D2 > IPPO가 seed 3개 중 3개, 또는 평균 차이가 seed 간 표준편차보다 크다.
  2. 첫 배송 시점이 앞당겨진다.
  3. PGCT-D2 ≥ allk_max: 맞는 팀원이 중요하다.
  4. PGCT-D2가 Oracle의 상당 부분을 회복한다.
- 분석(추가 계산 없음): 같은 seed의 IPPO와 PGCT에서 agent별 첫 배송 시점을 비교한다. 실행 호스트의 `max_gae.jsonl`로 "어느 donor가 어느 receiver의 advantage를 바꿨는지"를 잇는다. 이것이 Codex가 본 A4←A6 사례(E95)를 seed 3개로 일반화하는 근거가 된다.

## 추천 2: 전문가 추종 실험 F1 — 이 PC (이미 준비됨)

- 사용자 아이디어지만, 제가 보기에 **전이 효과의 인과 근거로 가장 깔끔한 설계**라 추천한다. 팀 안 격차를 실험으로 직접 만들고, 같은 환경·같은 전문가에서 critic 접근만 바꿔 비교하기 때문이다.
- 설정: `LBF/configs/follow_12p3t.json`(12명 = 3팀 × 4). 고정 전문가는 agent 11(팀 0, L1 IPPO seed 1의 5M agent)이다. 후발은 같은 팀 3명, 비교군은 다른 팀 8명이다.
- 방법 조정 제안: **IPPO, PGCT-D2, All**.
  - LBF 12명에서는 식별이 완벽했다(E106). 그래서 PGCT-D2의 donor는 첫 승인(약 0.12M) 이후 Oracle과 같아지고, Oracle은 거의 중복이다.
  - 대신 All을 넣으면 **다른 팀 전문가의 critic이 다른 팀 agent 8명을 어떻게 흔드는지**(잘못된 donor의 해)를 같이 본다. 식별이 왜 필요한지 보여 주는 직접 근거가 된다.
- 2M × seed 1–3, 동시 3개다. job당 약 9시간, 전체 약 28시간이다. L1 seed 1이 끝나는 오늘 밤 이후에 시작할 수 있다.
- 판정:
  - (PGCT-D2 후발 − IPPO 후발)이 seed 3개에서 양수다.
  - All에서는 비교군(다른 팀)이 IPPO보다 나빠지거나 이득이 없고, PGCT-D2에서는 비교군이 IPPO와 같다.
  - 후발 보고서는 `python -m pgct_lbf.follow`로 낸다.

## 추천 3: 계산 없는 정리 분석

- **적용 조건 지표**: IPPO run에서 "같은 팀 안 학습 시점의 엇갈림"을 계산한다. agent별 첫 보상 도달 시점의 팀 내 차이나, 구간별 보상 사건의 팀 내 최대/최소를 쓴다. 이 값을 PGCT 이득과 나란히 놓는다.
  - RWARE: 엇갈림이 크고 이득이 있다.
  - LBF 4명·12명: 엇갈림이 작고 이득이 없다.
  - F1: 엇갈림을 만들었을 때 이득이 있는지 본다.
  - 이것이 "PGCT는 언제 효과적인가"에 대한 논문의 설명이 된다.
- **한계 정리**: 자기 보상 사건이 전혀 없는 agent(RWARE A2)는 성숙하지 못해 도움도 받지 못한다. 방법의 알려진 한계로 명시한다.

## 보류·중단 권고

| 항목 | 이유 |
|---|---|
| LBF 12명 seed 2·3 | 식별은 seed 1로 충분히 보였고, 엇갈림이 없어 전이 시험 의미가 작다. 11-08 사용자 요청으로 중단 |
| LBF 4명 seed 1·3 재실행 | 결과 폴더 소실로 seed 2만 남았다. Q1 보조 근거로는 W&B 지표와 seed 1 식별 1M으로 충분하다. 필요하면 나중에 IPPO·PGCT만 돌린다 |
| VMAS V1·V2 | 느리고(4명 69 step/s, 8명 35 step/s), 엇갈림이 생길지 불확실하다. 추천 1·2 결과를 본 뒤 결정한다 |
| Overcooked 추가 실험 | 경계 사례로 정리 완료 |

## 기계 배치

| PC | 지금 | 다음 |
|---|---|---|
| 다른 PC(RWARE) | — | 추천 1, 약 3일 |
| 이 PC(DJKLSFJ) | L1 seed 1(오늘 밤 종료) | 추천 2(F1), 약 28시간, 동시 3개 |

- 두 PC 모두 Windows 업데이트 자동 재시작을 미뤄 둔다(E103).

## 기계별 배치 (2026-11-08 추가, T261, E108)

측정(이 PC, LBF L1 job 3개와 동시 실행 중, 짧은 측정):

| 학습기 | CPU만 | GPU 사용 | 결론 |
|---|---|---|---|
| RWARE 학습기(`examples/train_recurrent_ippo_consensus.py`, device auto) | 10.0 step/s | 13.7–14.7 step/s(RTX 3060) | GPU 이득 약 1.4배. 대부분 CPU(환경·Python 루프)에 묶임 |
| pgct_core 학습기(LBF 4명) | 128 | 44 | GPU가 약 3배 느림(작은 신경망, step마다 장치 왕복) |
| pgct_core 학습기(LBF 12명) | 46 | 15 | 같음 |

- 참고: RWARE seed 1은 DESKTOP-03ASKH8(20코어, RTX 5070 Ti, device auto)에서 job당 25–27 step/s였다.
- 따라서 지금 모든 실험은 **CPU 코어 성능과 코어 수**가 속도를 정한다. 좋은 GPU를 거의 쓰지 못한다.

| 기계 | 배치 | 이유 |
|---|---|---|
| 연구실 PC(RTX 5080, CPU가 강하다고 가정) | **추천 1: RWARE 본 비교** 10 job(seed 2·3 × ippo·pgct_max·oracle_max·allk_max, seed 1 × oracle_max·allk_max), 동시 4–5개 | RWARE는 job당 CPU 1코어 + GPU 일부를 쓴다. 강한 CPU와 GPU가 모두 도움이 된다. 약 2–3일 |
| 이 PC(RTX 3060, i5-13400) | **추천 2: F1 전문가 추종**(ippo·pgct_d2·all × seed 1–3, 2M), 동시 3개 | pgct_core는 CPU 전용이다. 소음 때문에 3개로 둔다. 약 28시간 |
| 서버(GPU 강함, CPU 약함) | 먼저 10분 RWARE 속도 측정. job당 15 step/s 이상이면 RWARE ippo·pgct_max seed 4·5. 아니면 계산 분석(오프라인 gate 재생, 영상, 집계) | 지금 학습기는 GPU를 거의 못 쓴다. 약한 CPU에서는 job이 느리다 |

- 서버의 GPU를 제대로 쓰려면 환경 여러 개와 agent들을 한 번에 묶어 GPU로 계산하는 **벡터화 학습기**가 필요하다. VMAS는 GPU 병렬 환경을 기본으로 지원해서 가장 큰 이득을 본다. 다만 새 학습기라 기존 결과와 수치가 이어지지 않고, 개발·검증에 1–2일이 든다. 추천 1·2보다 우선순위가 낮다.
- 서버가 Linux면 PowerShell 스크립트 대신 Python 명령을 직접 쓴다. 예: `python -m pgct_lbf.train ...`, RWARE는 `latent_team/train_from_scratch.py --plan ... --run-job ...`. EcoQoS 문제는 Windows에서만 생긴다.
