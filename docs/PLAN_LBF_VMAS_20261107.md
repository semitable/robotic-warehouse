# LBF·VMAS 실험 계획 (2026-11-07, T255, D144)

## 왜 다시 설계하나

- LBF main(4 agent = 2팀 × 2명, 5M) seed 2에서 방법 간 차이가 잡음 수준이었다. 정답 팀을 쓰는 Oracle도 IPPO와 같았다(E100·E102).
- 원인으로 본 것: 같은 팀 안에서 보상을 발견하는 시점이 엇갈리지 않았다(T246·T252). 2명 팀에서는 함께 드는 먹이를 둘이 동시에 발견하고, 혼자 드는 먹이는 둘 다 일찍 찾는다.
- RWARE에서 PGCT가 이긴 상황: 같은 팀 안에 앞선 agent와 뒤처진 agent가 있었고, 뒤처진 agent가 앞선 팀원의 critic을 빌려 배송을 시작했다(E95, E97).
- **가설 H**: 같은 팀 안에서 보상 경험량이 엇갈리면 PGCT-D2가 IPPO보다 빨리 배운다. 팀이 클수록 그런 엇갈림이 생기기 쉽다.

## 실험

| ID | 환경·설정 | agent | 엇갈림 메커니즘 | 방법 | seed | 예산 | 실행 위치 | 상태 |
|---|---|---|---|---|---|---|---|---|
| L1 | LBF `large_12p3t` | 12 = 3팀 × 4 | 무거운 먹이(절반)는 팀원 2–3명만 함께 든다. 낀 팀원과 못 낀 팀원이 갈린다 | pgct_d2, ippo, oracle (all·random_gate는 11-08 소음 때문에 중단, E105) | 1–3 | 5M | 이 PC, 동시 3개 | **2026-11-07 시작** |
| V1 | VMAS `main_4p2t` | 4 = 2팀 × 2 | (사전 고정 기본 설정) | 같은 5개 | 1–3 | 2M | 다른 PC 권장 | 계획 |
| V2 | VMAS `large_8p2t` | 8 = 2팀 × 4 | 혼자 덮는 목표 4개를 팀원 4명이 두고 경쟁(RWARE형) | 같은 5개 | 1–3 | 2M | V1 다음 | 계획 |
| L0 | LBF `main_4p2t` seed 1·3 재실행 | 4 | — | 같은 5개 | 1, 3 | 5M | L1 다음 | 보류(seed 2만 완료) |
| R | RWARE seed 2·3 | 6 | — | ippo, pgct_max | 2, 3 | 2M | 다른 PC | 최우선 권장 |

- L1·V2는 결과를 본 뒤 정한 **사후 변형**이므로 논문에는 보조 실험으로 밝힌다. 설정은 시작 전에 고정한다. gate·PPO·Max-GAE 값은 main과 같다(상대 보상 사건 규칙).
- 정답 팀을 쓰는 것은 Oracle과 평가·감사뿐이다.

## 계산량 (이 PC, i5-13400, job 하나 단독 측정)

| 설정 | IPPO | PGCT-D2 | 동시 5개일 때 예상 | job 하나 |
|---|---|---|---|---|
| LBF 4명 | 186 step/s | 174 | 약 140 | 5M ≈ 10시간 |
| LBF 8명 | 93 | 90 | — | — |
| LBF 12명(3×4) | 63 | 57 | 실측 40–41 (L1 시작 직후) | 5M ≈ 34–37시간 |
| LBF 16명(4×4) | 42 | 40 | — | — |
| VMAS 4명 | 69 | 69 | 약 55 | 2M ≈ 10시간 |
| VMAS 8명 | 35 | 34 | 약 28 | 2M ≈ 20시간 |

- 속도는 agent 수에 거의 반비례한다(agent마다 신경망을 따로 돌림). gate 비용은 5–10%다.
- 최대 프로세서 상태 99%(터보 끔)에서는 약 20% 더 느리다.
- 총량(15 job씩, 동시 5개, 묶음 3개): L1 약 4.5일(실측 기준), V1 약 30시간, V2 약 60시간, L0 약 20시간.
- 한 PC로 다 돌리면 8일 이상이다. 그래서 L1은 이 PC, R·V1은 다른 PC를 권장한다. 다른 PC에는 `pip install --no-deps vmas==1.5.2 gym==0.26.2 gym-notices==0.0.8`이 필요하다.

## 분석 계획 (사전에 정함)

1. **엇갈림 확인(H의 전제)**: IPPO run에서 팀 안 agent별 자기 팀 보상 사건 수(`episode_collected_own_per_agent`, VMAS `covered_own`)의 팀 내 비율을 본다. 1M 구간마다 "팀 내 최대 ÷ 최소"를 계산한다. 2배 이상 갈리는 팀이 있어야 H를 시험할 수 있다.
2. **식별(Q1)**: gate 로그로 precision·recall·오승인·첫 승인 시점을 본다. 팀이 3개(L1)이면 다른 팀 쌍이 많아진다(12명 중 같은 팀 쌍 18, 다른 팀 쌍 48).
3. **전이(Q2)**: 평가 보상의 학습곡선 면적, 마지막 20회 평균, 첫 도달 시점을 본다. 핵심은 **뒤처진 agent의 개선**이다. IPPO에서 뒤처진 위치(같은 seed·같은 숨은 팀 배정)의 agent가 PGCT-D2에서 얼마나 빨리 보상을 얻는지 비교한다. random_gate(같은 수의 무작위 donor)와 all(모든 donor)로 "올바른 팀원"의 효과를 분리한다.
4. **판정**: seed 3개 평균 ± 표준편차로 본다. seed 1 묶음(약 30시간)이 끝나면 중간 점검하고, 엇갈림이 전혀 없으면 사용자와 상의해 중단 여부를 정한다.

## 실행 명령 (저장소 루트, PowerShell)

```powershell
# L1 (시작함)
powershell -ExecutionPolicy Bypass -Command "& '.\scripts\start_detached.ps1' -Experiment LBF -Name lbf_large12_5m -Config large_12p3t.json -Methods pgct_d2,ippo,oracle,all,random_gate -Seeds 1,2,3 -MaxConcurrent 5"
# V1
powershell -ExecutionPolicy Bypass -Command "& '.\scripts\start_detached.ps1' -Experiment VMAS -Name vmas_main_2m -Config main_4p2t.json -Methods pgct_d2,ippo,oracle,all,random_gate -Seeds 1,2,3 -MaxConcurrent 5"
# V2
powershell -ExecutionPolicy Bypass -Command "& '.\scripts\start_detached.ps1' -Experiment VMAS -Name vmas_large8_2m -Config large_8p2t.json -Methods pgct_d2,ippo,oracle,all,random_gate -Seeds 1,2,3 -MaxConcurrent 5"
# 집계
powershell -ExecutionPolicy Bypass -File scripts\aggregate_experiment.ps1 -Experiment LBF -Name lbf_large12_5m
```

- 주의: 앱 밖(숨김) 실행도 **PC 재시작에는 멈춘다**(E103). Windows 업데이트 자동 재시작을 미뤄 둔다. 멈춘 job 폴더는 지우지 말고 `_crashed_<날짜>`로 옮긴 뒤 같은 명령으로 다시 띄운다.

## F1. 전문가 추종 실험 (2026-11-08 추가, T259, D145)

- 질문: 이미 학습을 끝낸 agent가 팀에 있을 때, 같은 팀의 새 agent(후발)가 그 agent의 critic을 빌려 얼마나 빨리 따라오나. RWARE의 leader→lagger 상황(E95)을 일부러 만들어 직접 잰다.
- 설정 `LBF/configs/follow_12p3t.json`: `large_12p3t`(12명 = 3팀 × 4)에서 팀 배정을 L1 seed 1과 같게 고정(team_seed 1).
  - agent 11(숨은 팀 0)을 L1 IPPO seed 1의 5M 학습 agent 11로 교체하고 학습하지 않게 고정한다(`init_checkpoint`, `init_agents`, `frozen_agents`).
  - 후발: 같은 팀 0의 agent 5·6·8. 비교군: 팀 1·2의 8명(전문가 없음). 같은 run 안에 비교군이 있다.
- 방법: IPPO(빌리지 않음), PGCT-D2(gate가 고른 팀원에게서 빌림), Oracle(정답 팀원에게서 빌림). 선택으로 All(다른 팀 전문가 critic까지 빌림, 잘못된 팀원의 영향 확인).
- 예산: 2M, seed 1–3(학습 seed만 다름), 동시 3개. job당 약 9시간, 3묶음이면 약 28시간이다.
- 지표(`python -m pgct_lbf.follow --plan LBF/runs/<name>`): 25만 step 구간마다 후발·비교군의 agent당 자기 팀 보상 사건 수, 전문가 수준의 50%·80% 도달 시점.
  - 핵심 비교: (PGCT-D2 후발 − IPPO 후발)과 (Oracle 후발 − IPPO 후발).
  - IPPO에서 후발이 비교군보다 빠른 정도는 환경 효과(전문가와 함께 들기, 먹이 경쟁)다. 방법 간 차이만 critic 전이 효과로 본다.
- 전제: warm-up 100 update 동안은 빌리지 않는다(다른 실험과 같음). 전문가 critic은 전문가 자신의 정책 가치라서, 후발의 상태를 전문가 기준으로 평가해 준다.
- 시작 조건: L1 IPPO seed 1의 `final.pt`(11-08 저녁 생성). L1 seed 2·3과 같은 PC를 쓰므로 순서 결정이 필요하다.
