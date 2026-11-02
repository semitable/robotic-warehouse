# PGCT 현재 연구 상태

## 최신 확인 — 2026-11-02 KST, Claude Code (T164, D98 필수 수정 1–4 구현)

- **구현(D99, E67)**: 분기 goal 배치 검사, D2 `shrink` 척도(`--max-gae-scale` 기본값 relative → shrink), rollout 첫 7 step Δ = 0, agent별 critic explained variance 로그. pytest 57개 통과. smoke에서 donor를 끈 동등성(비트 단위), 1M 분기, seed 2 거부를 확인했다.
- **분기 실행 명령**은 T164 최종 답변에 있다(seed 1·3·5, `--ac-gate v2`, 4 모드, 12 job, 사용자 실행). 아직 시작하지 않았다.
- **남은 것**: PGCT-D2(필수 5–8), donor critic 성숙 조건, 보상 사건이 없는 agent(a0·a4) 처리.

## 최신 확인 — 2026-11-02 KST, Claude Code (T163, PGCT-D2 통합 전 코드 검토)

- **PGCT-D2는 미구현이다.** Max-GAE 모드에서 v2는 돌지 않고(learn=False), donor는 정답 라벨 ∧ rollout 접촉뿐이다. v2 승인을 donor로 잇는 코드(`approved()`, donor 규칙 gate, `pgct_max`)가 필요하다(D98).
- **T154의 1M 분기 명령은 그대로 실행하지 않는다.** seed 2는 goal 좌우가 바뀐다(`env_candidates.py:104`). D2 기본 `relative` 척도는 작은 b를 키워 IPPO로 연속 퇴화하지 않는다. rollout 첫 7 step 가림이 양의 편향을 만든다(E66).
- **성숙 조건 충돌**: A+C IPPO s1에서 a0·a4는 1–1.5M에 배송·반납이 0이다. v2 규칙으로는 receiver가 될 수 없다.
- `runs/`에 `maxgae_br1M_*` 폴더가 없다(프로세스는 확인하지 않음). 학습·코드 변경 없음.
- **미결(사용자 결정)**: 필수 수정 구현 범위, D2 donor 접촉 기준(권고: rollout 접촉), 보상 사건이 없는 agent 처리(연구 범위 축소 또는 shaping probe로 v2 변경).

## 최신 확인 — 2026-10-29 KST, Claude Code (T143–T147, 정확한 시각은 대화 기록 참고)

- **실행 상태(10:40)**: A+C v1 run `runs/ac_20261023_185556`에서 IPPO·PGCT·Oracle s1은 complete다. All s1은 1.6M에서 진행 중이고, seed 2의 job 4개가 대기 중이다. Claude는 시작·중단하지 않았다. v2 실행 명령만 안내했다(T143, T144).
- **Oracle s1(E61, D90)**: 정답 팀원 critic 전이로도 IPPO보다 빠르지 않았다. 1M 배송 0.4 대 2.4, 4.5M 21.5 대 21.7. 증류는 probe 위 팀원 V를 150–250배 맞췄지만, 실제 방문 상태에서는 후반에 효과가 없다. 같은 팀이어도 팀원 critic은 내 return을 2–4배 못 맞힌다. 원인은 증류 세기가 아니라 대상(V^{π_j})으로 본다.
- **JS 가중 제안(T147)**: 미성숙일 때 가중이 최대이고 배울수록 줄어 방향이 반대다. 권하지 않는다.
- **PGCT `probe/cross_team_gate`**: 물리 이웃 비율일 뿐이다. 실제 v1 다른 팀 전이는 0이다. v2 run에서도 이 그래프는 같으므로 `update/*_arcs`·`update/ac_v2_audit_*`를 본다.
- **전이 재설계(D91, E62, T148)**: 권고는 팀원 critic의 max를 advantage에 쓰는 방식이다(GPI/MAMBA 계열). V_i는 자기 return으로 학습한다. r̂ 전이는 철회한다(발견은 병목이 아니다). 오프라인 점검 결과, 뒤처진 agent만 상향되고 다른 팀 critic은 가짜 상향을 준다. 팀 특이성은 작고 과대추정 위험이 있다.
- **반사실 점검(E63, D92, T149)**: 팀원 정책을 내 자리에서 실행해 보니, 뒤처진 agent에게 같은 팀 critic이 높게 나오면 대체로 실제로 더 나은 정책이 있었다(dV>.2에서 dMC>0이 67–100%). 다른 팀은 이득이 작고, 앞선 agent에게 가짜 상향을 준다. 행동 수준의 이득은 미입증이다. 통신 범위와 성숙 제한으로 기회는 21–40%다. 질의는 L8 window를 쓴다. 근거는 MAMBA로 고쳤다.
- **전이 방향 논의(T150–T153, D93–D95)**: B(팀 평균 보상)보다 D(max 결합 advantage)를 유지한다. MAMBA는 방법의 출처로만 인용하고 보장은 주장하지 않는다. 오프라인 행동 검증 대신 1M checkpoint 분기 학습을 관문으로 둔다.
- **Max-GAE 구현(E64, D96, T154)**: `latent_team/max_gae.py`, 모드 `oracle_max`·`oracle_maxd1`·`all_max`·`allk_max`, `--init-checkpoint` 분기를 구현했다. 테스트 49개가 통과했다. donor가 없으면 IPPO와 비트 단위로 같다. 규모 측정에서 raw Δ가 뒤처진 agent의 자기 advantage보다 2.6–12배 커서, D2는 상대 척도를 기본으로 했다. 학습은 시작하지 않았다.
- **실행 상태(11-02 13시 무렵)**: v1 run(PID 18880)은 seed 2 PGCT가 진행 중이다. v2 PGCT s1(`runs/acv2_s1_20261029_084340`)은 complete이고 아직 분석하지 않았다.
- **희소 보상과 r̂(E65, D97, T155)**: 우리 환경은 shaping·감점으로 agent-step의 25–55%에 보상이 있지만, 팀을 가르는 사건(배송·반납, r ≥ .5)은 1M 이전 전이의 .01–.03%다. v2 PGCT s1에서는 0.25M 무렵 같은 팀 거리 .25 대 다른 팀 .99로 갈렸다. 다른 팀 전이 352개는 모두 4.28M 이후다(원인 미분석). 보상 사건을 겪지 않은 agent는 보상 기반으로 식별할 수 없다는 한계를 논문 조건으로 명시하기를 권했다.
- **미결(사용자 결정)**: 1M 분기 실험(12 job) 실행, v1 run 처리, v2 PGCT s1의 후반 다른 팀 승인 원인 분석.

## 최신 확인 — 2026-10-26 13:22 KST, Claude Code

- **원칙(D82)**: PGCT는 각 agent의 개인 보상만 쓴다. 사용자가 절대 바꾸지 말라고 했다. 옛 팀 보상 run은 `env_candidates.py`의 후보 fallback 때문이었다.
- **방향(D83)**: 식별은 A+C 보상 프로브가 중심이다(개인 보상 예측기 + 경험 기반 사건 probe + 변형). B(정책+보상 결합)는 후속.
- **실행 중**: A+C v1 본 학습 `runs/ac_20261023_185556`(PID 18880, 사용자 10-23 18:55 시작, 8 job 순차, 모서리·개인, 좌표 마스킹). IPPO s1 complete(5M, 평가 배송 24.1/ep, 6명 중 5명 학습, agent 1은 0). PGCT s1은 13:22에 3.01M. 다음 순서는 Oracle s1, All s1, 이어서 seed 2의 IPPO·PGCT·Oracle·All이다. 사용자는 T132에서 진행 중 실험을 멈출 생각이라고 했지만, 이후 직접 이 run을 걸었다. Claude는 시작·중단하지 않았다.
- **v1 gate 진단(E56, E57)**: 2.87M까지 승인 5회, 전이 update 3개. 첫 전이 전까지 PGCT s1은 IPPO s1과 비트 단위로 같았다. 원인은 정확 일치 lookup·분산 필터·평균 희석·엄격한 τ. W&B `probe/*` gate 그래프는 물리 이웃 수일 뿐이다.
- **gate v2(E58, D84)**: `--reward-probe-ac --ac-gate v2`(기본 v1 유지). MLP 개인 보상 예측기 + 자기 보상 사건 probe·좌우 대칭 + 미성숙 보류 + 정규화 거리 < .5 + W&B `update/ac_v2_*`. 테스트 18개·smoke 통과. 본 학습은 사용자가 시작한다(권고: v2 PGCT만 seed 1·2, 기존 IPPO·Oracle·All을 비교군으로 사용).
- **오프라인 판정 비교(E59)**: A+C IPPO s1 5M 경험 재생에서 v2는 결정한 10쌍 모두 정답, 다른 팀 승인 0(같은 팀 거리 .14–.18, 다른 팀 .58–.70). v1은 정확도 .25–.63. 1M 모델은 배운 agent가 적어 v2가 모두 보류.
- **미결(사용자 결정 대기, D85)**: 배송을 못 배운 agent는 보상 probe로 식별되지 않는다. 배송 전 보상(pickup·shaping)으로 식별을 시도할지, 주장을 '식별된 팀원끼리 학습 가속'으로 줄일지. 그 밖에 v2 시점별 온라인 정확도, 인접 배치(음성 사례 후보), v1 PGCT s2를 계속 돌릴 가치.
- **설계 검토(D87, 같은 날)**: v2의 '2.56M 첫 승인'은 대부분 오프라인 재생 설정의 산물이다. 학습 로그상 agent마다 0–0.1M에 배송 기록이 12–23회 있다. 온라인 시점은 미확인. 초기 팀 신호 후보는 goal 진척 shaping의 방향 부호(관측이 북쪽 고정)다. pickup은 관측상 식별 불가다. 권고는 V2.1(대조 부호 비교)이며 V3(가상 궤적)는 근거가 약하다. 속도 상한은 Oracle s1로 먼저 확인한다.
- **초기 식별 측정(E60, D88, 같은 날 저녁)**: 실제 경험량으로 재생하면 V2 gate의 첫 같은 팀 승인은 모서리 0.26–0.32M, 인접 0.86M, 다른 팀 승인 0. 초기 팀 신호는 goal 진척 방향(모서리만)과 goal 근처 배송(두 배치)이고, pickup은 식별 불가. 대조 부호는 순위를 더 일찍 맞추지만 판정에는 보류 규칙이 필요하다. 남은 핵심은 증류의 학습 가속(Oracle s1).
- `docs/SESSION_HANDOFF.md`는 10-23 17:33 스냅샷이라 현재 상태와 다르다.

## 최신 변경 — 2026-10-23 KST, Codex (정확한 시각 미기록)

- 사용자 승인으로 A+C 실행 경로를 구현했다. `latent_team/train_from_scratch.py --reward-probe-ac`가 선택 옵션이다. 개인 보상만 사용하고 정책/critic/평가의 self_x/y를 마스킹한다.
- 보상 모델은 개인 경험 조건부 평균(비모수)이다. CNN 일반화 구현은 아니다. 공통 물리 전이와 관측에서 생성한 가상 local action branch를 현재 이웃끼리 비교한다. hidden team/ownership map으로 질의를 재라벨하지 않는다. 지원 부족/모호함/평탄 응답은 보류한다.
- `tests/test_reward_probe_ac.py` 11개 + 기존 `tests/test_latent_team.py` 20개, 총 31개 통과. PPO 최적화를 실행하지 않았으며 실제 팀 발견 AUROC/전이 향상은 미검증이다. 실행 중 프로세스는 확인/중단/재시작하지 않았다.
- 안내: `docs/REWARD_PROBE_AC.md`. 이전 문서의 실행 상태/대기 계획은 과거 스냅샷이다. 다음 본 학습은 별도이며 이번에는 시작하지 않았다.

## 최신 확인 — 2026-10-23 17:33 KST, Claude Code

- **새 세션 시작점**: `docs/SESSION_HANDOFF.md`. 연구 맥락은 `docs/RESEARCH_CONTEXT.md`, 실험 이력은 `docs/EXPERIMENT_HISTORY.md`에 있다.
- **실행 중**: 2×2 인접 칸 IPPO 1M × seed 2(`runs/queue_20261022_140114_b6_adjacent_{team,indiv}_pen005`). 17:33에 팀 0.84–0.85M, 개인 0.57–0.59M이다. 모서리 두 칸은 complete(E52).
- **대기 중**: 메인 1단계, 모서리·개인 IPPO/Oracle/All × seed 1–3, 5M(`runs/queue_20261023_172201_*`, PID 32236). 시작한 job은 없다. 인접 칸이 끝나면 자동으로 시작한다.
- **10-23 결과·결정**
  - 감점 .005로 pickup 정체가 풀렸지만, 1M에서는 한 agent가 독주한다(E52, E53).
  - 1M으로 최종 결론을 낸 것은 과했다. 메인은 5M이다(D78).
  - goal 방향 축이 개인 보상 칸에서도 팀 신호를 냈다(E54, D80).
  - 메인 설정을 개인 보상으로 바꿨다(D81).
- **미결(사용자 답 대기)**: 메인 대기열을 멈추고 팀당 요청 수 6·9개 pilot을 먼저 할지 정해야 한다(T130, E55).

## 최신 확인 — 2026-10-22, Claude Code

- **nopen pilot(E45→E46) 종료**: A·B nopen × seed 1·2, 1M, 모두 complete. 초기 가중치는 기존 pilot과 같다.
  - TOGGLE·요청 선반 pickup 붕괴는 사라졌다. pickup 정체의 원인(D69)이 인과적으로 확인됐다.
  - 그러나 네 run 모두 1M 안에 평가 배송 ≥1에 도달하지 못했다. 요청되지 않은 선반 pickup이 50–150/episode로 폭증했다.
  - 판정은 D70. 감점 0은 본 실험 환경으로 부적합하다는 의견이다.
- **다음 후보(승인 필요)**: 학습 없이 감점 크기 sweep으로 TOGGLE이 무작위 정책에서 가장 손해가 아니게 되는 값을 찾고, 그 값으로 B pilot을 다시 돌린다.
- E1-F는 여전히 1.536M에서 정지 상태다. 재시작은 승인 대기다.

## 최신 확인 — 2026-10-19 16:23 KST, Claude Code

- **E1-F main 5개가 종료됐다.** 13:07–13:08 KST에 Claude 데스크톱 앱 업데이트(이벤트 로그 13:08:12)로 앱 프로세스 트리와 함께 끝났다. 코드 오류는 없다.
  - 마지막 metrics: 1.67–1.70M step.
  - 마지막 checkpoint: 01500(1.536M).
  - W&B는 crashed로 표시된다.
- **재시작은 승인 대기**다. 처음부터 다시 할지, checkpoint에서 이어 할지는 새 wrapper가 필요하다. 어느 쪽이든 앱과 독립된 터미널이나 작업 스케줄러로 실행해야 한다(E26 추가).
- probe 분석은 E37–E39와 D57–D61을 참조한다. 보상 probe는 "유망 후보, 본 PGCT 도입 보류" 상태다.

## 최신 확인 — 2026-10-19 오후, Claude Code

- **E1-F main**: 계속 running이다. 12:31 KST에 약 1.66M이었다. 2M checkpoint는 오늘 저녁 무렵으로 예상된다.
- **probe 분석(E29–E34)**
  - 현재 probe JS는 팀 식별 지표로 믿기 어렵다.
  - 학습된 보상 예측기 probe는 철회했다(D50).
  - E1 판별 probe와 규칙 B는 독립 계보에서 일반화되지 않아 보류했다(D52).
  - 판정은 E1-F 학습 후 Δ_m(t)로 한다. 원 기획은 유지한다(D54 2×2 판정표).
- **Random-gated 구현(E35)**: `--modes random --random-reference`, smoke까지 했다. 본 학습은 참조 PGCT(10M)가 끝난 뒤, 승인을 받고 진행한다.
- **도구**
  - `latent_team/probe_wandb_viz.py`(W&B probe 컬러맵, 업로드 미시작)
  - `latent_team/render_checkpoint_videos.py`(영상)

## 최신 확인 — 2026-10-19 10:48 KST, Claude Code

- **E1-F main**: 5개 running, 약 1.52–1.55M step, 약 19 step/s. 평가 delivery는 거의 0이다(IPPO 1.48M에 0.05 한 번). PGCT gate는 아직 같은 팀을 고르지 않는다(E28).
- **지표 신뢰성(D45)**: 같은 seed에서도 전이 mode의 첫 발견 시점이 0.2–2.1M 흔들린다. 첫 발견 이후 0.5M 동안 PGCT가 IPPO보다 느리다. 단일 seed의 최종 성능 비교는 발견 운이 증폭된 값일 수 있다.
- **영상**: `output/videos_20261019/`(E1 6.40M 4개, E1-F 1.28/1.54M).
- **대기(승인 필요)**: 공통 발견 checkpoint에서 분기하는 실험(D45), 기존 대기 항목.

## 최신 확인 — 2026-10-18 14:30 KST, Claude Code

- **새 세션 시작점**: `HANDOFF.md`(전면 갱신)를 먼저 읽는다.
- **현재 실행**: E1-F main 5개(`runs/e1f_20261018_124343_main_canonical` 4 mode, `..._main_v2` PGCT-v2)가 running이다. 각 약 0.12M step, job당 약 19 step/s이고 10M까지 약 6일(10월 24일 무렵) 예상이다(E26).
- **연구 방향**: PGCT는 원 기획(잠재 팀 발견 + 팀 내 critic 전이)이다(D41). E1은 팀 대칭 대조 조건, E1-F가 본 시험이다.
- **논문**: 방법 논문 초안 v0.2 `output/ijcai_method_paper_20261018/`(D42). E1-F와 다중 seed는 TBD다.
- **대기(사용자 승인 필요)**: E1 seed 2·3(10M), E1-F 팀 보상 pilot, 개입 실험, 비결정성 조사, baseline 3종.

## 최신 확인 — 2026-10-18 12:44 KST, Claude Code

- **E1 run 중지(사용자 지시)**: `scratch_20261015_093349_stage2`를 약 6.5M에서 중지했다. 마지막 checkpoint는 06250이다(E24 추가).
- **현재 실행**: E1-F main `runs/e1f_20261018_124343_main_canonical`(IPPO/PGCT/Oracle/All, canonical)과 `..._main_v2`(PGCT, v2). individual reward, seed 1, 10M, 5 프로세스, W&B. 초기 hash는 E1 seed 1과 동일하다(E26).
- **다음**: 2M 무렵부터 `e1f_team_divergence.py`로 팀 분화와 probe JS를 확인한다. E1 seed 2·3, 팀 보상 pilot은 미실행이다.

## 최신 확인 — 2026-10-18 12:23 KST, Claude Code

- **현재 실행**: `scratch_20261015_093349_stage2` 4개 running, 약 6.4–6.55M step. 15M까지 약 4일 남아 10월 22일 무렵 종료 예상이다.
- **중간 성능(seed 1, 평가 최근 5회 delivery)**: Oracle 32.3 > PGCT 27.9 > IPPO 24.3 > All 20.4.
  - All은 배송 시작이 3.99M으로 늦었지만, 4명 모두 배송하며 가장 가파르게 오른다.
  - IPPO/PGCT/Oracle은 A4 배송이 0이다.
- **PGCT 전이**: 2M 이후 같은 팀 전이가 거의 사라졌고, 다른 팀 1–2·3–4 연결만 남았다.
- **재현성**: 같은 seed의 E17과 비교하면 IPPO는 완전히 재현됐다. 전이 mode는 1.4–1.6M에서 갈라졌고, All 시작 시점이 2M 차이 난다. 다중 seed가 필요하다(E24).
- **대기**: 15M 완료 후 seed 2·3, E1-F IPPO 시험 등은 사용자 결정 대기다.

## 최신 확인 — 2026-10-15 13:17 KST, Codex

- **읽기 전용 점검:** HANDOFF와 D24–D30/E20–E22, 현재 manifest/metrics/status/초기 hash/오류 로그, E1-F/v2 smoke 산출물을 대조했다. 기존 E1/canonical, home .5, individual reward, training seed 1, from-scratch 15M 비교가 진행 중이다. IPPO 321,536 / PGCT 316,416 / Oracle 314,368 / All 318,464 env steps이며 로그가 증가한다. 속도는 23.49–24.04 step/s다.
- **최근 평가:** 공통 307,200 step, deterministic masked, reset seeds 5,000,001–5,000,020, 20×500에서 요청 pickup/delivery/return 모두 0이다. 초기 구간으로 성공·실패나 mode 우열을 판정하지 않는다. 네 run의 마지막 checkpoint는 00250(256,000 step)이다. 오류 로그는 Gymnasium 경고/W&B 안내이며 Traceback 등 치명 오류 문자열은 발견하지 못했다.
- **Claude 추가 작업 확인:** E1-F 목표 분리와 gate 48/transfer 64의 v2 bank 옵션, seed 1 각 64-step smoke 완료 및 실제 distill_probe_count 64를 확인했다. 전체 183 tests 통과는 Claude 기록이며 이번에 재실행하지 않았다. 현재 본 학습은 여전히 E1/canonical이다. E1-F/v2 본 학습·팀 판별력/공유 이득 검증은 미실행이다.
- **대기/한계:** 4–5M 읽기 전용 점검, 독립 seed 추가와 E1-F/v2 및 MAPPO/SePS/HetDPS 본 학습은 기존 대기 상태다. learned-team online 모드도 미구현이다. E21 정리로 일부 과거 중간 checkpoint/smoke 출력은 없고 주요 로그/final은 보존됐다는 기록을 인계받았다. 이번 점검에서 삭제하지 않았다.
- **근거:** `output/status_review_20261015_131716/snapshot.json`, HANDOFF 및 EXPERIMENTS E19 추가. 새 학습/평가/tests/학습 코드 변경 없이 상태·대화 기록을 갱신했다.

## 최신 확인 — 2026-10-15 12:49 KST, Claude Code

- **새 세션 시작점**: `HANDOFF.md`가 현재 상태·대기 항목·원칙을 요약한다. 모든 기록을 합친 사본은 `research_archive/RESEARCH_ARCHIVE.md`다(D30).
- **현재 실행**: `scratch_20261015_093349_stage2` 4개 run running. IPPO 281,600 / PGCT 277,504 / Oracle 275,456 / All 279,552 step, job당 약 24 step/s. 4–5M 읽기 전용 점검 예정(약 10월 17일).
- **이후 추가 구현(학습 미실행)**: E1-F 환경 변형과 probe bank v2 옵션(D29, E22). 본 학습은 사용자 승인 대기다.

## 최신 확인 — 2026-10-15 11:49 KST, Codex

- **현재 실행(로그 확인):** `runs/scratch_20261015_093349_stage2`의 IPPO/PGCT/Oracle/All, E1 11×10, home .5, individual reward, training seed1, 무작위 초기화, 각 15M 목표. 마지막 env steps는 각각 200,704 / 198,656 / 196,608 / 199,680이다. 상태 파일은 `running`이며 11:48–11:49 로그 갱신을 확인했다. 네 초기 actor+critic hash가 같다. 이번 Codex 작업이 시작한 학습이 아니다.
- **평가:** 네 mode 모두 최근 153,600 step의 deterministic masked 20×500 평가(reset seeds5,000,001–5,000,020) delivery0.0이다. 초기 구간이므로 성능 판정을 하지 않는다.
- **이전 실행과 구분:** E17은 약2.5M 중단 run이고 이번 run은 seed1의 처음부터 재실행이다. 정확한 resume이나 독립 seed 추가가 아니다. 아래 과거 항목의 ‘재실행 결정 대기’와 ‘실행 중 없음’은 최신 상태로 읽지 않는다.
- **논문 분석(제안, 사용자 결정 아님):** 원래 목표인 국소 행동/작업 증거→잠재 팀 추론→critic 공유를 유지한다. 현재 learned-team online 모드는 미구현이고 성능 개선도 미확정이다. 기존 결과는 진단 근거로 쓰며, from-scratch 반복·Oracle/All 비교·online 방법 검증 후 주장을 확정한다.
- **산출물:** `output/ijcai_review_20261015/REPORT.md`(MD 교차 분석, 주장/증거, IJCAI 구성, 영문 탐색적 초록, 추가 실험 제안), `live_run_snapshot.json`. E19·D23 참고. 새 학습/평가/tests, 알고리즘·환경 변경은 하지 않았다.

아래는 이전 시점의 연구 스냅샷과 판단을 보존한 기록이다.

기록 작성: 2026-10-13, Codex. 최신 정량 분석 스냅샷: **2026-10-13 14:37 KST**. 진행 중인 실험은 이후 달라질 수 있다.
갱신: 2026-10-13 저녁(정확한 시각 미기록), Claude Code — 실행 상태 재확인, 연구 주제 재정립, 실험 A 결과 추가.

## 2026-10-13 갱신 — 연구 주제 재정립과 실험 A (Claude Code)

- **실행 상태(실측)**: 현재 실행 중인 학습 없음. E3 PGCT는 18:21 KST Windows 재부팅(시스템 이벤트 로그상 정상 재부팅)으로 2,855,936 step에서 중단. 마지막 checkpoint 2,764,800 step(`checkpoint_02700.pt`), 마지막 평가 2,816,000 step delivery 28.6. 매니페스트에는 여전히 `running`, E3 IPPO는 `planned`. 기존 실행기는 이어 학습을 지원하지 않는다. 재개 여부는 사용자 결정 대기.
- **연구 주제**: 사용자 원래 목표 A(국소 행동으로 잠재 팀 추론 → 팀 내부 critic consensus)를 본 연구로 확정. 팀 무관 JS 기반 transfer(B)는 별도 가설. D17.
- **실험 A(고정 모델, 학습 없음)**: 국소 pickup·delivery 목격 증거는 6개 모델 모두 같은/다른 팀 AUROC 1.00, 1 episode(500 step) 내 팀원 식별. 현재 canonical JS는 AUROC .375–.50, 팀원 식별률 0–.25(우연 .33). 정책 이력 JS는 E1 계열 .27–.29, E2 .81–.85.
- **핵심 문제**: 같은 팀 critic이 더 호환된다는 전제가 이 환경에서 성립하지 않았다. E1 계열 blend gain AUROC .25–.33(같은 팀이 더 나쁨), E2는 모든 쌍의 이득이 0 근처. 관측이 자기 팀 기준이고 individual reward에서 팀원이 같은 요청을 두고 경쟁하기 때문이라는 해석은 가설이다. E11.
- **실험 B 실행기**: oracle/all/wrong 분기 구현·smoke 완료, 장기 학습 미실행. E12.
- **본 학습 진행 중(E17)**: 사용자가 2026-10-13 20:23:52에 `scratch_20261013_202352_stage2`(IPPO/PGCT/Oracle/All, 15M, seed1)를 시작했다. 약 7일 예상이다. 4–5M 점검은 무중단, 읽기 전용으로 한다.
- **2026-10-15 중단**: E17이 Windows Update 재부팅(01:01)으로 약 2.46–2.50M step에서 끊겼다. 재개 방식은 사용자 결정 대기다(처음부터 재시작 권장).
- **2.5M 중단 시점 분석(E18)**
  - 네 mode 모두 1.6–1.9M부터 학습이 시작됐다.
  - 2.4M에서 Oracle이 delivery 약 10으로 가장 앞섰지만, 그 성능은 Agent2 한 명에게서 나왔다(seed 1, 귀속 불가).
  - PGCT gate는 초기에 같은 팀과 다른 팀에 비슷하게 열렸다.
  - 다음 단계로 재실행 시 seed 2·3을 병행하고, agent별 onset을 관찰할 것을 권장한다.
- **핵심 비교 계획(D22, E16)**
  - IPPO/PGCT/Oracle/All from-scratch 15M, seed1, 병렬로 진행하고 4–5M에 무중단 점검한다.
  - GPU 실제 설정에서 4 mode 초기 파라미터 일치를 확인했다.
  - 정확한 resume은 불가능하다(트레이너에 재개 경로 없음).
  - 본 학습은 사용자 실행/승인 대기다.
  - 목적은 critic 공유 효과 검증이며 잠재 팀 발견 해결이 아니다. 팀 목표 대칭성을 해석에 반영한다.
- **W&B·비교군 추가(D21, E15)**: from-scratch 파이프라인에 W&B(Step=env steps)와 MAPPO/SePS/HetDPS mode를 추가했고 smoke까지 완료했다. 기본 예산은 Stage1 4M, Stage2 15M이다. 본 학습은 사용자 승인 대기다.
- **처음부터 학습으로 전환(D20, E14)**: 사용자는 커리큘럼 기반 기존 데이터를 신뢰하기 어렵다고 판단했다. IPPO/PGCT/oracle/all/wrong 무작위 초기화 실행기를 구현하고 smoke까지 완료했다. 권장 첫 실행은 IPPO+oracle 2M pilot이며, 사용자가 직접 실행한다. Stage03 기반 oracle(D18)은 후순위다.
- **외부 검토 반영(D18)**: 국소 증거는 관측자 자신의 관측 채널과 100% 일치했다(E13, 2,146건). 상대 팀 ID는 미사용이고, 이웃 식별 가정이 남는다. 짧은 matched-prefix oracle 실행을 지원한다. 권장 다음 실행은 E1 oracle 1,024,000 step(약 7–8시간, 사용자 승인 대기)이다. team reward는 공통 사전학습부터 재정렬이 필요하고, 관측이 팀 기준인 한 팀별 critic 함수가 갈라지지 않을 수 있다(가설).

## 한눈에 보는 현재 판단 (14:37 스냅샷 기준, Codex)

**E1에서 PGCT의 평가 학습곡선 평균에 약 2.9% 이득 신호가 있지만, 최종 성능·critic 정확도·안정성 개선은 입증되지 않았다. E2의 on/off 성능은 거의 같고, E3는 대조군이 아직 없다.**

JS 수학 계산의 오류보다는 **어떤 probe/관측 분포에서 측정한 정책 유사성이 유익한 critic transfer를 예측하는가**가 현재 연구 질문이다. 실제 팀 발견을 목표로 유지할지, 팀 소속과 무관한 선택적 critic transfer로 확장할지는 확정하지 않았다.

## 프로젝트와 핵심 조건

- 루트: `C:\RWARE\MARL_RWARE\third_party\robotic-warehouse`.
- Python: `C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe`.
- 4 agents / 2 teams. 표시번호 1·3 = team 0, 2·4 = team 1. 코드의 agent index는 0-based다.
- `team_reward_mode=individual`. 같은 팀 소속이 동일한 개인 value target을 보장하지 않는다.
- Recurrent IPPO, semantic CNN observation, sensor range 5, action mask 사용.
- PGCT는 공통 canonical probe의 actor 응답으로 JS를 계산하고, 물리 이웃에서 갱신하는 EMA < 0.12 등을 만족할 때 critic 보조 전이를 허용한다.
- 실제 학습 probe는 48개, sequence length 1, zero hidden, unmasked다. 최종 raw JS와 과거 EMA 연결을 동일시하지 않는다.

## 모델·실험 경로

- 기존 curriculum: `runs/pgct09700_physical_mixing_20260929_155338_079437/`.
- 기존 최종 모델: 위 폴더의 `stage04_4ag_physical_home0.500/final.pt`.
- 새 비교의 공통 초기 모델: 위 폴더의 `stage03_4ag_physical_home0.750/final.pt`.
- 실제 공간 실험: `runs/sp_20261008_120638_983201/experiment.json`.
- 모든 공간 분기는 같은 Stage03 actor/critic에서 출발하며 optimizer/graph는 초기화된다. 따라서 **Stage03 이후 추가 전이의 조건부 효과**를 검증한다.

| 조건 | 학습 step | 마지막 delivery/episode | shelf return/episode | 상태(스냅샷) |
|---|---:|---:|---:|---|
| E1 11×10 PGCT seed1 | 4,000,768 | 44.35 | 42.65 | 완료 |
| E1 11×10 IPPO seed1 | 4,000,768 | 44.65 | 43.70 | 완료 |
| E2 16×16 PGCT seed1 | 4,000,768 | 37.85 | 36.15 | 완료 |
| E2 16×16 IPPO seed1 | 4,000,768 | 38.20 | 36.85 | 완료 |
| E3 20×22 PGCT seed1 | 2,252,800 | 27.25 | 25.95 | 14:37 기준 진행 중. **18:21 재부팅으로 2,855,936 step에서 중단**(최종 평가 2,816,000 step 28.6) |
| E3 20×22 IPPO seed1 | — | — | — | 시작 전 |

Home ratio .5, 요청 4개, LR1e-4, entropy .02. Task 평가는 공통 seeds5,000,001–5,000,020의 deterministic masked policy, 20×500이다. 구체적인 결과·분모는 `EXPERIMENTS.md`를 확인한다.

## 최신 핵심 관측

1. E1 평가 delivery의 정규화 AUC: PGCT45.856, IPPO44.573(+2.88%). 마지막10회 평균46.170/44.515. 최종 snapshot은 IPPO가0.30 높다. 단일 training seed이며 독립 반복 재현은 미확인이다.
2. E2 정규화 AUC34.167/34.245(−0.23%). Same-team gate가 다시 생겼지만 성능 이득과 연결되지 않았다.
3. E1 실제 전이는 3–4만2,005 pair-updates, same-team0. E2 same870/cross2,360. Gate 활성량을 성공으로 대체하지 않는다.
4. 네 agent 모두 업데이트하고 작업을 수행한다. E1은3·4가 delivery의 약61%를 담당하지만 E2/E3는 더 균등하다. 공간 편중은 IPPO에서도 관측된다.
5. 완료된 네 모델의 추가 stochastic critic 평가(각10×500)에서 정확도 개선은 미확인이다. 초기200-step 유한 MC MSE: E1 PGCT1.725/IPPO1.103, E2 .286/.281. 방문 분포 차이·유한 horizon 편향이 남는다.
6. 기존 Stage04 공통 recurrent rollout 진단에서는3–4 policy JS .060, critic 예측 간 MSE .107로 가장 낮았다. 이는 예측의 일치이며 실제 return 정확도나 전이 효과와 다르다.
7. Probe 중복 민감성은 지속된다. 최신 E1 PGCT에서3–4 JS48 .0921→고유26 .1421, 1–4 .2114→.1232로 순위 역전. E2 bank의 고유 입력은25개다.

## 추가 검토: E2 연결 집단 — 2026-10-13

E2의 완료 로그 3,907 update에서 최대 연결 성분 크기 3이 661회(세 쌍 모두 연결244회), 크기4가18회(완전 연결0회)였다. 환경의 팀 배정은 여전히 1·3/2·4다. 큰 성분을 모든 쌍의 유사성으로 읽어서는 안 되지만, 팀이 섞인 3개 완전 연결도 있어 단순 연쇄만으로 설명할 수 없다.

공통 이동 probe의 비중이 원인이라는 가설은 아직 검증하지 않았다. 원래 팀 발견 목표와 전이 유익성 목표를 구분하고, 사후 목표 변경으로 기존 실패를 정당화하지 않는다. 근거: `output/learning_review_20261013/e2_components.json`, `DECISIONS.md` D12, `EXPERIMENTS.md` E09.

## 다음 단계와 미해결 질문

- (2026-10-13 Claude Code) 사용자 결정 대기: ① E3 재개 여부(권장: 재개하지 않음), ② 실험 B를 현재 환경에서 확인용으로 실행할지(oracle+all, E1 seed1, 약 60시간), ③ 팀 소속이 value를 바꾸는 조건(예: 기존 옵션 `team_reward_mode="team"`)을 먼저 정할지. 아래 기존 항목 중 JS gate·맵 확장·seed 반복은 방향 B 검증이므로 A보다 후순위다.

- 새 환경 요청 수행(2026-10-13): 원래 코드와 별도 환경ID/실행기로control·detour(기본)·bottleneck·dock4단계 장애물/배송 대기 환경을 추가했다. 관련32tests와8조건64-step CPU smoke를 완료했다. 추가4M 장기 학습은 시작하지 않았다. `examples/HARD_RWARE_EXPERIMENTS.md`, `EXPERIMENTS.md` E10, `output/hard_rware_validation_20261013/validation.json` 참고. 기존 E1/E2/E3의 정량 스냅샷과 혼합하지 않는다.
- 방향별 전이 유익성(2026-10-13 검토): 대칭 JS와 `C(receiver←donor)`를 구분한다. 현재 allocation도 수신자별 정규화로 비대칭일 수 있으나 방향별 유익성은 측정하지 않는다. 동일 receiver 조건에서 peer 유무를 비교하는 검증 surrogate가 후속 아이디어이며 아직 실행하지 않았다.
- 사용자 정정(2026-10-13): SMAC 도입을 요청한 것이 아니다. 같은 형태의 agent라도 역할 분담의 성격과 차이가 크게 달라지는데 행동 유사도 기반 전이가 이를 감당할 수 있는지가 질문의 핵심이다. 환경 추가 요청으로 해석하지 않는다.
- 기존 진행 중인 E3와 예정된 IPPO 결과를 보존한다. 이 노트는 중단·재시작 명령이 아니다.
- E1 seeds2·3의 동일 조건 반복이 우선 제안이다. 장기 학습 완료 결과는 아직 확인되지 않았다.
- 이후 random donor와 drop34로 JS donor 선택과3–4 연결의 기여를 검증한다. 구현·64-step 검증은 있지만 장기 결과는 없다.
- 다른 사전학습 seed의 Stage03에서도 재현되는가? 현재 하나의 사전학습 모델 조건부 비교다.
- 낮은 policy JS가 전이 전 critic compatibility나 실제 성능 기여를 예측하는가? 이전 전이로 critic이 비슷해진 역인과 가능성이 있다.
- Probe 가중치는 실제 방문 빈도 또는 중요한 value 상태의 비중을 반영하는가? 아직 정당화되지 않았다.
- 공간 확대 시 shelf 밀도·경로 길이·접촉 빈도도 바뀌므로 순수 맵 크기 효과는 분리되지 않았다.

## 근거와 사용 안내

- [최신 전체 분석](output/learning_review_20261013/REPORT.md), [수치 스냅샷](output/learning_review_20261013/analysis.json).
- [추가 critic 평가](output/learning_review_20261013/critic_summary.json), [최종 probe 응답](output/learning_review_20261013/probe_responses.json).
- [공통 rollout 진단](output/pgct_transfer_review/common_rollout_stage04_e1_u600/summary.json).
- [기존 공간·gate 진단](output/pgct_spatial_analysis/REPORT.md), [transfer 실행기 문서](examples/PGCT_TRANSFER_EXPERIMENTS.md).

현재 공용 기록은 `AGENTS.md`·`CLAUDE.md`·이 파일·`DECISIONS.md`·`EXPERIMENTS.md`·`CONVERSATION_LOG.md`다. 원본 대화 자동 수집이나 Hook/MCP 동기화는 구성하지 않았다.

2026-10-13 추가 사용자 지시: 이후 모든 대화를 원문으로 기록한다. `CONVERSATION_LOG.md`의 L01–L12는 과거 요약이며, T001부터 사용자 메시지·진행 안내·최종 답변 원문을 남긴다. AGENTS.md와 CLAUDE.md에 지속 기록 규칙을 반영했다.

## 2026-10-23 KST Codex — A+C 코드 분석 (시각 미기록)
- 개인 보상/절대좌표 없는 A+C 방향. 공통 사건의 agent별 관측 대응 미구현, 배송 벡터 정확도 미검증. 상세 DECISIONS.md 같은 날짜 항목.
- 학습/알고리즘 변경 없음. 프로세스 상태는 이번에 재확인하지 않음.
