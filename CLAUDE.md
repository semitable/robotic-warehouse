@AGENTS.md

# Claude Code 시작 안내

이 프로젝트의 공통 작업 지침은 `AGENTS.md`다. 위 import가 적용되지 않는 환경에서도 이 파일을 직접 읽는다.

새 세션은 `docs/SESSION_HANDOFF.md`(최신 인수인계, 2026-10-23)부터 읽고, 이어서 `HANDOFF.md`(10-18 기준 옛 인수인계)를 본다. 작업 전에 `RESEARCH_STATE.md`, `DECISIONS.md`, `EXPERIMENTS.md`를 읽고, 과거 대화가 필요하면 `CONVERSATION_LOG.md`를 확인한다. 작업 종료 시 공통 지침에 따라 해당 기록을 갱신한다.

공용 노트에 있는 실험 계획은 실행 완료나 새 실행 허가가 아니다. 기존 학습·로그·체크포인트를 보존하고 사용자의 현재 요청 범위를 따른다.

2026-10-13 사용자 요청: 이후 모든 사용자 메시지와 사용자에게 보인 답변·진행 안내를 `CONVERSATION_LOG.md`에 역할·순서별 원문으로 기록한다. 짧은 대화도 생략하지 않으며 종료 전 최종 답변까지 남긴다. 과거 요약과 이후 원문을 구분한다.

## 핵심 목표

- **PGCT(Probe-Gated Critic Transfer)**: 팀 라벨 없이 같은 팀(같은 목표를 가진 agent)을 probe 응답으로 찾고, 찾은 팀원끼리만 critic을 증류해 학습을 돕는다. 사용자는 이 원래 기획을 유지한다. "전이 적합성 발견"으로 중심을 옮기지 않는다(D41, D71).
- 질문은 두 가지다. Q1 식별(국소 정보로 팀원을 찾는가), Q2 전이(Oracle-team이 IPPO·All-peer보다 나은가).
- 기여 후보: 식별 가능성 조건 두 가지와 조건별 probe. (가) 공통 좌표에서 goal 기하가 다르다 → 행동 probe. (나) 보상 결합(팀 보상) → 보상 결합 probe(D71).
- 목표 학회: IJCAI 2027.

## 현재 연구 방향 (2026-10-23)

- 메인 설정은 **개인 보상**이다. 팀 보상에서는 보상 결합이 곧 팀의 정의라 자명하므로 참조로만 쓴다(D81).
- 환경: 두 팀 RWARE 11×10, 6 agent(3+3), 통신 3, pickup 감점 .005. 2×2 = goal 배치(인접/모서리) × 보상(개인/팀). 메인 칸은 모서리·개인(`b6_corner_indiv_pen005`)이다.
- 식별 주력 후보는 **goal 방향 축**이다(E54, D80). 정책 JS는 단독으로 실패했다.
- 메인 1단계: IPPO / Oracle-team / All-peer × seed 1–3, 5M. 판정은 5M에서 한다. 1M pilot으로 최종 결론을 내지 않는다(D78).

## 알고리즘 구조 (요약, 식은 `docs/RESEARCH_CONTEXT.md` 2절)

- 이웃(통신 범위 안)끼리 P=5 update마다 공통 probe에 대한 응답(h=0, unmasked)을 교환한다. 평균 JS → EMA(β=.35) → warm-up 100, n_min 8, `D̄ < τ=.12`인 이웃을 팀원으로 본다 → critic 증류(λ=.01, donor에 stop-gradient).
- mode: ippo / pgct(`graph_mode=policy`) / oracle / all(`unrestricted`) / wrong / random(Random-gated, `--random-reference`).
- probe bank: canonical 48개(고유 26개), v2(gate 48 + 증류 64).

## 코드 원칙

- 저장소 구조(2026-11-05 분리, D131): 공유 알고리즘 `PGCT/pgct_core`, RWARE 실험 `RWARE/`(기존 `rware/`·`latent_team/`·`examples/`·`tests/`·`runs/`·`output/`이 이 아래로 이동), 새 실험 `Overcooked/`. 안내는 `docs/ARCHITECTURE.md`·`docs/MIGRATION.md`·`docs/ALGORITHM.md`. RWARE 명령은 `RWARE/` 폴더에서 예전 그대로 실행한다.
- 수정 금지: `RWARE/examples/train_recurrent_ippo_consensus.py`(trainer), `RWARE/rware/`(환경), `RWARE/examples/`. 새 RWARE 기능은 `RWARE/latent_team/`에 새 파일이나 옵션으로 넣고 기본값은 유지한다. `pgct_core`를 고치면 `RWARE/tests/test_pgct_core_equivalence.py`(분리 전 동결 사본과 정확히 같은지)를 반드시 돌린다.
- 환경 후보: `RWARE/latent_team/env_candidates.py`. 실행: `RWARE/latent_team/train_from_scratch.py`, `RWARE/latent_team/run_queue.ps1`. 판정: `RWARE/latent_team/pilot_verdict.py`. Overcooked: `Overcooked/scripts/run_overcooked.ps1`.
- 수백만 step 학습은 사용자 승인 없이 시작하지 않는다. 긴 학습은 사용자가 Claude 앱 밖 PowerShell에서 건다. 앱 업데이트 때 하위 프로세스가 함께 죽은 적이 있다(E26).
- 자동화는 감시·판정·보고까지만 한다. 실행 중인 학습은 멈추거나 다시 시작하지 않는다.
- Python: `C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe`. pytest는 `--basetemp`를 scratchpad로 준다. 단, `C:\PGCT_Overcooked` 클론이 있는 PC에는 RWARE_MARL이 없고 `marl-policy-sharing` 환경(pyglet 없음)을 썼다.
- 백슬래시가 들어가는 내용(LaTeX, Windows 경로, Python 스크립트)은 Bash heredoc 대신 Write 도구로 파일에 쓴다.
- 커밋은 요청이 있을 때만 한다. 사용자에게 무조건 동의하지 말고 냉정하게 의견을 낸다.

## 문서 지도

- `docs/SESSION_HANDOFF.md`: 현재 실행 상태, 미결 결정, 다음 작업 순서, 작업 요령
- `docs/RESEARCH_CONTEXT.md`: 동기, 알고리즘·식, 검토한 신호, 방향 전환, 해결·실패·보류
- `docs/EXPERIMENT_HISTORY.md`: 실험별 환경·보상·seed·step·결과·경로
- 원본 기록: `RESEARCH_STATE.md`, `DECISIONS.md`(D), `EXPERIMENTS.md`(E), `CONVERSATION_LOG.md`(T, CRLF)
