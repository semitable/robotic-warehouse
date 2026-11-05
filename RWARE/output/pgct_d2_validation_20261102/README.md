# PGCT-D2 구현 검증 — 2026-11-02 KST, Codex (E68)

대상: `latent_team/reward_probe_ac_v2.py`, `latent_team/max_gae.py`, `latent_team/train_from_scratch.py`, `tests/test_pgct_max.py`. trainer·환경 소스는 수정하지 않았다.

최종 검사: 새 15개 + 기존 Max-GAE/A+C/latent_team 57개 = **72 passed**. 결과 XML: `pytest_final.xml`. `pytest_final/`은 테스트가 만든 계획·로그·checkpoint다. `pytest_first/`는 테스트 fixture 두 오류가 있었던 첫 실행의 보존 자료이며 최종 성공 근거가 아니다. 그보다 먼저 임시 경로의 부모가 없어 17개 setup 오류가 발생했으며, 상위 폴더를 만든 뒤 재실행했다.

```powershell
& 'C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe' -m pytest tests/test_pgct_max.py tests/test_max_gae.py tests/test_reward_probe_ac_v2.py tests/test_reward_probe_ac.py tests/test_latent_team.py --basetemp output/pgct_d2_validation_20261102/pytest_final -p no:cacheprovider -q --tb=short --junitxml output/pgct_d2_validation_20261102/pytest_final.xml
```

이 명령은 실행 사실을 기록한 것이다. 재실행할 때 기존 basetemp/XML을 덮어쓰지 않도록 새 이름을 쓴다.

실제 trainer smoke는 `pytest_final/test_real_trainer_closed_gate_0/` 아래 9 job이다. 공통 조건은 `b6_corner_indiv_pen005`, 개인 보상, A+C 좌표 마스킹, seed 1, CPU, rollout 32 × 2 update = 64 step/job, 초기 모델 `runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt`(1,024,000 step)다. 최종 16-step 평가도 smoke용이며 학습 성능 근거로 쓰지 않는다.

| 하위 계획 | 검사 | 결과 |
|---|---|---|
| `plan/` (4 job) | IPPO, 승인 없는 PGCT-D2, 성숙 Oracle-D2, Oracle-D2 | 앞의 세 모드 최종 모든 tensor 비트 동등; v2 predictor version은 모두 2로 실제 재학습 확인 |
| `injected/` (2 job) | 두 새 모드에 정답 승인·성숙을 테스트에서만 주입 | 둘 다 Oracle-D2와 최종 모든 tensor 비트 동등; 실제 donor/actor 변화 있음; cross-team donor 0 |
| `filtered/` (2 job) | 같은 주입 + EV floor=1e9(테스트값) | 승인 존재에도 실제 donor 0; 둘 다 IPPO와 비트 동등 |
| `held/` (1 job) | PGCT-D2, warm-up=1 | update 1 donor 0, update 2 donor 존재 |

주입과 `min_events=1000000`, interval=1, train_steps=2, batch_size=16은 테스트 조건이다. 실제 gate 기준이나 연구용 설정을 바꾼 것이 아니다. 자연스러운 온라인 팀 승인이나 장기 학습 이득을 검증한 결과로 설명하지 않는다. 보상 사건이 없는 agent 문제와 E65 후반 오승인은 남아 있다.

실제 D2 donor는 `max_gae.jsonl.donors`에서 확인한다. `transfer.jsonl.actual_allocation`은 증류 allocation이므로 0이다. 기존 본 학습 폴더에는 쓰지 않았다. 커밋·본 학습 시작/중단 없음.
