# A+C 실제 gate 그림 (2026-11-02, Claude Code, T158)

W&B의 `probe/same_team_gate`·`probe/cross_team_gate`는 trainer의 `unrestricted` gate, 즉 통신 범위 안 이웃 비율이다. A+C의 판정이 아니다. 이 그림은 저장된 학습 로그만 읽어 실제 판정과 전이를 그린다. 학습·추론은 하지 않는다.

- 실행: `python output/ac_gate_figs_20261102/plot_gate.py <job 폴더> [출력.png]`
- v2 job은 네 패널을 그린다.
  - A. 누적 실제 전이 arc(같은 팀/다른 팀, `transfer.jsonl`)
  - B. v2 거리 EMA와 문턱 .5
  - C. update당 승인 쌍
  - D. 성숙 agent 수
  - B–D는 `update/ac_v2_*` 지표를 5만 step 구간으로 평균했다.
- v1 job은 A만 그린다.
- 현재 그림:
  - `b6_corner_indiv_pen005_acv2_pgct_s1_gate.png`(`runs/acv2_s1_20261029_084340`)
  - `b6_corner_indiv_pen005_ac_pgct_s1_gate.png`(`runs/ac_20261023_185556`)
- 진행 중인 run에도 쓸 수 있다. 다만 그 시점까지 기록된 로그만 그린다.
