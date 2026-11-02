# A+C r̂와 희소 보상 — 읽기 전용 측정 (2026-11-02, Claude Code, T155)

사용자 질문(T155): RWARE 보상은 일부 행동에만 있으니 대부분의 전이에서 r̂ = 0일 텐데, 보상 기반 식별이 가능한가.

- 대상: `runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1`(A+C v2 PGCT, 모서리·개인 보상, 좌표 마스킹, seed 1, 5M). 학습 로그와 `experiment.json`만 읽었다. 학습·추론은 하지 않았다.
- `reward_settings.py`: run의 환경 보상 설정(배송·반납·shaping·감점).
- `reward_sparsity.py`: 학습 로그 구간 평균으로 본 보상 밀도(0이 아닌 agent-step 비율), v2 사건(r ≥ .5) 누적 수, 성숙 agent 수, 사후 감사 거리(같은 팀/다른 팀), 승인 수.
- 결과는 `*_stdout.txt`이고, 요약은 `EXPERIMENTS.md` E65에 있다.
- 한계: seed 하나. 감사 거리는 그 update에서 비교한 쌍만의 평균이다. 다른 팀 승인이 4M 이후에 생긴 원인은 분석하지 않았다.
