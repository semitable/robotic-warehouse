# E1 from-scratch 실험 — IJCAI 논문용 자료

2026-10-18 Claude Code. 대상 run은 `runs/scratch_20261015_093349_stage2`(E1, seed 1)다. 15M 계획이었으나 사용자 요청으로 약 6.5M에서 중지했다. 원본은 읽기만 했고, 이 폴더에만 결과를 썼다.

다시 만들기: `python output/paper_e1_20261018/make_paper_materials.py`. CPU에서 수 분 걸리며, checkpoint로 probe JS도 다시 계산한다.

## 파일

| 파일 | 내용 | 논문 위치(제안) |
|---|---|---|
| `figures/fig1_learning_curves.pdf` | 4개 방법의 평가 배송 학습곡선 | 본문 Fig. 1 (1단 폭) |
| `figures/fig2_per_agent_deliveries.pdf` | 방법별 agent 4명의 배송(팀 색) | 본문 Fig. 2 (2단 폭) |
| `figures/fig3_critic_transfer.pdf` | 실제 critic 전이: 같은 팀/다른 팀, PGCT 쌍별 | 본문 Fig. 3 (2단 폭) |
| `figures/fig4_probe_js_gap.pdf` | probe JS(같은 팀) − JS(다른 팀) | 본문 Fig. 4 (1단 폭) |
| `figures/figA1_reproducibility.pdf` | 같은 seed 재실행 비교(E17) | 부록 |
| `tables/table1_performance.tex` | 배송 시작 시점, 평균·최종 배송, 배운 agent 수 | 본문 Table 1 |
| `tables/table2_transfer.tex` | 구간별 전이 비율, 같은 팀/다른 팀 arc | 본문 Table 2 |
| `tables/table3_probe_js.tex` | 1/2/4/6M probe JS 같은 팀/다른 팀 | 본문 또는 부록 |
| `tables/tableA1_hyperparameters.tex` | 환경·학습 설정 | 부록 |
| `RESULTS_DRAFT.md` | 영어 실험 설정·결과 문단, 그림 캡션, 한계 | 원고 초안 |
| `numbers.json` | 모든 수치 원본 | 확인용 |

그림은 PDF(벡터)와 PNG(300dpi)로 함께 있다. 글꼴은 Times New Roman이며, IJCAI 1단 폭(3.3in)과 2단 폭(6.9in)에 맞췄다. 표는 `booktabs` 패키지가 필요하다.

## 핵심 수치 (seed 1, 공통 평가 구간 0–6.4M)

| | 배송 시작(≥1) | 학습 중 평균 배송 | 최종 배송(최근 5회) | 배운 agent |
|---|---|---|---|---|
| IPPO | 2.05M | 10.4 | 23.2 ± 1.1 | 2/4 |
| PGCT | 1.79M | 11.7 | 27.3 ± 1.4 | 3/4 |
| Oracle-team | 1.64M | 13.2 | 32.2 ± 1.7 | 3/4 |
| All-peer | 3.99M | 3.7 | 19.4 ± 1.6 | 4/4 |

## 논문에 쓸 수 있는 것과 없는 것

| 주장 | 지금 쓸 수 있나 | 이유 |
|---|---|---|
| Oracle-team이 IPPO보다 빠르고 높다 | **아니오**(관찰로만) | seed 1개. 같은 seed 재실행에서도 전이 방식은 결과가 갈라졌다 |
| 공유 범위가 넓을수록 배우는 agent가 많다 | **아니오**(가설) | 패턴은 깔끔하지만 seed 1개. 누가 뒤처지는지는 우연적이다 |
| PGCT gate가 E1에서 팀을 찾지 못한다 | **예**(이 run의 사실로) | 실제 전이 로그상 2M 이후 같은 팀 전이 거의 0 |
| E1에서는 probe JS가 팀 신호가 아니다 | **예**(E1·이 seed 한정) | 4개 방법 모두 대부분 시점에서 같은 팀 JS > 다른 팀 JS. 이전 고정 모델 분석(E11, E23)과 일치 |
| 전이 방식은 같은 seed에서도 재현되지 않는다 | **예** | IPPO는 완전 재현, 나머지는 1.4–1.6M에서 갈라짐. 원인(GPU 비결정성 추정)은 미확인 |

## 바로잡은 수치

E17 All의 배송 시작은 **1.89M**이다(E18 기록과 같다). 2026-10-18 대화에서 "1.79M"이라고 쓴 것은 pickup 기준 값과 혼동한 것이다.
