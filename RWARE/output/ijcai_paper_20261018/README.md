# IJCAI 논문 초안 (Overleaf 프로젝트)

2026-10-18 Claude Code 작성. 프로젝트 기록(RESEARCH_STATE, DECISIONS D01–D37, EXPERIMENTS E00–E27, 대화 기록, `output/`의 분석 보고서, Codex의 논문 기획 `output/ijcai_review_20261015/REPORT.md`)을 종합했다.

## 파일

| 파일 | 내용 |
|---|---|
| `main.tex` | **영어 원고(제출용)**. IJCAI 형식, 익명 |
| `main_ko.tex` | **한국어 번역본**(검토·공유용). 같은 구조, 같은 그림·표 |
| `references.bib` | 참고문헌 11개 |
| `figures/` | 그림(PDF 5개, PNG 2개) |
| `tables/` | 자동 생성 LaTeX 표 4개(`output/paper_e1_20261018/make_paper_materials.py`) |


Overleaf 업로드용 zip은 2026-10-23 사용자 요청으로 지웠다. 위 파일들을 그대로 묶으면 같은 zip이 된다.

## Overleaf에서 여는 방법

1. 위 파일(`main.tex`, `main_ko.tex`, `references.bib`, `figures/`, `tables/`)을 zip으로 묶어 Overleaf → New Project → Upload Project로 올린다.
2. **IJCAI 공식 style 파일을 추가**한다. IJCAI 홈페이지의 Author Kit(제출 연도 것)에서 `ijcai27.sty`(또는 `ijcai26.sty`)와 `named.bst`를 프로젝트 최상위에 올린다.
   - 이 파일이 있으면 `main.tex`가 자동으로 IJCAI 형식을 쓴다. 없으면 비슷한 2단 형식(letter, 여백 0.75in)으로 컴파일된다. 쪽수 확인은 반드시 공식 style로 한다.
   - Overleaf의 IJCAI 공식 템플릿으로 프로젝트를 만들고 `main.tex`, `references.bib`, `figures/`, `tables/`만 옮겨도 된다.
3. 영어본: Menu → Main document = `main.tex`, Compiler = **pdfLaTeX**.
4. 한국어본: Main document = `main_ko.tex`, Compiler = **pdfLaTeX**(kotex). 한글 글꼴 오류가 나면 **XeLaTeX**로 바꾼다.

로컬에 LaTeX가 없어 직접 컴파일해 보지 못했다. 대신 환경 짝, 중괄호, `$` 짝, 인용 키, `\ref`–`\label` 대응을 스크립트로 검사했고 모두 통과했다. Overleaf에서 처음 컴파일할 때 오류가 나면 로그를 알려 주면 고친다.

## 논문의 방향

- **제목**: *Policy Similarity Is Not Team Membership: Diagnosing Selective Critic Sharing among Latent Teams*
- **형태**: 방법 성능을 주장하는 논문이 아니라 **진단 논문**이다. 현재 증거가 "PGCT가 학습을 개선한다"를 뒷받침하지 않기 때문이다. Codex 기획서(D23)의 RQ1–RQ3 구조를 따랐다.
- **본문 구성**: 서론 → 관련 연구 → 문제 설정(E1, E1-F) → 비교 방법(수식) → RQ1 식별 가능성(E11, E23) → RQ2·RQ3 처음부터 학습(E24, E27, D33, D37) → 논의 → 결론.
- **부록**: 하이퍼파라미터, 고정 모델 probe 비교, 공간 역할, probe-JS 표, 재현성, Stage03 연장 학습.

## 반드시 지킨 것

- seed 1개 결과는 모두 **pilot**으로 표시했다(본문 † 표시). "개선된다"는 표현은 쓰지 않았다.
- 미구현(온라인 팀 추론)이나 미완료(seed 2·3, E1-F 결과) 내용은 결과처럼 쓰지 않고 **다음 단계**와 빨간 `[TODO]`로 남겼다.
- Oracle-team은 상한이 아니라 특권 정보를 쓰는 참조 기준으로 썼다.
- E1의 팀 대칭성과 관측 가정(이웃 식별, 자기 목표 인지)을 명시했다.

## 제출 전에 할 일

1. **결과 보완**: seed 2·3(이상)과 E1-F가 끝나면 Table 1·2, Fig. 1–4를 seed 평균과 구간으로 바꾸고 † 표시를 정리한다. 이 결과에 따라 논문 방향(진단 논문 vs 방법 논문)을 다시 정한다.
2. **쪽수**: 공식 style로 컴파일해 본문 7쪽 + 참고문헌 2쪽(IJCAI 2026 기준, 2027 CFP에서 다시 확인) 안에 맞춘다. 지금 초안은 그림 4개·표 3개가 본문에 있어 넘칠 수 있다. 줄일 후보는 Table 3(이미 부록), Fig. 4, 관련 연구다.
3. **참고문헌**: `hu2025heterogeneity`(HetDPS)의 저자·연도·발표처를 확인해 채운다. 다른 항목도 최종 서지 정보를 한 번 대조한다. 완전한 선행 연구 조사는 아직 하지 않았다.
4. **익명성**: 저장소 경로, W&B 주소, 개인 정보가 본문에 없는지 확인했다. 코드 공개 링크는 익명 저장소로 따로 준비한다.
5. **IJCAI 2027 일정**: 초록 2027-01-04, 본문 2027-01-11(AoE, Codex가 2026-10-15에 공식 홈페이지에서 확인). 접수 전에 다시 확인한다.

## 수치의 출처

| 원고 내용 | 근거 |
|---|---|
| Table (식별 AUROC), blend gain | EXPERIMENTS E11, `output/latent_team/exp_a_20261013/` |
| 2,146개 사건 관측 검증 | E13 |
| canonical 26/48, 순위 변화 | E04, E20 |
| 고정 모델 20개 조합 | E23, `output/probe_team_check_20261015/` |
| 처음부터 학습 결과, 표 1–3, 그림 1–4, 재현성 | E24, E27, `output/paper_e1_20261018/numbers.json` |
| 공간 역할(Stage04) | `output/pgct_spatial_analysis/REPORT.md` 3.4, D33 |
| 커리큘럼 gate 거리 | `output/gating_diagnosis/README.md` |
| Stage03 연장 학습 | E02, E06 |
