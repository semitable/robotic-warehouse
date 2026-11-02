# PGCT 논문·발표자료 — 2026-11-02 Claude Code 재작성본 (T171)

Codex가 T167에 만든 초안을 바탕으로, 논문 본문·그림·발표자료를 새로 썼다. 최종 알고리즘(D99·D103 반영)과 실제 학습 로그만 근거로 한다. 진행 중인 학습 비교는 결과 칸을 비워 두었다(*pending* / *진행 중*).

## 결과물

| 파일 | 내용 |
|---|---|
| `paper/english/main.tex`, `main.pdf` | 영어 논문, **IJCAI-27 공식 템플릿**(`ijcai27.sty`, `named.bst`, 줄 번호, 익명). 7쪽 = 본문 약 5.6쪽 + 참고문헌 약 1.2쪽 (T173) |
| `paper/english/supplementary.tex`, `.pdf` | 보조 자료 2쪽(구현 세부, 환원 점검, 근거 출처). 본문 9쪽 제한 밖에서 따로 제출 |
| `paper/overleaf_ijcai27/`, `paper/overleaf_ijcai27.zip` | Overleaf(pdfLaTeX)에 그대로 올릴 수 있는 평평한 사본. `make_overleaf.py`로 다시 만든다 |
| `paper/korean/main.tex`, `main.pdf` | 같은 구성·수식·수치·인용의 한국어 논문, 8쪽(`kotex`, XeLaTeX) |
| `paper/method_equations.tex` | 두 논문이 함께 쓰는 수식 11개. rollout 시작 스냅샷 표기($\mathcal Q_{ij}^u$, $\widehat r^{\,u}$) 반영 |
| `paper/references.bib` | 공통 참고문헌 13개 |
| `paper/figures/fig_architecture.tex` | 그림 1: 전체 PGCT 구조(7단계, 발견/활용 두 영역, 접촉 시 교환 정보) |
| `paper/figures/fig_discovery.tex` | 그림 2: Reward Probe 기반 팀 판별(공통 probe, 같은 팀/다른 팀 응답, 거리·승인·보류). 막대는 설명용 |
| `paper/figures/fig_critic.tex` | 그림 3: 선택적 팀원 Critic과 Max-GAE(마스크·shrink, Critic은 자기 return만) |
| `paper/figures/pgct_style.tex` | 세 그림과 Beamer가 함께 쓰는 색·TikZ 스타일 |
| `paper/figures/discovery_log.pdf`, `baseline_ippo.pdf` | 실제 로그 그래프(논문용) |
| `presentation/pgct_beamer.tex`, `pgct_beamer.pdf` | 16:9 영어 Beamer 17장 |
| `presentation/pgct_script_ko.md` | 슬라이드별 한국어 발표 대본 |
| `presentation/figures/discovery_ema.*`, `discovery_alloc.*`, `baseline_ippo.*` | 발표용 실제 로그 그래프 |
| `presentation/pgct_presentation.pptx` | 같은 원본에서 함께 생성되는 편집 가능한 PowerPoint(참고용). 사용자가 Beamer만 필요하다고 해서 더 다듬지 않았다 |

## 원본과 재생성

- 실험 수치·그래프: `output/pgct_docs_20261102/evidence_plots.py`가 원본 로그(`runs/acv2_s1_20261029_084340`, `runs/ac_20261023_185556` IPPO seed 1·2)에서 다시 계산한다. 결과 `evidence.json`, CSV 3개.
- 슬라이드: `output/pgct_docs_20261102/deck_spec.py`(문구·수식·대본의 단일 원본) → `build_deck.py`가 Beamer `.tex`, PPTX, 대본 `.md`를 함께 만든다. Beamer `.tex`는 생성 파일이므로 문구는 `deck_spec.py`에서 고친다.
- PPTX 수식 이미지: `render_deck_equations.py`(matplotlib). Beamer는 같은 수식을 LaTeX로 쓴다.
- 점검: `check_consistency.py`(한영 수식·인용·라벨·그림·수치 일치, 슬라이드 순서), `preview_pptx.py`(PowerPoint가 없는 환경의 근사 미리보기와 글자 넘침 검사).
- 빌드(이 PC): `output/pgct_authoring_20261102/texruntime/tectonic.exe --keep-logs <tex>` (TECTONIC_CACHE_DIR=`output/pgct_authoring_20261102/texcache`). 일반 TeX 환경에서는 각 폴더에서 `xelatex` → `bibtex` → `xelatex` ×2. Beamer는 `presentation/`에서 `xelatex` 2회. Windows 글꼴 Times New Roman·Arial·Malgun Gothic을 쓴다.

## 검증 (2026-11-02)

- 세 문서 모두 컴파일 오류, Overfull box, 누락 글자 0.
- PDF를 그림으로 렌더링해 전 페이지·전 슬라이드를 눈으로 확인했다. 그림·표 잘림, 글자 겹침 없음.
- 한영 논문: 수식 매크로·인용·라벨·참조·그림 파일이 같고, 본문 수치 집합이 같다(한국어에만 있는 1.08은 줄 간격 설정).
- Beamer 17장과 대본 17개 절의 제목과 순서가 같다. 대본은 같은 원본에서 생성된다.
- PPTX는 PowerPoint로 직접 열어 보지 않았다(이 PC에 PowerPoint·LibreOffice 없음). 근사 미리보기에서 글자 넘침 0.

## 참고 자료

- `이기윤_ICCAS.pdf`(16장): 구성·스타일만 참고했다. 흰 배경, 파란 제목 "Section: Topic", 파란 삼각형 글머리표, 핵심어 색 강조(한계·핵심 메시지는 빨강), 왼쪽 글머리표·오른쪽 그림 배치, Related Work 표(Approach/Strength/Limitation)와 빨간 Research gap, Goal & Contributions에 목표·핵심 아이디어·기여 번호, 전체 구조도 뒤 구성요소별 슬라이드, 실험은 설정 → 정성 → 표 → 그래프, 결론은 Summary·Main contribution·Validation·Future work 순서. Multi-UAV 알고리즘·식·그림·수치는 쓰지 않았다.
- `음성 261023_135711.m4a`: 파일 머리에서 길이 812.98초만 확인했다. 이 PC에 ffmpeg·음성 전사 도구가 없어 듣거나 전사하지 못했다. 발표자의 실제 설명 순서·강조·수식 설명 깊이는 확인하지 못했고, 추측해서 반영하지 않았다.

## Codex 초안(T167)에서 대체된 파일

새 결과물이 쓰지 않는 Codex 산출물은 지우지 않고 남겼다: `paper/figures/{architecture,critic_access,discovery_process}.tex`, `discovery.*`, `ippo_delivery.*`, `paper/evidence_manifest.json`, `presentation/{slides.json,quality_report.json,reference_analysis_ko.md}`, `presentation/figures/{discovery_chart,learning_chart,equation_*}.*`, `output/pgct_authoring_20261102/`(제작 스크립트; Tectonic 실행 파일과 캐시는 계속 사용).

## 주장의 경계

- 핵심 기여는 잠재 팀 발견(Reward Probe 검정)이다. Critic 활용은 발견한 관계의 유용성을 검증하는 보조 메커니즘이다.
- 발견 근거는 seed 1, 5M의 발견 구성요소 학습 로그(update 종료 위치 접촉, 이전 팀원 활용 규칙)와 오프라인 재생(E59), 반사실 점검(E63)이다. 통합 PGCT의 학습 성능은 진행 중이며 수치를 넣지 않았다.
- MAMBA의 성능 보장은 주장하지 않는다.

## IJCAI-27 형식 (T173)

- 영어 논문은 공식 키트(`FormattingGuidelines-IJCAI-27`)의 `ijcai27.sty`·`named.bst`·전처리 구성을 그대로 따른다. `\pdfinfo{/TemplateVersion (IJCAI.2027.0)}`, `\linenumbers`, `\cite`(named), `algorithm` float, 초록 170단어, 부록 없음(보조 자료로 분리).
- 제출용 PDF는 Overleaf에서 pdfLaTeX로 만든다(`paper/overleaf_ijcai27.zip` 업로드, 주 파일 `main.tex`). 이 PC의 Tectonic(XeTeX) 빌드는 `\pdfinfo`를 대체 처리하고 Times 대신 같은 치수의 TeX Gyre Termes를 쓰므로, 쪽수 확인용이다.
- 한국어 논문은 검토용이라 기존 2단 형식을 유지하되 내용(초록, 알고리즘 1, 수식, 수치)은 영어판과 맞췄다. 한국어판 부록은 영어판 보조 자료에 해당한다.
- 남은 일: 재현성 체크리스트(제출 시 작성), Overleaf pdfLaTeX 빌드로 최종 쪽수 확인, 결과 표·그래프 채우기(약 1.4쪽 여유).
