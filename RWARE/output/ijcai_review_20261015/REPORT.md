# PGCT 연구 기록 분석과 IJCAI 논문 설계

2026-10-15 KST, Codex. 사용자 요청: Codex·Claude로 만든 프로젝트의 Markdown 내역 분석, 현재 내역을 포함한 IJCAI 논문 작성 방향 제시.

이 문서는 연구 분석과 집필 제안이다. 새 알고리즘의 구현, 환경·보상·probe·threshold 변경, 학습 시작·중단·재시작은 수행하지 않았다. 새 학습 실행은 기존 파일에서 발견한 사실이며 이번 작업이 시작한 것이 아니다.

## 1. 핵심 판단

**현재 자료는 잠재 팀 발견과 critic 공유의 관계를 연구할 근거를 제공하지만, PGCT가 잠재 팀을 찾아 학습 효율을 개선했다는 논문의 핵심 결론을 아직 뒷받침하지 않는다.**

사용자가 확정한 목표(D17·D22)는 다음이다.

> 국소적으로 관측한 타인의 행동에서 공통 목표의 팀원을 추론하고, 추론한 관계에 따라 critic 지식을 공유하여 학습 효율을 개선한다.

현재 구현의 정책 응답 유사도, 실제 팀 소속, critic 예측 일치, critic 정확도, 전이의 학습 기여는 서로 다른 변수다. 이들을 분리한 문제 정의와 검증 구조가 가장 설득력 있는 논문의 중심이다. 팀과 무관한 donor 선택 문제로 연구 목표를 자동 변경하지 않는다.

권장 전략은 **원래 목표를 유지한 방법론 논문을 준비하되, 기존 PGCT의 한계와 진단을 동기로 사용**하는 것이다. 현 자료만으로 먼저 쓸 수 있는 것은 탐색적 실증 분석 초고다. 진단 논문도 가능하지만, 현재 단일 seed·좁은 환경의 반례를 일반적 결론으로 확장하려면 추가 검증이 필요하다. 어느 방향도 채택 가능성을 보장하지 않는다.

## 2. 읽은 자료와 근거 범위

프로젝트 내부 Markdown 20개를 식별했다. 연구 노트와 실행 문서를 중심으로 본문 또는 관련 절을 확인했고, 대화 로그는 과거 요약·연구 방향 전환·실험 요청·수행 기록·최근 결과 구간을 확인했다. 길게 붙여넣은 외부 답변의 모든 원출처를 다시 검증한 것은 아니다.

| 자료 묶음 | 파일 | 분석에서의 역할 |
|---|---|---|
| 공용 상태·판단 | `RESEARCH_STATE.md`, `DECISIONS.md`, `EXPERIMENTS.md` | D01–D22, E00–E18의 목표·관측·계획 분리 |
| 대화와 지침 | `CONVERSATION_LOG.md`, `AGENTS.md`, `CLAUDE.md` | 사용자 목표, 수행 허가의 범위, Codex·Claude 기록의 수정 이력 |
| 실행기와 비교군 | `latent_team/README.md`, `latent_team/BASELINES.md` | from-scratch 비교, 미구현 learned-team, baseline 구현 차이 |
| 기존 방법과 실험 문서 | `examples/EX5_PGCT.md`, `PGCT_PHYSICAL_MIXING_CURRICULUM.md`, `PGCT_SPATIAL_EXPERIMENTS.md`, `PGCT_TRANSFER_EXPERIMENTS.md`, `HARD_RWARE_EXPERIMENTS.md`, `examples/README.md` | 실제 방법, 조건부 비교, random/drop ablation, 장애물 환경의 범위 |
| 분석 보고서 | `output/gating_diagnosis/README.md`, `output/pgct_spatial_analysis/REPORT.md`, `output/learning_review_20261013/REPORT.md`, `output/latent_team/exp_a_20261013/REPORT.md`, `output/scratch_review_20261015/REPORT.md` | 정량 진단과 한계 |
| 환경 개요 | `README.md` | RWARE와 커스텀 multi-team 환경의 구분 |

추가 확인: 중단·신규 run의 `experiment.json`, 신규 run의 `metrics.jsonl`·`job_status.json`·초기 가중치 hash, E18 `analysis.json`, 기존 curriculum manifest. 관측과 peer loss 관련 코드를 제한적으로 대조했다. 전체 학습 코드·baseline의 독립 재현 감사나 테스트 재실행은 하지 않았다.

대화 로그의 L01–L12는 과거 요약이고 T001 이후는 저장된 원문 기록이다. 별도 Codex·Claude 세션을 자동 수집하거나 누락된 원문을 복원하지 않았다. 과거 ex4/ex5 PDF와 첨부 텍스트는 이번에 다시 열지 않았고, 현재 MD에 적힌 내용만 참고했다.

## 3. 프로젝트가 발전한 과정

| 단계 | 기록상 한 일 | 연구적으로 남은 것 |
|---|---|---|
| 초기 PGCT | 공통 canonical probe의 actor 응답 JS, EMA gate, critic 보조 손실 구현 | 정책 유사도로 팀과 유익한 공유 상대를 구분할 수 있다는 전제 |
| 혼합 curriculum | complete-graph 사전학습 모델에서 물리 통신·home ratio 단계 적용 | 후반 same-team gate 소멸, 일부 cross-team 전이 |
| Codex 진단·대조 설계 | 접촉/승인 분리, probe 중복·공통 이력·critic 진단, Stage03 이후 on/off와 ablation 실행기 | 계산 오류보다 측정 분포와 전이 유익성 문제가 중요해짐 |
| 공간·난이도 확대 | E1/E2/E3 비교, 별도 장애물 환경과 smoke | 공간 확대가 팀 식별을 해결하지 않음; 난이도 자체가 원인인지는 미확인 |
| Claude의 목표 재정립 | 실험 A로 팀 식별 가능성과 critic 호환성 분리, oracle/all/wrong 준비 | 결과 수준 국소 증거는 유용하지만 같은 팀 critic의 우월한 호환성은 미확인 |
| 사용자 요구에 따른 from-scratch 전환 | 동일 무작위 초기화 비교, W&B, MAPPO·SePS·HetDPS 구현·smoke | 학습 초기부터 공유가 도움이 되는지 공정하게 묻기 시작 |
| E17·E18 | seed 1의 4 mode 학습이 약 2.5M에서 재부팅으로 중단, 초기 구간 분석 | Oracle 선행의 재현성과 원인은 미확인 |
| 이번 최신 확인 | 10월 15일 새 seed 1 run 발견 | 이전 MD의 ‘재실행 결정 대기’가 최신 실행 상태를 반영하지 않음 |

구현·로그·진단 기반은 많이 갖춰졌다. 반면 논문 주장에 해당하는 두 연결, **팀 추론→올바른 관계**와 **그 관계→학습 개선**, 중 어느 것도 현재 end-to-end 방식으로 검증되지 않았다. 테스트 개수와 실행기의 수를 연구 성과의 크기로 바꾸어 쓰면 안 된다.

## 4. 최신 실행 상태: MD보다 실제 로그가 앞서 있다

신규 run: `runs/scratch_20261015_093349_stage2`.

조건: E1 11×10, 4 agents/2 teams, home .5, individual reward, sensor/communication range 5, 무작위 초기화, training seed 1, mode별 요청 예산 15,000,000 env steps. manifest의 rollout 반올림 예산은 15,000,576이다. 네 mode 모두 `init_checkpoint=""`이다.

2026-10-15 11:49 KST의 읽기 전용 스냅샷(`live_run_snapshot.json`):

| mode | 마지막 기록 env steps | 로그 수정 시각 KST | 최근 평가 step | delivery/episode |
|---|---:|---|---:|---:|
| IPPO | 200,704 | 11:48:58 | 153,600 | 0.0 |
| PGCT | 198,656 | 11:49:28 | 153,600 | 0.0 |
| Oracle | 196,608 | 11:49:12 | 153,600 | 0.0 |
| All | 199,680 | 11:49:18 | 153,600 | 0.0 |

평가 조건은 deterministic masked policy, reset seeds 5,000,001–5,000,020, 20 episodes×500 steps다. 네 초기 actor+critic hash는 `f78ec390e0fad60c9bee9ea045525a79befbd71357f41235f04d61356b91c7e3`로 일치했다. 상태 파일은 `running`이고 로그가 갱신되는 것을 확인했다. 시스템 프로세스 상세 조회는 CIM 접근 제한으로 확인하지 못했으므로 PID 생존 여부를 별도로 입증했다고 쓰지 않는다.

이 신규 run을 E17의 정확한 resume이나 seed 2로 취급하지 않는다. 같은 seed 1의 처음부터 재실행이다. 향후 중단 run의 prefix와 새 run을 연결해 하나의 연속 학습곡선을 만들지 않는다. 이 초기 평가값으로 학습 성공·실패를 판정하지 않는다.

## 5. 현재 결과에서 쓸 수 있는 주장

### 5.1 Stage03 이후 추가 공유: 약한 조건부 신호

모델: `runs/sp_20261008_120638_983201/e{1,2}_{pgct,ippo}_s1/final.pt`. 공통 초기 모델은 physical curriculum Stage03 final. training seed 1, 추가 4,000,768 env steps, home .5, individual reward. 평가는 공통 seeds 5,000,001–5,000,020, deterministic masked policy, 20×500.

| 환경 | PGCT 평가 delivery 정규화 AUC | IPPO | 상대 차이 | 최종 delivery PGCT/IPPO |
|---|---:|---:|---:|---:|
| E1 11×10 | 45.8559 | 44.5734 | +2.88% | 44.35 / 44.65 |
| E2 16×16 | 34.1674 | 34.2449 | −0.23% | 37.85 / 38.20 |

AUC는 평가 51,200–4,000,768 step의 사다리꼴 적분을 구간 길이로 나눈 값이다. 실제 학습 중 누적 배송량이 아니다. 두 결과는 이미 PGCT로 사전학습한 동일 모델에서 분기했으므로, **처음부터 독립 IPPO보다 PGCT가 좋다**는 결론에 사용할 수 없다.

특히 curriculum manifest는 Stage01 이전에 `checkpoint_09700.pt`를 읽었다고 명시한다. 전체 사전학습 비용을 ‘Stage01–03 약 6M’만으로 보고하면 이전 사전학습 비용을 누락한다. 전체 효율 비교에는 원 checkpoint의 실제 누적 env step도 메타데이터로 확인해 포함해야 한다. 파일명만으로 총 학습량을 확정하지 않는다.

### 5.2 팀 식별 진단: 현재 JS 신호의 한계

E11 frozen 진단: Stage03→E1, E1/E2 IPPO·PGCT seed 1 final, Stage04 PGCT의 6개 checkpoint 조건. 자체 stochastic masked policy, reset seeds 5,200,001–5,200,020, 모델별 20×500. 서로 다른 6개 독립 training seed라는 뜻은 아니다.

- canonical 정책 JS 기반 같은/다른 팀 AUROC는 .375–.500, 팀원 recovery는 .000–.250였다. recovery의 균등 무작위 기준은 1/3이다.
- 국소 pickup·delivery 결과 증거의 AUROC는 1.000이었고, 누적 첫 episode 후 6쌍 모두의 팀 관계를 맞혔다.
- E13의 세 frozen 모델·각 5×500 진단에서 목격 증거 2,146건이 관측자 자신의 semantic 채널과 일치했다.

**쓸 수 있는 해석:** 이 평가 조건의 국소 작업 결과에는 팀 구분 정보가 있었으며, 현재 canonical 정책 응답 JS는 그 정보를 안정적으로 포착하지 못했다.

**쓸 수 없는 해석:** 온라인 잠재 팀 추론 알고리즘을 구현해 100%로 해결했다. 현재 진단은 env 이벤트로 사건을 재구성했고, 익명 관측의 사건을 특정 이웃에게 연결하는 가정이 남는다. 학습 초기에는 성공 사건이 드물어 이런 정보가 늦게 생긴다. frozen 숙련 정책의 첫 episode 결과를 무작위 정책의 첫 episode 결과로 일반화하지 않는다.

또한 .375–.500이라는 수치는 제한된 평가의 기술 통계다. 모든 행동 기반 팀 추론이 불가능하다는 통계적 증명은 아니다. 2 same/4 cross 쌍, episode·pair 종속성, 평가 모델 선정 편향을 명시해야 한다.

### 5.3 같은 팀과 critic 호환성은 동의어가 아니다

E11의 E1 계열에서는 같은 팀 critic의 예측 차이가 더 컸고, 정적인 blend gain은 낮았다. E2에서는 critic 차이와 blend gain이 작았다. E06의 stochastic on-policy MC 평가에서도 최종 critic 정확도 개선은 확인되지 않았다.

E06 모델은 위 공간 실험의 E1/E2 final, training seed 1, 추가 4,000,768 step이다. reset/action RNG seeds 5,100,001–5,100,010, stochastic masked policy, 10×500, 초기 200-step의 유한 return MC MSE는 E1 PGCT/IPPO 1.724620/1.103477, E2 .286130/.281301이었다. 모델별 방문 분포와 zero-tail 편향이 달라 이 차이만으로 전이의 손상을 단정하지 않는다.

팀 보상이 같아도 각 agent의 관측 이력과 역할이 다르면 조건부 value가 다를 수 있다. 반대로 다른 팀이어도 자기 팀 기준 관측으로 과제가 정규화되면 같은 입력에서 value 함수가 비슷할 수 있다. 따라서 팀 보상 변경만으로 same-team consensus가 항상 유리해진다고 가정하면 안 된다.

### 5.4 중단된 from-scratch 비교: Oracle의 초기 선행

E17: `runs/scratch_20261013_202352_stage2`, 무작위 초기화, E1 11×10, home .5, individual reward, training seed 1. 15M 목표였으나 약 2.46–2.50M에서 중단. 평가는 deterministic masked policy, 공통 reset seeds 5,000,001–5,000,020, 20×500.

| mode | 자기 run 마지막 3회 평균 delivery | 공통 평가 구간 정규화 delivery AUC |
|---|---:|---:|
| IPPO | 3.10 | .41 |
| PGCT | 1.92 | .26 |
| Oracle | 9.98 | 1.28 |
| All | 3.45 | .69 |

AUC의 공통 구간은 51,200–2,406,400 step이다. 마지막 3회는 mode별 마지막 평가 시점이 완전히 같지 않은 요약이므로 최종 본 논문에서는 공통 step으로 다시 맞추는 편이 좋다.

Oracle 마지막 3회 delivery는 Agent2 한 명이 모두 담당했고 같은 팀 Agent4는 0이었다. 이는 Oracle의 총 성능 개선이 **그 한 명의 실적에서 나왔다**는 뜻이다. Oracle이 Agent2의 학습을 촉진했을 가능성까지 배제하지는 못한다. 그 원인을 consensus로 귀속하거나 우연으로 확정할 수 없다는 것이 정확한 결론이다.

PGCT는 same 3,496 / cross 5,590 directed arc-updates를 기록했다. same 가능한 쌍은 2개, cross는 4개이므로 원시 합계 비율만으로 판별 능력을 비교하지 않는다. 물리 후보·접촉 노출을 분모로 한 정밀도·재현율·승인율이 필요하다.

## 6. MD 내역에서 바로잡거나 구분할 부분

1. **실행 상태의 시차:** `RESEARCH_STATE.md`의 ‘실행 중 없음’, ‘진행 중’, ‘재실행 결정 대기’가 서로 다른 시점에 추가돼 한 페이지에 공존한다. 이번 최신 확인을 맨 위에 추가하고 과거 스냅샷은 보존한다.
2. **오래된 계획과 실제 결과:** E3의 진행 표, Stage03 oracle 계획, from-scratch 장기 미실행 문구는 작성 당시 상태다. 최신 E17/E18과 신규 run을 구분해 읽어야 한다.
3. **예산 기본값:** `latent_team/README.md`에 과거 2M/10M 예시와 최신 4M/15M 안내가 함께 있다. 논문에는 README의 임의 예시가 아니라 각 run manifest의 실제 설정을 쓴다.
4. **Oracle의 의미:** Oracle은 정답 팀을 쓰는 비교군이다. 최적 donor·weight·critic 호환성을 아는 수학적 성능 상한은 아니다. same-team만 고집하는 것이 all-peer보다 반드시 유리하지는 않다.
5. **공유 방식의 이름:** 현재 peer loss는 공통 probe에서 donor 예측을 stop-gradient target으로 삼는 value distillation 형태다. 네트워크 파라미터 평균이나 consensus 수렴 보장이 있는 방식과 구분해야 한다.
6. **분산성:** actor가 자기 관측만 쓰는 것과 학습 프로토콜의 완전 분산성은 다르다. 공통 bank 생성·배포, 이웃 식별, actor 응답/critic 값 교환, 전역 라벨의 logging-only 사용 범위를 명시해야 한다. ‘통신 없는 학습’이라고 쓰면 부정확하다.
7. **관측 정보:** semantic 관측은 상대 팀 ID를 직접 주지 않지만, 자기 요청·관련 목표는 알려준다. ‘목표 정보가 전혀 없는 순수 행동만으로 추론’이라는 표현은 범위가 과하다.
8. **성공률 정의:** 기존 full-cycle 지표는 episode에서 delivery와 return이 있었는지를 보는 값이다. 모든 주문·선반의 개별 cycle이 100% 완수됐다는 뜻으로 사용하지 않는다.
9. **불확실성:** 인접 평가 차이의 표준편차에는 학습 추세도 섞인다. 순수 평가 잡음, training seed 분산, 신뢰구간으로 부르지 않는다.
10. **구현 비교의 공정성:** `BASELINES.md`에 공통 PPO 외에도 advantage 정규화 범위와 zero-reward actor skip 차이가 기록돼 있다. ‘알고리즘 구조만 다르고 학습 설정이 완전히 동일’하다고 쓰기 어렵다. 공식 재현과 통제된 이식 구현을 구분하고, 결과 확보 전에 어느 비교를 할지 고정해야 한다.

기존 자료를 버릴 이유는 없다. 기존 curriculum·공간 실험은 가설 생성과 한계 진단으로 가치가 있다. 주성능 표에는 from-scratch 결과를, 조건부 적응 실험에는 기존 결과를 각각 배치한다.

## 7. 논문의 중심 질문과 기여 설계

### RQ1. 목표 관계는 국소 행동·작업 결과에서 식별되는가?

정책 응답 JS, 실제 이력 JS, 국소 작업 결과 증거를 비교한다. 학습 초·중·후반, 접촉량, 사건 희소성, 역할 차이, 환경 대칭성에 따라 식별 성능이 어떻게 달라지는지 측정한다.

### RQ2. 정확한 목표 관계를 아는 것이 critic 공유에 도움이 되는가?

IPPO/Oracle/All/Wrong 비교로 본다. 팀 라벨 정확도가 높아도 공유 효과가 작을 수 있음을 허용한다. 같은 reward 목표와 같은 value 함수라는 전제를 분리한다.

### RQ3. 추론한 관계를 사용해 실제 학습 효율이 개선되는가?

새 learned-team 모드가 필요하다. 현재 PGCT를 이 모드로 이름만 바꾸면 안 된다. 팀 추론을 거친 공유가 IPPO와 All보다 유리하고, Oracle과의 차이가 줄어드는지를 여러 training seed에서 확인한다.

권장 기여는 세 가지다. 아래는 **달성 목표**이며 현재 실적 목록이 아니다.

| 기여 목표 | 현재 확보 | 부족한 증거 |
|---|---|---|
| 정책 유사도·목표 관계·value 공유 유익성을 분리하는 문제 정식화와 진단 | 상당 부분 있음 | 통제된 반례, 기존 연구 대비 차별성 검토 |
| 국소 작업 증거에 기초한 온라인 팀 추론과 선택적 critic distillation | frozen 정보존재 진단 | 관측 기반 사건 검출·이웃 귀속·online posterior·learned allocation 구현 |
| 팀 선택이 도움이 되는 조건과 실패 조건의 실증 분석 | 조건부 on/off, 중단 pilot | 완료된 from-scratch 반복, 정보조건·전달량 ablation, 역할/목표 분리 |

단순 Beta 누적이나 JS 대신 다른 거리 사용만으로 독창성을 주장하기는 어렵다. **사건이 드문 초기 학습에서 어떻게 오판을 줄이고, 역할이 달라도 같은 목표 관계를 유지하며, 그 관계가 언제 실제 공유 가치를 갖는가**를 해결해야 한다. 이 관점이 기존 선택적 parameter sharing과의 실질적 차이가 될 수 있다. 다만 전면 문헌조사를 마친 독창성 판정은 아니다.

## 8. 방법 절을 쓰는 방식

### 8.1 문제 정의

부분관측 다중에이전트 확률게임, agent i의 관측 이력 h_i, 행동 a_i, 개인 보상 r_i, 숨은 목표 집단 z_i, 물리 이웃 N_i(t)를 정의한다. 각 agent는 자기 목표에 관한 task-relative 단서를 알지만 상대의 z_j는 모른다는 현재 정보 조건을 정확히 쓴다. 공통 목표와 보상 공유 여부는 별도 정의한다.

### 8.2 기존 PGCT: 재현 가능한 기준 방법

공통 probe 분포 Q의 정책 응답 거리:

`D_ij = E[x~Q] JS(pi_i(.|x,h=0), pi_j(.|x,h=0))`.

실제 설정은 48 probe, sequence length 1, unmasked natural-log JS다. 이웃에서 갱신하는 EMA, threshold .12, warm-up 100 update, min-count 8, 측정 간격 5 update, 현재 물리 이웃 마스크를 구분한다. 최종 raw JS 하나를 과거 gate와 동일시하지 않는다.

receiver별 allocation은 gate 행을 정규화한 alpha_ij다. 단순화한 critic 목적은 다음과 같다.

`L_i = L_i^PPO + lambda_peer * sum_j alpha_ij E[x~Q](V_i(x) - stopgrad(V_j^old(x)))^2`.

현 설정의 lambda_peer는 .01이다. `L_i^PPO` 안의 local value loss coefficient는 .5다. donor target은 update 시작 시 고정한다. 수렴 증명 없이 ‘같은 팀의 최적 value로 수렴’이라고 주장하지 않는다.

### 8.3 후속 learned-team 방법: 제안이며 미구현

국소 관측 이력에서 상대 j의 작업 사건 e_ij를 검출하고, `q_ij(t)=P(z_j=z_i | local evidence through t)` 같은 신뢰도를 추정한다. 증거가 부족하면 공유를 보류하는 기준을 명시한다. 물리 후보 안에서 q_ij로 gate/weight를 정하고 기존 critic distillation을 적용하는 방식이 가장 작은 확장이다.

필요한 구현·평가 항목:

- env의 정답 이벤트를 detector 입력으로 쓰는 조건은 privileged 진단으로 분리한다. 실제 방법은 허용된 관측 이력에서 사건을 검출한다.
- 이웃 ID와 익명 관측 물체의 대응을 가정하는지, 추가 통신이나 tracking으로 해결하는지 적는다.
- 양의 사건만 기다리는 cold-start 지연, 잘못된 pickup·delivery 귀속, 관측 누락, 증거 없음을 오판으로 처리하는 문제를 다룬다.
- q_ij는 관측자별로 다를 수 있다. 대칭성·전이성을 강제할지, pair 관계만 쓸지 정하고 비용을 기록한다.
- 모든 agent의 중앙 clustering을 쓴다면 분산 주장과 정보 조건을 다시 정리한다.
- 추론 데이터와 검증 데이터를 분리하고 threshold·hyperparameter를 최종 test seed로 고르지 않는다.

팀 추론과 방향별 transfer utility까지 동시에 새로 학습하는 복잡한 방법은 별도 확장이다. D15의 방향별 유익성 surrogate는 유용한 진단 후보지만, 지금 논문의 목표를 다시 팀 무관 transfer로 옮기는 이유가 되지 않는다.

### 8.4 넣을 수 있는 간단한 분석 예시

정책 유사성만으로 value 호환성을 보장할 수 없음을 설명하는 자체 구성 예시다. 한 상태·한 행동의 무한 할인 MDP에서 두 agent의 정책은 같아서 JS=0이다. 매 step 보상이 각각 0과 c라면 value는 0과 `c/(1-gamma)`로 다르다. 따라서 정책 JS만으로 보상·value 차이를 제어하는 보편적 보장은 추가 가정 없이 성립하지 않는다.

이 예시는 현재 RWARE를 분석적으로 설명했다는 증명이나 새로운 정리의 독창성 주장이 아니다. 본 논문에서 일반 보장을 쓰려면 observation alignment, reward/transition 차이, 정책 평가 목표, probe coverage 가정을 명시하고 별도 증명해야 한다. 현재 recurrent nonlinear PPO+distillation에 기존 선형 consensus 수렴 정리를 그대로 적용할 수 없다.

## 9. 필요한 실험과 우선순위

이 표는 제안이며 새 실행 허가나 실제 수행 기록이 아니다. 현재 실행 중인 seed 1 run은 그대로 보존한다.

| 우선순위 | 비교 | 답하려는 질문 | 완료 기준 |
|---|---|---|---|
| P0 | 현재 IPPO/PGCT/Oracle/All from-scratch | 이 설정에서 공유 효과가 관찰되는가 | 동일 예산 완료, 공통 평가 step, agent·팀별 성과 |
| P1 | 같은 4 mode의 독립 seed 2·3, 이후 필요시 추가 seed | 효과가 초기 성공의 우연에 민감한가 | seed별 AUC 차이·최종 구간 차이·구간 추정 |
| P2 | learned-team + IPPO/PGCT/Oracle/All | 추론된 관계가 실제 학습을 돕는가 | 관측 기반 online 구현, 추론·학습 동시 평가 |
| P3 | Wrong / 무작위 관계 / 전달량 통제 | 같은 팀을 고르는 것이 특별히 중요한가 | 실제 allocation·부족량·loss 노출량 기록 |
| P4 | 역할 차이×목표 차이의 통제 조건 | 다른 역할의 팀원과 비슷한 행동의 타팀원을 구분하는가 | 같은 팀/다른 역할, 다른 팀/비슷한 역할 비교 |
| P5 | MAPPO·SePS·HetDPS, 다른 규모·layout | 경쟁력과 적용 범위 | 정보·구현 차이 공개, 추가 compute 포함 |

3 seed는 우선 점검용 제안이다. 최종 핵심 비교는 자원이 허용하면 5개 이상 독립 training seed와 불확실성 보고를 권장한다. 이는 IJCAI의 고정 seed 수 규정이 아니며, seed 수가 적다면 그 한계를 그대로 적는다. seed 수를 늘리는 것만으로 방법의 독창성이 생기지는 않는다.

최소 ablation은 (a) critic 공유 없음, (b) 모든 이웃 공유, (c) 정답 팀 공유, (d) 잘못된/무작위 관계, (e) 국소 증거 추론, (f) 기존 정책 JS다. 동일한 전달량 비교는 전체 방법 비교와 별도로 수행한다. 거리/위치에 따라 후보 수가 달라 완벽히 맞출 수 없다면 부족량을 보고한다. Stage03용 random 실행기를 from-scratch에 이미 적용했다고 가정하지 않는다.

환경 확장은 맵을 크게 만들거나 장애물을 늘리는 것만으로 대체되지 않는다. 팀 관계가 보상·value·공유 가치에 어떻게 관련되는지 통제해야 한다. 현재 individual reward와 대칭 task-relative 관측은 실패·경계 조건으로 남길 수 있다. 다른 보상/과제 조건을 추가한다면 처음부터 새 비교를 하고 기존 결과와 섞지 않는다. SMAC 도입은 현재 사용자 요청이 아니며 필수 조건으로 제시하지 않는다.

## 10. 지표와 통계 설계

- 주지표: 공통 env-step 구간의 delivery AUC/step. 최종 구간 delivery·return, 팀별 throughput, 실제 task return을 함께 보고한다. ‘수렴 속도’를 주장하려면 사전 지정한 목표 성능 도달 step도 쓴다.
- 추론: 같은 팀 edge precision/recall/F1, AUROC와 PR-AUC, 관측 가능한 후보 coverage, 첫 정확한 관계까지 env steps, 오판 지속 시간, 신뢰도 calibration. 미접촉·증거 없음은 따로 보고한다.
- 공유: 실제 directed allocation mass, donor 수, same/cross 후보 대비 승인, peer loss, 필요시 gradient 규모. pair-update와 arc-update를 혼합하지 않는다.
- 학습 분포: agent별 첫 pickup/delivery/return, 성공 agent 수, 팀별 최소 성과, 성과 집중도. 고정 agent 번호가 아니라 각 seed의 최초 성공 agent와 팀원을 기준으로 지연을 볼 수 있다.
- 원인 검증: 같은 팀원이 먼저 성공한 뒤 다른 팀원이 빨라지는 것은 관찰적 지표다. 공유 차단·donor 교체 같은 개입 대조가 없으면 인과 증거라고 부르지 않는다.
- critic: on-policy 유한 MC와 공통 이력 prediction gap을 분리한다. 다른 정책의 rollout return을 해당 정책 critic의 on-policy 정답으로 사용하지 않는다. 시간 제한 tail과 reward scale도 함께 보고한다.
- 비용: env steps, wall-clock, 평가·probe 비용, 전송한 값/byte 수, 모델 수와 파라미터 수. 분산성을 기여로 삼으면 중앙 기준 방법과 정보·통신 예산을 나란히 적는다.
- 불확실성의 기본 단위는 독립 training seed다. 여러 평가 episode와 수십 checkpoint를 독립 학습 반복으로 부풀리지 않는다. seed별 paired 차이를 우선 보고하며 episode 표본은 그 안에 종속된다.
- 동일 seed 1의 중단·재실행은 독립 seed 2개가 아니다. 여러 task를 종합할 때는 정규화와 task 간 가중을 사전에 정하고, 적절하면 IQM·구간 추정도 사용한다.
- 주지표·최종 구간·예산·실패/중단 처리와 tuning/test seed 분리를 결과 확인 전에 정한다. 좋은 구간만 사후 선택하지 않는다.

관련 평가 방법론은 [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)을 참고한다.

## 11. 결과에 따른 논문 방향 결정

| 반복 실험에서의 결과 | 지지하는 해석 | 다음 집필 판단 |
|---|---|---|
| Oracle이 IPPO·All보다 일관되게 유리 | 현재 조건에서 팀 선택의 가치가 있음 | learned-team을 완성하고 Oracle gap 분석 |
| Oracle·All 모두 IPPO보다 유리하고 서로 비슷 | 공유는 유용할 수 있으나 팀 선택의 추가 가치는 불명확 | 원래 목표의 증거가 부족함을 인정하고 역할/목표 조건 분해 |
| Oracle도 IPPO보다 뚜렷하지 않음 | 현재 방식·조건에서 팀 제한 공유의 이득 미확인 | 공유 메커니즘·value 정의·환경 조건 점검, 팀 추론부터 복잡화하지 않음 |
| learned-team은 팀을 맞히지만 성능은 개선하지 않음 | 식별 성공과 학습 성공이 분리됨 | 핵심 실패 조건으로 보고, end-to-end 성공 주장 제외 |
| PGCT는 팀을 못 맞혀도 성능 향상 | 팀 무관 공유의 별도 증거 후보 | 원래 목표 달성으로 설명하지 않음; 목표 변경은 사용자와 명시적으로 결정 |

‘차이 없음’은 적은 표본으로 동등성이 입증됐다는 뜻이 아니다. 효과 크기·구간 추정·예산을 보고 판단한다. 원하는 결과가 나올 때까지 환경을 바꾸는 방식 대신 검증하려는 구조적 조건을 먼저 정한다.

## 12. IJCAI 원고 구성안

현재 날짜 2026-10-15 기준, 다음 회차인 [IJCAI 2027 공식 홈페이지](https://2027.ijcai.org/)에는 초록 2027-01-04, 본문 2027-01-11, 모두 23:59 AoE로 안내되어 있다. 접수 직전에 변경 여부를 다시 확인해야 한다. 이번 확인에서는 2027 main-track 상세 CFP를 확보하지 못했다.

아래는 [IJCAI-ECAI 2026 main-track CFP](https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-main-track/)의 본문 7쪽·참고문헌 2쪽을 참고한 집필 배분안이며, 2027 확정 규정으로 간주하지 않는다. 2026 CFP의 익명 제출·재현성 안내와 LLM 사용 책임 규정도 대상 연도 CFP에서 다시 확인한다.

| 본문 절 | 배분 예시 | 담을 내용 |
|---|---:|---|
| 1. Introduction | .8쪽 | 숨은 목표 집단에서 공유 관계가 왜 어려운가, 두 단계 검증, 검증 완료된 기여만 열거 |
| 2. Related Work | .6쪽 | 선택적 parameter sharing, 역할 발견, 분산 critic consensus/distillation의 차이 |
| 3. Problem and Diagnostic Motivation | 1.0쪽 | 정보·통신 가정, 팀/역할/value 구분, 기존 JS 실패 진단 |
| 4. Method | 1.4쪽 | online evidence 추론, 보류/갱신, allocation, PPO+distillation, 알고리즘 상자 |
| 5. Experiments | 2.5쪽 | from-scratch 주결과, 식별/Oracle gap, 메커니즘·비용·실패 조건 |
| 6. Discussion and Conclusion | .7쪽 | 검증된 적용 범위, 정보 가정, 제한, 향후 과제 |

7쪽에 모든 과거 작업을 넣지 않는다. Windows 재부팅·W&B 경로·파일 정리·smoke 상세는 재현성 부록과 실행 이력으로 정리한다. 과거 curriculum 결과는 진단 동기 또는 조건부 적응 부록으로 배치한다.

추천 본문 그림/표:

1. 관계 추론→물리 후보 내 공유→local PPO 구조와 허용 정보 범위.
2. 같은 팀/역할 차이와 다른 팀/행동 유사성을 구분하는 진단 그림.
3. from-scratch 학습곡선과 seed별 효과 구간; 미완료 run은 별도 pilot 표시.
4. 추론 정확도·증거 coverage·학습 성능·실제 allocation을 함께 보여 주는 ablation 표.
5. agent별 onset 또는 성과 집중도와 비용 비교의 작은 패널.

### 제목 후보

원래 목표의 알고리즘과 반복 검증을 완성한 뒤 사용할 후보:

> **Inferring Latent Teams from Local Task Evidence for Selective Critic Sharing**

현재 진단 중심 초고에 사용할 수 있는 후보:

> **Policy Similarity, Latent Teams, and the Limits of Critic Sharing in Multi-Agent Reinforcement Learning**

‘효율 향상’, ‘robust’, ‘fully decentralized’, ‘general’ 같은 제목 표현은 각각의 증거와 정보 가정이 충족된 뒤 추가한다.

### 현재 증거만 사용한 영문 초록 초안

아래는 **탐색적 분석 원고용**이다. 미구현 방법이나 미래 성능을 과거형으로 쓰지 않았다. 특정 효과의 통계적 검증이나 제출 가능한 완성본을 뜻하지 않는다.

> Selecting communication partners in multi-agent reinforcement learning requires distinguishing similarity in behavior from alignment in objectives and compatibility of value predictions. We examine these distinctions in a partially observable warehouse environment with hidden team memberships and team-relative task observations. We analyze a policy-similarity-gated critic distillation scheme using frozen-policy diagnostics and controlled training comparisons. Across six checkpoint conditions derived from a single training seed, canonical policy-response similarity yields same-team AUROC values of 0.375–0.500, whereas locally witnessed task outcomes contain substantially stronger membership information. These diagnostics establish the availability of local evidence under the evaluated conditions, rather than an implemented online team-discovery solution. Comparisons initialized from a shared pretrained model show small and environment-dependent differences in learning-curve performance, without establishing improved final critic accuracy. An interrupted from-scratch pilot shows an early advantage for oracle team-restricted sharing, concentrated in one agent, but cannot establish a reproducible learning benefit. Together, these observations motivate separate evaluation of team identifiability, the value of oracle sharing, and the performance of inferred sharing relations. The study identifies limitations of using policy similarity as a proxy for latent team membership and outlines the evidence needed to connect relation discovery to learning improvements.

방법론 논문 최종본에서는 online 방법 설명과 완료된 여러 seed 실험의 효과 크기·불확실성을 넣어 다시 쓴다. 결과가 없는 단계에서 개선율 placeholder를 실측처럼 채우지 않는다.

### Introduction에 사용할 논리 순서

1. 여러 agent가 공존해도 서로 같은 목표를 갖는 것은 아니며, 어떤 경험·예측을 공유할지가 학습에 영향을 준다.
2. 같은 목표라도 역할이 다르면 행동이 다르고, 다른 목표라도 공통 이동 능력 때문에 행동이 비슷할 수 있다.
3. 따라서 정책 유사성으로 팀과 value 호환성을 한꺼번에 추론하는 것은 강한 가정이다.
4. 현재 warehouse 진단은 이 구분을 실제 실패 사례로 보여 준다.
5. 논문에서 제안하는 online 방법과 식별·Oracle 공유·학습 개선을 분리한 평가 설계를 설명한다.
6. 완료한 결과의 범위 안에서만 기여를 열거한다.

## 13. 관련 연구와의 차이를 확인할 관점

| 일차 자료 | 이미 다룬 것 | 본 연구가 명확히 할 차이 |
|---|---|---|
| [Christianos et al., SePS, ICML 2021](https://proceedings.mlr.press/v139/christianos21a.html) | 선택적 parameter sharing으로 효율과 다양성의 양립 | actor/critic 파라미터 공유와 숨은 목표 관계에 기초한 critic 예측 공유의 차이 |
| [Hu et al., HetDPS 공식 구현](https://github.com/Harry67Hu/HetDPS) | objective/policy 등의 이질성 구분, 거리 기반 동적 공유 | ‘이질성이 있으니 선택적 공유’만으로 차이가 약함. 허용 정보, 국소 증거, critic만 공유하는 가치 제시 |
| [Zhang et al., ICML 2018](https://proceedings.mlr.press/v80/zhang18n.html) | 통신망의 분산 actor-critic, critic consensus, 선형 근사 수렴 분석 | 통신망과 잠재 팀은 다름. 숨은 관계 추론, 개인 목표, 비선형 distillation과의 차이 |
| [Yu et al., MAPPO, NeurIPS 2022](https://arxiv.org/abs/2103.01955) | PPO 계열 MARL 성능과 구현 요인 | 중앙 critic의 추가 정보를 명시하고 구현 차이를 성능 원인과 혼동하지 않음 |
| [Wang et al., RODE, ICLR 2021](https://arxiv.org/abs/2010.01523) | action effect로 역할 발견·과제 분해 | 역할과 공통 목표 팀은 다른 잠재 변수. 필수 baseline 여부는 과제 정합성으로 판단 |

위 내용은 논문 초록·출판 페이지·공식 README 확인에 기초한 초기 위치 설정이며 완전한 신규성 조사는 아니다. HetDPS의 정식 논문명은 공식 README의 **Heterogeneity in Multi-Agent Reinforcement Learning**이다. 프로젝트의 구현 설명 제목을 그대로 서지 제목으로 쓰지 않는다. 원 논문의 모든 수식·실험을 이번에 정밀 검토했다고 주장하지 않는다.

MAPPO/SePS/HetDPS는 현재 smoke를 마친 이식 구현이며 완료된 성능 비교 결과는 없다. 논문에는 `SePS-style sharing with recurrent PPO`처럼 이식 조건이 드러나는 이름과 구현 차이 표를 사용한다. 공식 구현 재현 또는 작은 규모의 일치 검증이 별도로 필요할 수 있다.

## 14. 이번 성과와 미실시 사항

실시: MD 교차 분석, 실행 상태 읽기 확인, IJCAI 공식 일정과 선행 연구 일차 자료 확인, 논문 구성·주장/증거·실험 우선순위 문서화, 공용 기록 갱신.

미실시: 새 학습·추가 frozen 평가·tests 재실행, learned-team 구현, baseline 수정, 환경/보상/gate/probe 변경, 실행 중 학습 중단·재시작, 완성 논문 제출.

**다음 실증 판단점은 현재 from-scratch Oracle 비교가 독립 seed에서 재현되고, All보다 목표 관계를 선택하는 가치가 있는가다. 논문의 문제 설정·관련 연구·기존 진단은 지금부터 쓸 수 있지만, 주결과와 성공 주장은 그 검증과 online 방법의 완성에 맞춰 확정한다.**
