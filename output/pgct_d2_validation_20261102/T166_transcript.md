## 원문 기록 T166 — 2026-11-02 KST, Codex (정확한 시각 기록하지 않음)

### 사용자 — 원문

현재 저장소에서 다음 자료를 확인하세요.

- `CLAUDE.md`
- `docs/SESSION_HANDOFF.md`
- `DECISIONS.md`
- `EXPERIMENTS.md`
- `RESEARCH_STATE.md`
- `CONVERSATION_LOG.md`
- `latent_team/reward_probe_ac_v2.py`
- `latent_team/max_gae.py`
- `latent_team/train_from_scratch.py`
- `latent_team/exp_b_oracle_consensus.py`
- `examples/train_recurrent_ippo_consensus.py`
- `docs/MAX_GAE.md`
- 관련 실험 로그와 기존 논문 초안

특히 D91\~D98, E57\~E66, T151\~T163을 확인하세요.





Checkpoint에 `team_goal_positions`가 저장되어 있어(1M: `[[0,10]], [[9,10]]`, seed 1) 계획 단계에서 비교할 수 있습니다. 이제 `max_gae.py`를 고칩니다.

Now the advantage blend, the explained-variance helper, and the update wiring.

Now `train_from_scratch.py`: the `shrink` option and the goal-layout check for branches.

Now the tests: update the fake checkpoint in the plan helper and add tests for the four fixes.

19개 모두 통과했습니다. 기존 A+C·latent_team 테스트도 함께 돌립니다.

기존 38개와 새 19개, 모두 57개가 통과했습니다. 이제 실제 trainer smoke(64 step)로 donor를 끈 동등성과 1M 분기 동작을 확인합니다.

Three checks ran. Now comparing weights and reading the branch log rows.

donor를 끈 `oracle_max` 실행은 IPPO와 최종 가중치가 비트 단위로 같았고, 짝수 seed 분기는 계획 단계에서 거부됐습니다. 이제 권고 명령(`--ac-gate v2`, seed 1·3·5)도 64 step smoke로 돌려 보겠습니다.

12개 smoke job이 모두 끝났고 초기 가중치도 같습니다. 이제 문서와 연구 기록을 갱신합니다.

필수 수정 1–4를 구현했습니다. trainer·`rware/`·`examples/`는 건드리지 않았습니다. 학습은 64 step smoke만 돌렸습니다.

## 바꾼 것

| # | 내용                                                                                                                                                                                                       | 위치                                                                                                                                   |
| - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | 분기 goal 배치 검사. `--init-checkpoint`의 `team_goal_positions`와 job 설정이 다르면 계획 단계에서 거부합니다. 폴더도 만들지 않습니다                                                                                                       | [train_from_scratch.py](https://claude.ai/epitaxy/latent_team/train_from_scratch.py) `checkpoint_goal_layout`, `check_branch_layout` |
| 2 | `shrink` 척도: Δ에 s = min(1, std(A_self)/std(Δ))를 곱합니다. 설계식 f = V + η·s·min(b, c)에서 η·s ≤ η인 경우입니다. `relative`는 기록용으로만 남겼습니다                                                                               | [max_gae.py](https://claude.ai/epitaxy/latent_team/max_gae.py) `blended_advantages`                                                  |
| 3 | rollout 첫 7 step(첫 reset 전)에서는 Δ = 0, 즉 actor advantage = A_self                                                                                                                                         | `blended_advantages(valid=...)`, `install.update`                                                                                    |
| 4 | 모든 agent의 critic explained variance를 매 update 자기 rollout에서 계산해 로그로 남깁니다(`update/max_gae_critic_ev_agent_k`, `max_gae.jsonl`의 `critic_ev`). 실제 반영 비중 `max_gae_eta_eff`도 추가했습니다. donor 자격 조건으로는 아직 쓰지 않습니다 | `explained_variance`, `install.update`                                                                                               |

**기본값 변경**: `--max-gae-scale`의 기본값을 `relative`에서 `shrink`로 바꿨습니다. relative로 실행한 run은 없어서 기존 결과에는 영향이 없습니다. ippo·pgct 등 기존 모드의 동작은 그대로입니다.

## 검증

- **pytest**: `tests/test_max_gae.py` 19개(새 8개)와 기존 A+C·latent_team 테스트 38개, 모두 57개가 통과했습니다. 새 테스트는 다음을 확인합니다.&#x20;
  - shrink는 raw보다 커지지 않고, 뒤처진 agent에서는 η·std(A_self)로 줄입니다.
  - b → 0이면 shrink는 IPPO로 돌아가고, relative는 그렇지 않습니다.
  - 가린 step은 A_self 그대로이고, 나머지 step은 바뀌지 않습니다.
  - explained variance 계산이 맞습니다.
  - seed 1 checkpoint에 seed 2를 주면 계획이 거부됩니다.
- **실제 trainer smoke(64 step, scratchpad)**:&#x20;
  - donor를 끈 `oracle_max`는 IPPO와 최종 가중치가 비트 단위로 같습니다(EV 계산을 넣은 뒤에도 같음).
  - 1M checkpoint 분기, 32 step rollout × 2 update:&#x20;
    - 뒤처진 a0·a1: `oracle_max`(shrink)는 η_eff .04–.08, 변화량 .41–.59였습니다. D1은 변화량 5.3–14.7이었습니다.
    - 앞선 a2·a3: 변화량 0입니다.
    - a5는 update 2에서 Δ가 작아 줄이지 않았고(η_eff .5), 변화량은 .35였습니다.
  - 아래 분기 명령과 같은 형태(`--ac-gate v2`, 4 모드 × seed 1·3·5)의 12 job이 모두 끝났고, 초기 가중치가 같았습니다.
  - CLI에서 seed 2를 주면 오류로 멈춥니다: `goal layout [[[9, 10]], [[0, 10]]] differs from the checkpoint's [[[0, 10]], [[9, 10]]]`
- smoke의 EV는 32 step rollout이라 음수가 많습니다(−26.6–.27). 실제 판단은 1,024 step run의 값으로 합니다.

## 분기 실험 실행 명령 (T154 명령 대체, 앱 밖 PowerShell에서 사용자 실행)
```powershell
Set-Location 'C:\RWARE\MARL_RWARE\third_party\robotic-warehouse'; $run = Join-Path 'runs' ('maxgae_br1M_' + (Get-Date -Format 'yyyyMMdd_HHmmss')); & 'C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe' latent_team/train_from_scratch.py --env-variant b6_corner_indiv_pen005 --reward-probe-ac --ac-gate v2 --team-reward individual --modes ippo oracle_max oracle_maxd1 allk_max --seeds 1 3 5 --timesteps 500000 --transfer-warmup 0 --init-checkpoint runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt --track --stage-label maxgae_branch --output-dir $run

```

- T154와 달라진 점: seed 1·2·3 → 1·3·5, `--ac-gate v2` 추가, D2 척도는 기본값 shrink입니다.
- 12 job이 순차로 돌고, seed 1의 4개 모드가 먼저 끝납니다. job당 약 5.5시간, 전체 약 66시간으로 추정합니다(T154 추정).
- 판정 규칙은 D98대로 미리 고정합니다. 뒤처진 agent a0·a1·a4의 평가 배송을 보고, 앞선 agent a2·a3·a5의 배송이 떨어지지 않는지를 안전 지표로 봅니다.
- `max_gae_critic_ev_agent_*`를 보면 donor critic 성숙 조건을 어디에 둘지 정할 수 있습니다.

## 아직 안 한 것

- PGCT-D2 단계(필수 5–8): v2 `approved()`, donor 규칙 gate, `pgct_max`·`oracle_mature_max` 모드, 감사 로그.
- donor critic 성숙 조건(EV 로그를 본 뒤 결정).
- a0·a4처럼 보상 사건이 없는 agent를 어떻게 다룰지(연구 범위 결정).

기록은 `DECISIONS.md` D99, `EXPERIMENTS.md` E67, `RESEARCH_STATE.md`, `docs/MAX_GAE.md`, 대화 기록 T164에 남겼습니다. 커밋은 하지 않았습니다.



클로드를 통해 여기까지 진행됐어

필수 수정 1–4를 구현했습니다: 분기 goal 배치 검사, D2 shrink 척도(기본값), 첫 7 step Δ=0, critic EV 로그. pytest 57개가 통과했고, smoke에서 IPPO와의 비트 동등성과 seed 2 거부를 확인했습니다. 분기 실험은 아직 시작하지 않았고 커밋도 하지 않았습니다.\
\
분기 실험이 도는 동안 PGCT-D2(필수 5–8)를 구현해줘

### Assistant — 진행 안내 (1)

먼저 인수인계와 D91–D99·E57–E67·최근 대화, 현재 코드를 확인해 PGCT-D2의 필수 5–8 범위를 정리하겠습니다. 실행 중인 실험은 건드리지 않고, 구현과 테스트·smoke 검증을 진행하겠습니다.

### Assistant — 진행 안내 (2)

필수 5–8은 v2 승인 API, 승인에 따른 donor 선택, `pgct_max`·`oracle_mature_max` 모드, 실제 donor 감사 로그입니다. 현재 Max-GAE는 IPPO 경로로 실행되어 v2 학습이 꺼져 있습니다. 새 모드에서만 이를 연결하고, 기존 분기 모드의 설정과 동작은 유지하겠습니다.

### Assistant — 진행 안내 (3)

분기 폴더 `runs/maxgae_br1M_20261102_165216`의 로그에서 IPPO seed 1의 6,144 step 진행을 확인했습니다. 프로세스 조회는 권한 제한으로 실패했으며, 실행에는 손대지 않았습니다.

새 모드의 v2 비교도 Oracle과 같은 rollout 접촉 기준으로 연결하겠습니다. critic EV 임계값은 아직 연구 판단이 남아 있어 기본으로 강제하지 않고, 선택 옵션과 로그를 추가하겠습니다.

### 사용자 — 원문 (작업 중 추가 전달)

토큰을 다 써서 PGCT-D2 구현은 Codex로 할꺼야 알고있어

실행됨 명령 1개

알겠습니다. PGCT-D2는 Codex에서 진행하시는 것으로 기록했습니다(T165).

Codex가 참고할 곳은 다음과 같습니다.

- 구현 명세: `DECISIONS.md` D98의 필수 5–8(`approved()`, donor 규칙 gate, `pgct_max`·`oracle_mature_max`, 감사 로그)과 6절 계획
- 현재 상태: D99, E67, docs/MAX_GAE.md
- 주의할 점: v2 `gate`는 update 직전 이웃이 아니면 지워지므로 donor로 그대로 쓰면 안 됩니다. IPPO 경로에서는 controller가 `active=False`로 불려 비교가 일어나지 않습니다(`exp_b_oracle_consensus.py:160`).

### Assistant — 진행 안내 (4)

확인했습니다. 말씀하신 D98·D99·E67을 기준으로 여기서 계속 구현하고 있습니다. 접촉이 끊겨도 보존되는 `approved()`와 Max-GAE의 v2 직접 호출을 추가했으며, 이제 실행 모드와 검증을 연결하겠습니다.

### Assistant — 진행 안내 (5)

새 모드와 감사 로그 연결을 마쳤습니다. 지금은 접촉·성숙 조건 테스트와 실제 trainer의 64 step 동등성 검증을 돌리고 있습니다. 첫 실행은 테스트 임시 폴더의 상위 경로가 없어 일부 설정 단계에서 실패했으며, 별도 검증 폴더를 만든 뒤 다시 실행 중입니다.

### Assistant — 진행 안내 (6)

승인이 닫힌 `pgct_max`와 `oracle_mature_max`는 1M checkpoint에서 각각 64 step 실행한 뒤 IPPO와 최종 가중치가 비트 단위로 같았습니다. v2 예측기가 실제로 학습된 것도 확인했습니다.

현재는 정답 팀 승인을 넣은 진단 실행이 `oracle_max`와 같아지는지, EV 필터와 warm-up이 donor를 막는지도 추가로 검증하고 있습니다.

### Assistant — 진행 안내 (7)

기존 57개와 새 15개, 총 72개 테스트가 모두 통과했습니다. 1M 공통 checkpoint·seed 1에서 실행한 64 step smoke에서도 정답 승인을 넣은 두 새 모드는 `oracle_max`와 비트 단위로 같았고, EV 필터로 donor를 막으면 IPPO와 같았습니다.

문서와 연구 기록을 정리하고 있습니다. PGCT-D2 본 학습은 시작하지 않았으며, 검증 로그와 checkpoint는 별도 폴더에 보존했습니다.

### 수행 기록 — 원문 대화와 별도

- 사용자 요청 파일과 D91–D99/E57–E67/T151–T165 관련 내용, trainer의 관측·update·checkpoint 연결부 확인. 기존 논문은 `output/pgct_progress_paper_20261023/main.tex`의 방법·문제 설정 부분, `output/ijcai_method_paper_20261018/README.md`의 현행 원고·결과 한계 안내를 읽었다. 논문 원고 전체를 읽거나 수정·컴파일한 것은 아니다. 별도 웹 대화를 조회한 것이 아니라 사용자 붙여넣기와 로컬 대화 기록을 읽었다.
- 구현 D100, 검증 E68. PGCT-D2 새 15개 포함 72 tests 통과. 실제 trainer seed 1, 1M 공통 checkpoint 분기, 64 step/job 9 job smoke. 본 학습 결과로 세지 않는다.
- 코드 변경은 `latent_team/` 세 파일과 새 테스트, 문서·기록. trainer·환경·examples 수정·기존 실행 중단·새 본 학습·커밋 없음. 프로세스 조회 실패는 CIM 권한 제한이며 자동 승인 심사 거부가 아니다.
- 연구 기록·문서를 쓰기 직전 재확인했으며 Claude가 추가한 T165를 보존했다. `RESEARCH_STATE.md`, `DECISIONS.md` D100, `EXPERIMENTS.md` E68, `docs/MAX_GAE.md`, 이 T166 및 통합 사본 갱신.

### Assistant — 최종 답변

PGCT-D2 필수 5–8을 구현했습니다.

- 접촉과 독립적인 v2 `approved()`·성숙 조회
- rollout 접촉과 승인을 결합한 donor 선택
- `pgct_max`·`oracle_mature_max` 실행 모드
- 실제 donor·같은 팀/다른 팀 arc·성숙·승인 범위 감사 로그

**테스트 72개가 통과했습니다.** 공통 1M checkpoint·모서리/개인 보상·seed 1에서 64 step씩 실행한 smoke 9개로, 승인 차단 시 IPPO와의 비트 동등성과 정답 승인 주입 시 Oracle-max와의 비트 동등성을 확인했습니다. 학습 성능 검증은 아직입니다.

critic EV 필터는 옵션으로 구현했으며 **기본 비활성**입니다. 임계값은 분기 로그를 보고 정해야 합니다.

[구현·사용법](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/docs/MAX_GAE.md) · [검증 기록](C:/RWARE/MARL_RWARE/third_party/robotic-warehouse/output/pgct_d2_validation_20261102/README.md)

기존 분기 실행은 건드리지 않았고, 새 본 학습이나 커밋은 하지 않았습니다.
