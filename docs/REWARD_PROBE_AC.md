# A+C 보상 프로브 구현 — 2026-10-23 Codex

사용자 승인 범위는 A+C를 가능하게 하는 코드 수정이다. 본 학습 실행과 실행 중인 프로세스 중단은 하지 않았다. 기존 기본 실행은 그대로이며 `--reward-probe-ac`로 새 경로를 선택한다.

## 구현과 정보 흐름

1. 정책·critic·평가의 `self_x`, `self_y`를 0으로 마스킹한다. 입력 차원은 보존한다. 국소 지형과 방향은 남는다. 이전 checkpoint가 좌표 없는 조건을 학습했다고 간주하지 않는다.
2. 각 agent의 실제 `(o_i, a_i, o'_i, r_i)`를 환경 reset 전에 캡처한다. 개인 scalar `r_i`만 해당 agent의 예측기에 들어가며 합산·공유하지 않는다. 평가 rollout은 학습 메모리에 넣지 않는다.
3. 공통 질의에서는 절대 위치, 개인 요청 채널 4, 선반 기준 목적지 채널 12, 개인 요청·반납·선반 home ego 정보를 제외한다. 자기 loaded/direction/highway와 반경 2칸 안의 공통 물리 특징(벽·통로·선반·agent·generic goal)은 남긴다. 자기 요청 표시를 송신자에게서 복사하지 않는다.
4. C bank는 실제 지역 전이 템플릿이다. A bank는 자기에게 보였던 장면에서 NOOP/LEFT/RIGHT/FORWARD를 가상으로 수행한 전이다. 다른 agent는 NOOP으로 두는 가정이며, FORWARD는 실제 보이는 빈 앞칸으로만 구성한다. 바깥 테두리 한 칸을 더 관측해야 하며 숨은 지형을 채워 넣지 않는다. 가상 pickup이나 보상 라벨은 생성하지 않는다. 운반 상태에서 관측된 goal에 진입하는 가상 배송 전이는 포함할 수 있다.
5. 두 현재 물리 이웃은 자기 bank의 공통 물리 질의만 교환한다. 각자는 자기 경험에서 같은 관측 전이를 찾아 자기 개인 보상 조건부 평균을 예측한다. 이것이 `phi_i(q)`의 보수적인 구현이다. 숨은 팀 ID, 선반 소유 map, goal 소유 map, 절대좌표로 대응시키지 않는다.
6. 쌍별로 양쪽 모두 지지하는 질의에서 개인 보상 응답 차이의 평균과 EMA를 계산한다. 충분한 샘플·episode·질의·새 증거가 필요하며, 보상 변동·표준오차가 크거나 두 응답이 모두 평탄하면 보류한다. 승인된 현재 이웃만 기존 critic 증류의 실제 donor가 된다.

예측기는 **비모수 경험 조건부 평균**이다. CNN이 아니다. 대응/정보 접근/불확실성 처리를 검증하기 위한 실행 가능한 첫 구현이며, 안 본 사건의 CNN 일반화가 해결됐다는 주장이 아니다. 같은 지역 물리 패턴에 숨은 선반 소유가 섞이면 보상이 모호할 수 있다. 분산이 커지면 보류하지만, 아직 관측하지 않은 모호함까지 감지할 수 있다는 보장은 없다.

## 실행기

다음은 명령 예시이며 본 학습을 실행한 기록이 아니다.

```powershell
python latent_team/train_from_scratch.py --env-variant b6_corner_indiv_pen005 --modes ippo pgct oracle all --reward-probe-ac --team-reward individual --seeds 1 2 --timesteps 5000000 --plan-only --output-dir runs/ac_new_plan
```

- `--plan-only`는 manifest와 공통 critic bank만 만들고 학습하지 않는다. 기존 폴더가 있으면 거부한다. 새 job 이름에는 `_ac_`가 붙는다.
- 옵션은 `--ac-config <새 JSON 파일>`로 `ACConfig` 필드를 덮어쓴다. 결정한 값은 manifest에 저장한다. 기본값은 연구로 검증된 최적값이 아니다.
- A+C 계획은 후보의 옛 팀 보상 fallback보다 개인 보상을 우선한다. 명시적인 `--team-reward team` 조합은 거부한다. 실행 시에도 개인 보상을 다시 검사한다.
- IPPO/Oracle/All/Wrong 비교군도 같은 좌표 마스크를 쓴다. PGCT에만 A+C gate가 적용된다. PGCT의 내부 `graph_mode=unrestricted`는 준비 경로일 뿐이며 실제 donor는 A+C callback의 `actual_allocation`이다. 정책 JS gate는 사용하지 않는다.
- 기존 canonical bank는 critic 증류용으로만 유지하고 좌표를 마스킹한다. reward gate에는 이 bank의 팀 기준 채널이나 생성에 쓰인 팀 정보가 들어가지 않는다. 통신의 실제 송수신 비용 및 critic bank 자체의 분산 생성은 이번에 새로 검증하지 않았다.
- `transfer.jsonl`의 `ac_pairs`에 지원 질의 수·virtual 지원 수·거리·EMA·오차·변동·승인 이유를 기록한다. 정답 same/cross 팀 감사는 기존 로거에서 사후 수행하며 gate 입력과 분리돼 있다.
- checkpoint에 `reward_probe_ac`(메모리·gate 상태)와 `ac_observation`을 저장한다. 기존 실행기의 init-checkpoint는 완전한 학습 재개가 아니다. 저장된 AC 상태를 자동 복원하는 resume 기능은 이번에 추가하지 않았다.
- 독립 평가 스크립트에서도 `reward_probe_ac.install(..., learn=False)` 또는 `LocalObservationEnv`를 적용해야 좌표가 복원되지 않는다. 기존 평가 스크립트가 checkpoint 표지만 보고 자동 적용하지는 않는다.

## 검증 범위와 연구상의 한계

`tests/test_reward_probe_ac.py`는 자기 보상 분리, 공통 표현의 private/좌표 불변성, 가상 전이와 실제 환경 1-step 일치, 충분한 합성 경험에서의 반대 보상 응답, 미지/모호/평탄 사건의 보류, 물리 범위 제한, 실제 donor override 전달과 checkpoint 저장을 검증한다. runner 통합 검사에서는 PPO 최적화를 stub으로 바꾼다.

## Gate v2 — 2026-10-26 Claude Code

`--reward-probe-ac --ac-gate v2`로 선택한다(`latent_team/reward_probe_ac_v2.py`, job 이름 `*_acv2_*`). 기본값은 v1이며 v1 코드·옵션·job 이름은 바꾸지 않았다. 관측 마스킹, 개인 보상 강제, critic 증류 경로는 v1과 같다.

- 바꾼 이유(오프라인 측정 `output/ac_offline_20261026/`): v1은 정확히 반복된 공용 전이만 답하고 보상 분산이 크면 보류한다. 배송 장면은 같은 공용 전이에서도 요청 여부(숨은 정보)에 따라 +1/0이 섞여 std ≈ .45였고, 판별 사건이 모두 보류됐다.
- 예측기: agent별 작은 MLP `r̂_i(공용 before, action, 공용 after)`. 자기 전이 버퍼(20,000)와 자기 보상 사건 버퍼(|r| ≥ .5, 512)로만 학습하고 사건을 25% 과표집한다. P=5 update마다 50 step 학습한다(warm-up 중에도).
- probe: 두 이웃 각자의 최근 자기 보상 사건 공용 관측(C)과 그 좌우 대칭(A). 보상은 교환하지 않는다.
- 보류: 자기 사건 20개 이상, 자기 사건에 대한 평균 예측 ≥ .3이 아니면 그 agent는 판정하지 않는다. 보상을 받아 본 적 없는 agent는 누구와도 묶이지 않는다.
- 거리: `Σ|a−b| / Σ(|a|+|b|)`(0 = 같은 응답, 1 = 겹치지 않음), EMA β .35, 3 round 이상, `< .5`이면 같은 팀. 데이터로 고르지 않은 척도 무관 기준이다.
- 대칭 gate다(비대칭 donor 규칙 없음). 물리 이웃만 비교하고 접촉이 끊기면 해제한다.
- `contact_snapshot`(기본 False, D103): 켜면 rollout 시작 때 agent별 최근 사건 probe·성숙·예측기 버전을 고정하고, update에서 그 고정본으로 먼저 비교한 뒤 재학습한다. PGCT-D2 모드(`pgct_max`, `oracle_mature_max`)는 rollout 접촉으로 비교하므로 계획 단계에서 자동으로 켠다. 기존 v2 증류 모드는 그대로다.
- 난수: 예측기 초기화·표본 추출은 별도 RNG를 쓴다. trainer 난수 흐름을 바꾸지 않으므로 첫 전이 전까지 IPPO와 같은 학습이다(테스트로 확인).
- 기록: `update/ac_v2_*`(W&B·`metrics.jsonl`) — 성숙 agent 수, 비교·승인 쌍, gate arc, 예측기 손실, 평균 거리. `ac_v2_audit_*`는 env info의 팀 ID로 사후 집계한 같은 팀/다른 팀 승인·거리이며 gate 입력이 아니다. `transfer.jsonl`의 `ac_pairs`, checkpoint의 `reward_probe_ac_v2`(예측기 가중치·사건 버퍼·EMA).
- 검증: `tests/test_reward_probe_ac_v2.py` 7개(대칭 변환, 같은 팀 승인·다른 팀 거부·미성숙 보류, 접촉·비활성, 전역 RNG 불변, 팀 보상 거부, 계획 opt-in, runner 연동). 실제 trainer smoke(`--smoke-test`, 64 step) 통과. 본 학습은 하지 않았다.

오프라인 정답 검사에서만 환경 내부를 사용해 동일한 물리 배송을 두 운반자 팀에 구성한다. 이는 운영 gate의 정보원이 아니다. 실제 정책으로 수집한 경험에서 양쪽 팀이 충분한 배송 사건을 배우는지, 팀 발견 AUROC, critic 전이 성능 향상은 본 실험 전까지 미검증이다. 모든 장면이 관측상 대칭이거나 교차 사건 경험이 없으면 gate가 열리지 않을 수 있다. `[1,0]/[0,1]`을 강제로 출력하지 않는다. 보상 예측 대상은 배송 성분만이 아니라 shaping을 포함한 각자의 실제 개인 scalar 보상이다.
