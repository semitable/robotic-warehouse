# Canonical probe bank 시각화 — 실행 중 run `scratch_20261015_093349_stage2`

2026-10-15 Claude Code. 읽기 전용이며 학습에는 영향을 주지 않는다. 재현: `python output/env_view_20261015/probes/draw_probes.py`.

| 파일 | 내용 |
|---|---|
| `official/probe_contact_sheet.png`, `official/NNN_*.png`, `official/probe_manifest.json` | 저장소의 `rware.utils.probe_maps` 렌더러로 그린 48개 probe. 각 probe의 이름(의도)과 world 지도·agent별 시야가 함께 나온다 |
| `decoded_bank.png` | 실제 학습에 쓰이는 bank 벡터를 해독해 agent 관측(11×11)으로 그린 것. 빨간 제목은 다른 probe와 입력이 완전히 같은 경우 |
| `probes.csv` | probe별 고유 id, bank 안 중복 수, 방향, 적재/요청 적재/반납 단계, 위치, 보이는 요청 선반·목표·agent 수 |

## 확인한 것

- 저장소 builder로 같은 env 상태(reset seed 1)와 rng(seed+9173)를 써서 다시 만든 bank가 run의 `canonical_bank_e1_s1.pt`와 **완전히 같다.**
- 48개 probe 중 **고유 입력은 26개**다.
  - `dual_requested_*` 7종(home_pair, other_home_pair, opposite_axis, front_side, near_far, diagonal, cross_zone_choice)이 팀마다 같은 입력 하나로 겹친다(8개씩, 2그룹 = 16개).
  - goal 계열 3종(goal_pair, goal_zone_pair, lane_choice)이 팀마다 하나로 겹친다(4개씩, 2그룹 = 8개).
  - `neutral`은 2개씩 2그룹이다.
- 겹치는 원인(관찰 기반 추정): 해독한 시야에서는 요청 선반이 대부분 **시야 가장자리에 1개만** 보인다. world 지도에서 서로 다른 두 번째 선반 위치는 11×11 관측 밖에 있어 입력에 반영되지 않는 것으로 보인다. builder 코드로 확인하지는 않았다.
- 상태 구성: 빈 손 38개, 요청 선반 적재 10개(이 중 반납 단계 4개). 요청되지 않은 선반을 든 상태는 0개다.
- 관측은 팀 기준(주황 = 내 팀 요청, 초록 = 내 목표)이다. 다른 agent의 팀은 표시되지 않는다.

## 해석 시 주의

이름이 다른 7가지 시나리오가 같은 입력이므로, 해당 상황은 JS 평균에서 의도보다 큰 가중치(8/48)를 받는다. E04·E07에서 본 "중복 제거 시 유사도 순위 변화"의 직접적인 원인이다. probe 설계 의도("두 요청 선반 중 선택")는 현재 관측 범위에서 대부분 사라진다.
