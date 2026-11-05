# ex5 기준 CNN PGCT

기준 문서: 사용자가 정정한 `C:/Users/95101/Downloads/ex5.pdf`.
ex6용 방향성 FoV, 전방 관측 마스크, 시간 방향 연결성 진단은 제거했다.
기존 실험 결과와 체크포인트는 보존하며 새 실험명에는 `ex5_`를 붙인다.

## 구현과 PDF의 대응

| 항목 | 현재 실행 설정 |
|---|---|
| 태스크 관측 | 방향과 무관한 SR5 정사각형 11×11, agent별 CNN/GRU |
| 후보 그래프 | 기본 complete. 실제 전달은 PGCT gate로 제한 |
| Canonical Q, pp.15–17 | 모든 정책에 동일한 고정 shared objective bank, zero probe hidden; online 상태를 사용하지 않음 |
| JS 거리, pp.18–20 | 전체 action distribution의 JS를 probe/time에 균등 평균 |
| EMA, pp.23–24 | beta=0.35, affinity=exp(-D/0.25)는 진단 점수이며 확률이 아님 |
| Gate, p.25 | warm-up 뒤 `D_ema < 0.12`인 쌍에 binary gate; 대각선 0 |
| Warm-up | PPO update 100회; 5 update마다 probe, 쌍별 측정 최소 8회 |
| Critic transfer, pp.26–28 | receiver별 gate 정규화, 고정 donor target에 stop-gradient, local value loss + 0.01 peer loss |
| 독립 학습 | 그래프/probe/peer loss/consensus 모두 OFF; 각 agent의 로컬 PPO는 계속 수행 |
| 영상 | 허용된 연결만 표시. 대칭 gate는 선 하나; 독립 모드는 선 없음 |

48개 probe, Lp=1을 기본으로 둔다. 이는 zero-state 단일 입력 비교이며 긴 recurrent history를 검증한 설정은 아니다. 48개는 PDF가 지정한 최적값이 아니라 기존 작업 시나리오 bank의 크기다. 같은 bank를 actor 거리와 critic 전달에 재사용하며 `canonical_probes.pt`에 저장한다. 실제 팀 라벨에 따라 정책별 probe를 달리 주는 `agent-team` 옵션은 이 실행기에서 허용하지 않는다.

threshold 0.12, warm-up 100, bank 크기는 실험 파라미터다. 초기 랜덤 정책은 D≈0이므로 **warm-up 이후에도 잘못 연결될 수 있다**. ex5는 이를 방지한다는 보장을 하지 않는다. 이전에 추가했던 KL-to-uniform 정보량 gate는 기본값 0으로 해제했다. Python CLI의 soft gate와 정보량 gate는 별도 ablation 옵션이며 ex5 기본 실행에서는 사용하지 않는다. 라벨 기반 threshold 보정도 하지 않는다.

반납 상태 관측, 동료 이동에 의한 잘못된 진행 보상 방지, timeout bootstrap, agent별 학습 진단과 4/8/16명·선반 혼합 설정은 유지했다. 이것들은 ex6 가정에 의존하지 않는 환경/학습 수정이다.

## 실행

아래는 PowerShell 현재 창에만 실행 정책을 적용한다. 두 학습은 순차 실행된다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# ex5 PGCT: 완전 후보 그래프 + 학습된 gate
.\examples\run_semantic_cnn_marl_pgct_from_scratch.ps1 -Run policy-complete -CommunicationEnabled $true -AgentCount 4 -SensorRange 5 -Seed 1 -TotalTimesteps 10000000

# 동일 환경의 independent PPO
.\examples\run_semantic_cnn_marl_pgct_from_scratch.ps1 -Run policy-complete -CommunicationEnabled $false -AgentCount 4 -SensorRange 5 -Seed 1 -TotalTimesteps 10000000
```

`-AgentCount 8` 또는 `16`으로 변경할 수 있다. W&B를 사용하지 않으려면 `-NoTrack`을 추가한다. 영상은 run 폴더의 `videos/final.mp4`, 전달 행렬과 거리/count는 `communication.jsonl`에 기록된다. 영상 연결은 허용된 peer 관계를 뜻하며 환경 매 step마다 optimizer transfer가 수행된다는 의미는 아니다.

비교군: `-Run oracle`(같은 objective), `-Run wrong`(다른 objective), `-Run unrestricted`(모든 peer), `-Run none`(독립). Oracle/Wrong만 팀 라벨을 사용하며, 학습용 PGCT는 이를 읽어 gate를 만들지 않는다. Oracle/Wrong/unrestricted는 warm-up 없이 해당 전달 관계를 사용한다. `policy-physical`은 별도의 거리 제한 ablation이다. `-Run all`은 여섯 실험을 순차 실행한다.

선반 소유권을 점차 섞는 커리큘럼은 [PGCT_PHYSICAL_MIXING_CURRICULUM.md](PGCT_PHYSICAL_MIXING_CURRICULUM.md)의 실행기를 사용한다. 과거 일반 혼합 커리큘럼 실행기(`run_semantic_cnn_mixing_curriculum.ps1`)는 2026-10-13 사용자 요청으로 삭제했으며 git 이력에 남아 있다.

각 stage는 고정 예산이며 이전 actor/critic을 가져오고 optimizer는 새로 시작한다. 성능 기준 자동 승급은 아니다. 새 bank는 stage별로 생성된다.

## 검증 범위

ex5의 pp.29–33은 같은 checkpoint·optimizer·batch에서 transfer/independent branch를 비교해 utility, Spearman correlation, negative transfer를 실험으로 검증하라는 내용이다. 위의 두 독립 학습 run만으로 이 matched-branch 검증을 대신할 수 없다. 이번 변경은 학습 방법과 비교군의 구현 정렬이며, 장기 수렴·전달 이득·optimal probe 수를 입증하지 않는다. matched-branch utility 분석은 아직 구현하지 않았다.

기존 FoV 체크포인트의 이어 학습을 ex5 from-scratch 결과로 취급하면 안 된다. 실행 중인 프로세스는 파일 수정이 자동 반영되지 않으므로 새 실험은 위 명령으로 시작한다.

검증 결과: 전체 pytest 130개 통과. 4-agent CNN으로 policy/none/wrong/unrestricted 각 32-step smoke run 완료, 실제 전달 arc 수 12/0/8/12 확인, policy/none 영상 저장 및 육안 확인. smoke에서는 전달 경로 검증을 위해 warm-up=0/min-count=1을 사용했으며 정식 launcher의 100/8 설정과 구분한다.
