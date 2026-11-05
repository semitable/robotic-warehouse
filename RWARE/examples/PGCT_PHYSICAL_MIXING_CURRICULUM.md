# 09700 → 거리 제한 PGCT + shelf 혼합 커리큘럼

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\examples\run_pgct_09700_physical_mixing_curriculum.ps1
```

기본 초기 모델은 ex5 `policy-complete`, 4개 에이전트, 센서 범위 5,
seed 1 학습의 `checkpoint_09700.pt`입니다. actor와 critic 가중치를 모두
이어받아 미세 조정하며, 원본 체크포인트는 수정하지 않습니다.

| 단계 | 통신 | Shelf 혼합률 | Home ratio | 추가 학습 예산 |
| --- | --- | --- | --- | --- |
| 1 | physical, 센서 범위 이내 | 0% | 1.0 | 100만 스텝 |
| 2 | physical, 센서 범위 이내 | 12.5% | 0.875 | 200만 스텝 |
| 3 | physical, 센서 범위 이내 | 25% | 0.75 | 300만 스텝 |
| 4 | physical, 센서 범위 이내 | 50% | 0.5 | 400만 스텝 |

에이전트 4개, 기존 맵, 센서 범위 5를 유지합니다. 09700에 저장된 환경의
보상 설정과 네트워크 구조를 읽어 사용합니다. 학습률은 `1e-4`, entropy
계수는 `0.02`로 설정했습니다. 이 예산은 시작용 설정이며, 성능 검증을
마친 최적 커리큘럼은 아닙니다. 단계 전환은 성공률 기준이 아니라 예산 소진
기준입니다. 실제 스텝 수는 1024 스텝 rollout 단위로 올림됩니다.

## 거리 제한의 의미

모든 단계에 다음 설정을 명시적으로 적용합니다.

```text
communication_enabled = true
communication_topology = physical
comm_graph_mode = physical
communication_range = sensor_range = 5
peer_transfer_mode = pgct
```

연결 조건은 `PGCT 허용 AND max(abs(dx), abs(dy)) <= 5`입니다.
원형 거리 대신 11×11 정사각형 관측 범위와 같은 거리 기준을 사용합니다.
방향이나 shelf에 의한 가림을 추가한 모델은 아닙니다.

렌더러의 초록 선은 매 환경 스텝의 위치로 제한됩니다. 기존 학습 코드는
rollout 뒤 PGCT 전이 대상을 정할 때도 현재 이웃으로 gate를 마스킹하므로,
이전에는 가까웠어도 업데이트 시점에 범위를 벗어난 에이전트는 제외합니다.
optimizer 업데이트 중에는 환경 위치가 변하지 않습니다.

기존 PGCT 구현을 그대로 사용합니다: 정책 유사도로 관계를 정하고,
공통 probe에서 다른 에이전트의 critic 예측을 전이하는 보조 손실을 적용합니다.
상대의 실시간 관측을 actor 입력에 전달하는 별도의 메시지 네트워크를 추가한 것은 아닙니다.

## Shelf 혼합과 단계 간 연결

`balanced_soft_zones`로 두 팀의 shelf 총수를 유지하면서 구역 사이의
**shelf 팀 소속**을 섞습니다. Shelf 좌표 자체를 뒤섞지는 않습니다.
혼합 배치는 환경 reset 시 생성되며, 각 단계에서는 해당 혼합률을 유지합니다.

각 단계의 `final.pt`를 다음 단계의 `--init-checkpoint`로 사용합니다.
actor·critic 파라미터는 모두 전달하지만 **Adam 상태와 PGCT 그래프 통계는
매 단계 초기화**합니다. 완전한 학습 상태 resume이 아닌 단계별 미세 조정입니다.
과거 complete 그래프를 사용하지 않고 새 물리 이웃에서 관계를 다시 추정합니다.
각 단계의 초기 100 업데이트는 PGCT 전이 warmup이며, 최소 probe 횟수 조건도 적용합니다.
따라서 초반에 초록 선이 없는 것이 정상일 수 있습니다.

## 실행 옵션

```powershell
# 설정과 연결 경로만 생성: 학습하지 않음.
.\examples\run_pgct_09700_physical_mixing_curriculum.ps1 -DryRun

# 네 단계마다 64스텝만 실제 학습: CPU, W&B 및 영상 비활성화.
.\examples\run_pgct_09700_physical_mixing_curriculum.ps1 -SmokeTest

# W&B 없이 본 학습. 로컬 평가 영상은 계속 저장함.
.\examples\run_pgct_09700_physical_mixing_curriculum.ps1 -NoTrack

# 단계별 예산 변경.
.\examples\run_pgct_09700_physical_mixing_curriculum.ps1 `
  -StageTimesteps 2000000,3000000,4000000,5000000

# 필요할 때 6개 또는 8개로 확장. 증가한 에이전트는 기존 정책을 순환 복제함.
.\examples\run_pgct_09700_physical_mixing_curriculum.ps1 -AgentCount 6
```

기본 출력은 `runs/pgct09700_physical_mixing_<timestamp>/`입니다.
`-OutputDir`로 지정할 수도 있으며 기존 폴더에는 덮어쓰지 않습니다.

- `curriculum.json`: 전체 단계 설정, 전 단계 체크포인트 경로, 진행 상태.
- `stageXX_.../final.pt`: 단계별 최종 모델.
- `stageXX_.../checkpoint_*.pt`: 100 업데이트마다 저장한 모델.
- `stageXX_.../videos/`: 50 업데이트마다 및 마지막 평가에서 저장한 MP4.

본 학습은 W&B 기록을 기본으로 사용합니다. `-NoTrack`은 W&B만 끄고,
`-NoVideo`는 영상 저장도 끕니다. 기본 평가는 20회 × 최대 500스텝입니다.
시험 학습의 성공은 코드 경로·가중치 전달 검증이며 최종 성능의 증거는 아닙니다.
