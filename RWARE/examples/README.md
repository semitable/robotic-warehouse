# RWARE IPPO / PGCT 실행 코드

현재 연구에 사용하는 학습, 평가, probe 도구를 정리한 목록입니다.

| 파일 | 용도 |
|---|---|
| `train_recurrent_ippo_consensus.py` | 공통 recurrent IPPO 학습기와 PGCT probe / critic transfer 구현 |
| `run_semantic_cnn_marl_pgct_from_scratch.ps1` | IPPO(`-Run none`), PGCT 및 oracle/wrong/unrestricted 비교군 실행; 체크포인트 초기화도 지원 |
| `env_kwargs_stage2_pgct_success_unblock_zones_individual.json` | 현재 실행기의 기본 환경·보상 설정 |
| `train_pgct_physical_mixing_curriculum.py` | 9700 모델에서 시작하는 PGCT 물리 이웃·shelf 혼합 커리큘럼 |
| `run_pgct_09700_physical_mixing_curriculum.ps1` | 위 PGCT 커리큘럼의 PowerShell 실행기 |
| `train_pgct_spatial_experiments.py` / `run_pgct_spatial_experiments.ps1` | 같은 초기 모델에서 3개 맵 × PGCT/IPPO를 독립적으로 실제 학습하는 비교 실험 |
| `train_pgct_transfer_experiments.py` / `run_pgct_transfer_experiments.ps1` | 기존 맵에서 Stage03 이후 on/off, 전달량 통제 random donor, 3–4 제거와 probe/critic 진단 |
| `plot_pgct_probe_charts.py` | 공통 probe 행동확률·pair별 JS 색상 차트와 학습 곡선; 기존 로그도 읽기 전용으로 지원 |
| `diagnose_pgct_rollout_compatibility.py` | 저장된 공통 observation 이력을 replay하여 recurrent policy JS와 critic 예측 차이를 추론만으로 비교 |
| `train_hard_rware_experiments.py` / `run_hard_rware_experiments.ps1` | 장애물·배송 제약 환경의 분리 실행기 ([HARD_RWARE_EXPERIMENTS.md](HARD_RWARE_EXPERIMENTS.md)) |

공통 환경, semantic 관측, canonical probe 생성 코드는 `../rware/`에 있고,
회귀 검증은 `../tests/`에 있습니다. 잠재 팀 추론·oracle consensus 검증은 `../latent_team/`에 분리돼 있습니다.

## IPPO 비교용 사전학습

다음 명령은 팀 shelf가 완전히 분리된 환경에서 PGCT 없이 학습합니다.
학습률은 실행기에 지정된 `2e-4`이며, 기존 PGCT 사전학습과 같은
4 agents, sensor range 5, home ratio 1.0, entropy 0.05, seed 1을 사용합니다.

```powershell
Set-Location C:\RWARE\MARL_RWARE\third_party\robotic-warehouse
.\examples\run_semantic_cnn_marl_pgct_from_scratch.ps1 `
  -Run none -AgentCount 4 -SensorRange 5 -ShelfHomeRatio 1.0 `
  -TotalTimesteps 9932800 -EntropyCoef 0.05 -Seed 1 -RunNameSuffix off
```

PGCT 사전학습은 같은 실행기에 `-Run policy-complete`를 사용합니다.
`-Run policy-physical`은 물리 이웃만 후보로 하는 비교 설정입니다.
이 실행기는 `-InitCheckpoint`를 지정하면 actor와 critic 가중치를 모두 초기화합니다.
Optimizer와 rollout 상태까지 복원하는 완전한 resume은 아닙니다.

## 커리큘럼과 평가

현재 stage04 결과를 만든 PGCT 커리큘럼의 상세 설정은
[PGCT_PHYSICAL_MIXING_CURRICULUM.md](PGCT_PHYSICAL_MIXING_CURRICULUM.md)에 있습니다.
이 실행기의 단계별 기본 예산은 1M / 2M / 3M / 4M이며 학습률은 `1e-4`입니다.

공간 확대 **학습** 비교는 [PGCT_SPATIAL_EXPERIMENTS.md](PGCT_SPATIAL_EXPERIMENTS.md)의
`run_pgct_spatial_experiments.ps1`을 사용합니다. 기본은 stage03에서 시작하는 6개 비교 실험입니다.
알고리즘과 문서의 대응은 [EX5_PGCT.md](EX5_PGCT.md)에 정리돼 있습니다.

현재 우선순위인 **기존 맵의 critic transfer 유익성 검증**은
[PGCT_TRANSFER_EXPERIMENTS.md](PGCT_TRANSFER_EXPERIMENTS.md)의
`run_pgct_transfer_experiments.ps1`을 사용합니다. 기본은 같은 Stage03에서
PGCT/IPPO × training seeds1·2·3이며 공간 실험의 완료된 E1 결과를 재사용할 수 있습니다.

## Windows W&B 영상 경로

긴 실행 이름은 Windows 파일 경로 제한에 걸릴 수 있습니다. 짧은 suffix를 사용하거나
프로젝트를 `subst` 드라이브로 연결해 실행할 수 있습니다. 드라이브 별칭 사용 시
W&B가 Git 경로와 현재 드라이브를 혼동하지 않도록 같은 PowerShell에서 다음을 설정합니다.

```powershell
$env:WANDB_DISABLE_GIT = "true"
```

이 옵션은 Git 메타데이터 수집만 끄며 지표·영상 기록은 유지합니다.
`-NoTrack`은 W&B 전체를 끄는 별도 선택입니다.

## 정리 범위

단일 에이전트 delivery / return 학습 진입점, 예전 actor-only PGCT 실행기,
중복 실행기와 그 전용 설정, 루트의 임시 랜덤 실행 예제를 제거했습니다.
공통 학습기, 환경 구현, 현재 IPPO/PGCT 실행기와 설정, 맵 평가·probe 도구,
테스트, 기존 체크포인트와 실험 로그는 보존했습니다.

2026-10-13 사용자 요청으로 다음을 추가 삭제했습니다(git 추적 파일은 이력에 남음).
- 끝난 09700 일반화 평가 도구: `eval_checkpoint_generalization.py`와 테스트, ps1 2개, 문서
- 일반 혼합 커리큘럼 실행기
- 그래프 시각화·PPT 이미지 도구
- 대체된 FoV·CNN 검토 문서
- 미사용 `Success_checkpoint/`, 임시 프레임·영상 분석 폴더
- 자동 생성 `generated_configs/`

`runs/`의 과거 결과는 보존했습니다.
