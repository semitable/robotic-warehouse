# 재실행 준비: 새 W&B 프로젝트·10M·최근 영상 10개

2026-11-04 KST, Codex. 사용자 요청 및 프로젝트명 확인 반영. 본 학습은 시작하지 않았다.

## 실행

앱 밖 PowerShell에서:

```powershell
Set-Location 'C:\RWARE\MARL_RWARE\third_party\robotic-warehouse'
.\latent_team\run_main_amax_v2.ps1
```

프로젝트 `rware-main-amax-v2`, 넓은 지도 `b6_wide_corner_indiv_comm5_pen005`, 개인 보상,
r_c=5, action_max·mirror 없음, pgct_max/ippo/oracle_max/allk_max × seed 1–3,
처음부터 job당 10,000,000 요청 step(rollout 정렬 후 10,000,384), 최대 4개 동시 실행.
queue 창은 마지막 job이 시작될 때까지 유지한다. 매 실행 새 `am2_MMdd_HHmmss` 이름을 만든다.

## 구현

- `latent_team/video_retention.py`: 새 옵션을 켠 job에서만 trainer 함수를 프로세스 안에서 감싼다.
  새로 저장한 평가 mp4만 순서대로 관리하며 11번째가 생기면 가장 오래된 것을 지운다.
  `final.mp4`도 10개에 포함한다. 기존 실험·다른 파일에는 소급 적용하지 않는다.
- `--save-video --video-keep-last 10`: job별 설정을 manifest에 기록한다. 생략 시 기존 동작 유지.
- W&B 수치 지표는 실시간 기록한다. 영상은 종료 시 로컬에 남은 최대 10개만
  `eval/tail_01`–`eval/tail_10`으로 기록한다. 학습 중에는 로컬 `videos/`에서 확인한다.
- W&B의 실제 로컬 저장 경로는 `wandb_data/<고유 ID>/`로 분리한다. 동작/목록/실패는
  job의 `video_retention.json`에 남긴다. 영상 파일 입력을 사용해 W&B 재인코딩을 피한다.
- Python 예외/Ctrl+C에서는 현재 tail 업로드를 시도한다. Stop-Process/강제 종료/전원 종료 시에는
  로컬 영상만 보존될 수 있다. 미디어 기록 오류는 경고와 상태 파일에 남기고 로컬 영상은 보존한다.
- 원 trainer, PPO, 환경, probe, gate threshold 변경 없음. 기존 `run_queue.ps1`의 옵션 전달 사용.

## 검증(학습 성능 결과 아님)

1. 관련 테스트 23개 통과. 새 영상 테스트 8개는 로컬 10개 제한·final 포함·W&B 한 번 기록·
   실시간 수치 보존·예외 종료·미디어 오류·기존 파일 보호·기본 동작·계획의 학습 설정 동등성을 확인한다.
2. 전체 `tests/`: **271 passed**, 53.15초. 명령:
   `python -m pytest tests -q -p no:cacheprovider --basetemp output/vr_fulltests_20261104`.
3. PowerShell AST 구문 검사 통과. `run_main_amax_v2.ps1 -Name verify_am2_1104 -PlanOnly` 실행:
   `runs/verify_am2_1104_b6_wide_corner_indiv_comm5_pen005/experiment.json`의 12 job에서
   project=rware-main-amax-v2, budget=10M, keep=10, action_max, mirror=False 확인. 학습 실행 없음.
4. 실제 trainer + W&B SDK 0.28.0 오프라인 검증(`verify_offline.py`): 넓은 지도 IPPO seed 1,
   CPU, 32-step rollout × 12 update=384 step, 결정적 평가 1 episode × 2 step.
   평가 영상 12개 + final 1개 생성 후 eval_00004–00012 + final 총 10개가 남았다.
   W&B 로컬 offline media도 10개, 최대 경로 길이 177자, final.pt 생성 및 status=complete.
   결과: `offline_result.json`, 실행 경로:
   `output/video_retention_20261104/main_amax_20261104_144148_b6_wide_corner_indiv_comm5_pen005/`.
5. 첫 검증 시도는 sandbox에서 W&B 로컬 backend 시작에 진전이 없어 검증 프로세스만 종료했다.
   같은 검증을 허용된 sandbox 밖 실행으로 새 폴더에서 완료했다. 실제 본 실험 프로세스는 조작하지 않았다.

온라인 W&B 업로드·본 학습은 수행하지 않았다. 새 프로젝트는 사용자가 실행하면 생성/기록된다.
기존 실험 결과와 checkpoint는 보존했다. 새 실험의 오래된 영상 자동 삭제는 이번 사용자 요청 범위다.

참고한 SDK 원본: https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/video.py
