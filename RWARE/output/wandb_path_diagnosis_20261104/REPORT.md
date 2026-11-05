# 51,200 step 종료 원인 분석

2026-11-04 14:31 KST 확인, Codex. 학습 실행·중지·재시작 및 학습 코드 수정 없음.

## 관측

대상: `runs/main_amax_20261104_115922_b6_wide_corner_indiv_comm5_pen005/experiment.json`.
넓은 지도, 개인 보상, r_c=5, 처음부터 10M, action_max, mirror 없음, W&B online 및 영상 저장.

- pgct_max s1–3, ippo s1–3, oracle_max s1–2: 8개 모두 51,200 step에서 failed.
- oracle_max s3: 47,104, allk_max s1/s2/s3: 45,056/44,032/43,008 step. `job_status.json`은 running이고 분석 중 metrics 증가를 확인했다. CIM 프로세스 조회는 접근 제한으로 불가했다.
- 8개 실패의 traceback: trainer 5233 `log_to_wandb` → W&B `run.log` → `Video.bind_to_run` → `shutil.move` → `open(dst, 'wb')` → FileNotFoundError. 로컬 `videos/eval_00050.mp4`는 이미 저장됐다.
- W&B 목적지 길이: IPPO 262자, PGCT 266자, Oracle 268자. 부모 디렉터리는 실제 존재한다.
- Windows 레지스트리 `HKLM/SYSTEM/CurrentControlSet/Control/FileSystem/LongPathsEnabled=0`(조회만 수행).
- `path_probe.py`의 격리된 파일 저장 검증: 짧은 경로 성공, 부모가 존재하는 260자 경로에 일반 쓰기는 FileNotFoundError, 같은 경로에 확장 경로 표기 쓰기는 성공. 학습 데이터에는 쓰지 않았다.

## 결론

W&B 영상 목적지의 Windows 기본 경로 길이 제한 초과가 원인이다. PGCT gate나 수렴 중단 조건이 원인이라는 근거는 없다.
설정은 rollout_steps=1024, eval_interval=50이므로 첫 평가가 1024×50=51,200이다. 이때 로컬 영상 저장 후 W&B 미디어 파일 이동이 실패하며 예외가 학습까지 전파된다.

`wandb_video_interval=5`여도 `save_eval_video=True`이면 평가가 영상을 생성한다(trainer 5192). trainer 5228–5233은 영상이 있으면 W&B에 넣으므로 첫 평가부터 업로드한다. 따라서 5번째 평가까지 기다리지 않는다.

## 복구 방향(제안, 실행하지 않음)

1. 새 실험은 짧고 사용하지 않은 `-Name`(예: `amax2`)으로 생성한다. 이 예에서는 기존 이름보다 20자가 줄어 가장 긴 Oracle 목적지가 268→248자, 10M step에서는 251자로 계산된다. 기존 결과는 보존한다.
2. 장기적 수정 후보는 `latent_team/` 실행기에서 W&B 로컬 디렉터리를 짧은 전용 경로로 분리하고, 미디어 저장 예외로 학습 전체가 종료되지 않게 처리하는 것이다. 현재 `init_wandb`는 `dir=str(save_dir)`를 명시하므로 `WANDB_DIR`만 설정하는 처방은 확실한 해결이 아니다.
3. `--save-video` 제거는 영상 없는 우회 방법이다. 사용자는 영상 확인을 원했으므로 경로 단축이 우선이다.
4. `-ResumePlan`은 기존 job 폴더를 건너뛰며 실패 job을 복구하지 않는다. save_interval=250이므로 첫 checkpoint는 256,000 step이다. 현재 실패 8개에는 `.pt`가 없고 51,200에서 이어 학습할 수 없다.
5. E77 영상 검증은 짧은 임시 경로에서 수행되어 본 실험의 긴 경로 오류를 검출하지 못했다.

근거: `diagnosis.json`, `path_probe.json`, 대상 plan의 각 `job_status.json`·`*.err.log`·`metrics.jsonl`, `examples/train_recurrent_ippo_consensus.py` 1135·5192·5228–5233, 설치된 wandb 0.28.0의 `sdk/data_types/base_types/media.py` 142–147.

공식 경로 제한 설명: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation

## 검증 범위

실제 장기 학습·새 평가 없음. 격리된 파일 저장 검증만 완료했다. W&B Video API의 추가 재현은 의존성 로딩 중 결과를 얻지 못해 해당 진단 프로세스만 중단했고, 최종 검증은 표준 라이브러리 파일 저장으로 수행했다. 현재 살아 있는 학습에 수정 사항을 주입하거나 중단하지 않았다.
