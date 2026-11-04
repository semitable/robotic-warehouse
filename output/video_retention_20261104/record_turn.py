"""Append the authorized video-retention work and verbatim user-facing conversation."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
final = json.loads((out / "final_text.json").read_text(encoding="utf-8"))

def next_id(filename, prefix):
    text = (root / filename).read_text(encoding="utf-8")
    pattern = r"(?m)^## " + re.escape(prefix) + r"(\d+)"
    return max(map(int, re.findall(pattern, text)), default=0) + 1

tid = next_id("CONVERSATION_LOG.md", "원문 기록 T")
did = next_id("DECISIONS.md", "D")
eid = next_id("EXPERIMENTS.md", "E")
for filename, body in [
    ("DECISIONS.md", f"\n\n## D{did}. 새 프로젝트 10M 재실행·job별 영상 최근 10개 보존 — 2026-11-04 Codex (T{tid}, 사용자 요청)\n\n- 사용자 지시: 기존 rware-from-scratch가 아닌 새 프로젝트, 동일 10M, 각 학습 마지막 영상 10개만 보존. 후속 답변으로 프로젝트명 rware-main-amax-v2 확정.\n- 구현: 새 run_main_amax_v2.ps1, --video-keep-last 10, 로컬 rolling 10개(final 포함), W&B는 종료 시 tail 10개만 기록·수치는 실시간. W&B 저장을 짧은 wandb_data/<고유 ID>로 분리해 D116 장애 해결. 기존 실험 파일에는 소급 적용하지 않는다.\n- 영상 업로드 시점(종료 시)은 구현 판단이며 진행 안내와 최종 답변에 명시했다. 강제 종료는 업로드를 보장하지 않음.\n- 기존 4 mode × seed 1–3, 넓은 지도·action_max·mirror 없음·10M 유지. 본 학습은 사용자가 실행한다.\n- trainer·알고리즘·환경·probe·gate 임계값 변경 없음. 근거: output/video_retention_20261104/REPORT.md, E{eid}.\n"),
    ("EXPERIMENTS.md", f"\n\n## E{eid}. 새 프로젝트·영상 10개 보존 실행기 검증(본 학습 아님) — 2026-11-04 Codex (T{tid})\n\n- 실제 수행: 관련 23개 및 전체 tests/ 271개 통과(53.15초), PowerShell AST 구문 검사, 새 launcher -PlanOnly로 12 job 계획 생성.\n- plan: runs/verify_am2_1104_b6_wide_corner_indiv_comm5_pen005/experiment.json. 프로젝트 rware-main-amax-v2, 요청 10M(실제 rollout 정렬 10,000,384), keep_last 10, 넓은 지도·개인·r_c5·action_max·mirror False, pgct_max/ippo/oracle_max/allk_max × s1–3 확인. 계획만이며 본 학습 미실행.\n- 실제 영상 smoke: output/video_retention_20261104/main_amax_20261104_144148_b6_wide_corner_indiv_comm5_pen005/b6_wide_corner_indiv_comm5_pen005_ippo_s1/final.pt. 처음부터 IPPO s1·CPU·넓은 지도, 32×12=384 step, 결정적 평가 1ep×2step, W&B offline. 13개 영상 생성 후 로컬10·W&B offline media10개, final 포함, 최장 W&B 경로177자, complete. 성능 결과/독립 seed 실험으로 세지 않는다.\n- 처음 sandbox 시도는 W&B backend 시작에 진전이 없어 해당 검증 프로세스만 중단. 새 폴더에서 sandbox 밖 offline 검증 완료. 실제 본 실험 프로세스 조작 없음.\n- 새 파일: latent_team/video_retention.py, run_main_amax_v2.ps1, tests/test_video_retention.py. 수정: train_from_scratch.py 옵션·설치 연결, README, .gitignore.\n- 한계: 온라인 업로드·본 학습 미실행. 종료 시 영상 업로드하며 강제 종료 시 로컬 파일만 남을 수 있음. 기존 결과 삭제 없음; 새 run의 오래된 영상만 사용자 지시대로 순환 삭제.\n- 근거: output/video_retention_20261104/REPORT.md, offline_result.json, verify_offline.py; 새 plan manifest.\n")]:
    path = root / filename
    path.read_text(encoding="utf-8")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(body)

path = root / "RESEARCH_STATE.md"
old = path.read_text(encoding="utf-8")
heading, rest = old.split("\n", 1)
addition = f"\n\n## 최신 확인 — 2026-11-04 14:42 KST, Codex (T{tid}, 새 프로젝트 재실행 준비 완료)\n\n- **사용자 요청 반영(D{did}, E{eid})**: rware-main-amax-v2, 기존 넓은 지도·4 mode×seed1–3·10M. job별 최근 영상10개(final 포함), 로컬 순환 보존·W&B 종료 시 tail 업로드·수치 실시간. 짧은 W&B 경로로 D116 장애 해결.\n- **검증**: 전체271 테스트, 12 job plan-only, 실제 IPPO s1 384-step offline 영상 검증 완료(로컬10·W&B media10·최장177자). 본 학습 미시작. 실행: latent_team/run_main_amax_v2.ps1(사용자 직접).\n- **기존 실행 상태**: 14:42 상태 파일 기준 이전 main_amax 12개 중11 failed·1 running. 실제 본 학습은 중단/재시작하지 않았다.\n- 근거: output/video_retention_20261104/REPORT.md. 강제 종료 시 W&B 영상 업로드 불가할 수 있음.\n"
path.write_text(heading + addition + rest, encoding="utf-8", newline="\n")

messages = [
    ("사용자 — 원문", "실험을 다시 돌릴꺼거든?\n근데 rware-from-scratch wandb 프로젝트 말고 새 프로젝트에다 할꺼고\n똑같이 10M으로 하고,\n모든 영상을 저장하면 너무 많으니까 각 학습마다 마지막 10개만 냅둬줘"),
    ("Assistant — 진행 안내", "새 W&B 프로젝트를 지정하고 10M 설정을 유지하면서, 각 학습의 평가 영상은 최근 10개만 남기도록 실행기를 수정하겠습니다. 기존 결과는 보존하고, 재실행 명령까지 준비하겠습니다."),
    ("Assistant — 질문 도구 표시 원문", "새 W&B 프로젝트 이름은 무엇으로 할까요? 답이 없으면 `rware-main-amax-v2`로 준비하겠습니다.\n\n선택지: rware-main-amax-v2"),
    ("사용자 — 질문 답변 원문", "rware-main-amax-v2"),
    ("Assistant — 진행 안내", "프로젝트는 `rware-main-amax-v2`로 설정하겠습니다. 로컬에는 학습 중에도 최근 영상 10개만 유지하고, W&B에는 종료할 때 남은 10개만 업로드하도록 하겠습니다. W&B 저장 경로도 짧게 분리해 앞서 발생한 경로 길이 오류를 함께 해결하겠습니다."),
    ("Assistant — 진행 안내", "최근 10개 영상만 남기는 처리를 추가했습니다. 수치 지표는 계속 W&B에 실시간으로 기록하고, 영상은 종료 시 한 번만 올리도록 분리했습니다. 이제 정상 종료·예외 종료에서 10개 제한이 지켜지는지, 긴 실험 폴더 이름에서도 W&B 저장이 되는지 검증하겠습니다."),
    ("Assistant — 진행 안내", "새 프로젝트·10M·12개 job·영상 10개 보존 설정이 계획에 정확히 들어가는 것을 확인했고, 관련 테스트 23개가 통과했습니다. 실제 영상 인코딩과 W&B 저장은 별도의 짧은 오프라인 실행으로 확인 중입니다. 본 실험은 시작하지 않았습니다."),
    ("Assistant — 진행 안내", "전체 테스트 271개가 통과했습니다. 실제 IPPO 짧은 검증에서도 영상 13개를 생성한 뒤 마지막 10개만 남았고, W&B 오프라인 기록에도 동일한 10개만 저장됐습니다. W&B 영상 경로는 최장 177자로 확인됐습니다. 재실행은 준비된 스크립트 한 줄로 시작할 수 있습니다."),
    ("수행 기록 — 대화 원문과 별도", f"구현·검증 범위는 D{did}/E{eid}와 output/video_retention_20261104/REPORT.md. 본 학습 미실행, 기존 결과 보존, 커밋 없음. 온라인 W&B 업로드를 검증했다고 주장하지 않음. 실제 읽은 SDK 원문·로컬 코드 및 생성 자료 경로는 보고서 참조."),
    ("Assistant — 최종 답변", final),
]
log = f"\n\n## 원문 기록 T{tid} — 2026-11-04 KST, Codex (개별 발화의 정확한 시각은 기록하지 않음)\n\n"
log += "\n\n".join(f"### {role}\n\n{body}" for role, body in messages) + "\n"
path = root / "CONVERSATION_LOG.md"
path.read_text(encoding="utf-8")
with path.open("a", encoding="utf-8", newline="") as stream:
    stream.write(log.replace("\n", "\r\n"))
print(f"Recorded T{tid}, D{did}, E{eid}")
