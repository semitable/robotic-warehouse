"""Append this diagnosis turn to shared research notes; preserve existing entries."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
final = json.loads((out / "final_text.json").read_text(encoding="utf-8"))
user = """PS C:\\RWARE\\MARL_RWARE\\third_party\\robotic-warehouse> Set-Location 'C:\\RWARE\\MARL_RWARE\\third_party\\robotic-warehouse'
>> .\\latent_team\\run_queue.ps1 -Candidates b6_wide_corner_indiv_comm5_pen005 -Modes pgct_max,ippo,oracle_max,allk_max -Seeds 1,2,3 -TotalTimesteps 10000000 -MaxConcurrent 4 -ModeMajor -Track -Name ("main_amax_" + (Get-Date -Format 'yyyyMMdd_HHmmss')) -ExtraPlanArgs @('--reward-probe-ac','--ac-gate','v2','--ac-distance','action_max','--ac-config','latent_team/acv2_nomirror.json','--team-reward','individual','--save-video')

51200에서 모든 학습이 끊기는데 이유 분석좀"""
progress = [
    "51,200 step에서 종료되는 원인을 실행 로그와 큐 설정에서 확인하겠습니다. 먼저 인수인계와 연구 기록을 읽고, 해당 실행의 오류·종료 조건을 추적하겠습니다.",
    "종료된 로그에서 공통 오류를 찾았습니다. 51,200 step의 첫 평가에서 W&B 영상을 저장하다 `FileNotFoundError`가 발생했습니다. 저장 경로 길이 문제인지 확인하고, 나머지 job도 같은 상태인지 확인하겠습니다.",
    "실패한 8개 job은 모두 첫 평가의 W&B 영상 저장에서 종료됐습니다. 저장 경로가 262~268자로 Windows 기본 경로 제한을 넘고, 이 PC는 긴 경로 지원이 꺼져 있습니다. 나머지 4개는 아직 51,200 step 전까지 로그가 갱신되고 있습니다. 학습 없이 작은 영상으로 같은 오류를 재현해 원인을 확정하겠습니다.",
    "경로 길이 문제를 재현했습니다. 같은 Python에서 짧은 경로에는 저장됐지만, 260자 경로에서는 로그와 같은 `FileNotFoundError`가 났고 긴 경로용 표기를 쓰면 저장됐습니다. 분석 결과와 재실행 시 주의점을 기록하겠습니다. 실패한 job은 첫 체크포인트 저장 시점인 256,000 step 전에 종료되어 이어 학습할 체크포인트가 없습니다.",
]

def next_id(path, pattern):
    text = path.read_text(encoding="utf-8")
    return max([int(x) for x in re.findall(pattern, text)], default=0) + 1

logpath = root / "CONVERSATION_LOG.md"
tid = next_id(logpath, r"^## 원문 기록 T(\d+)") if False else next_id(logpath, r"(?m)^## 원문 기록 T(\d+)")
did = next_id(root / "DECISIONS.md", r"(?m)^## D(\d+)\.")
eid = next_id(root / "EXPERIMENTS.md", r"(?m)^## E(\d+)\.")
summary = "넓은 지도 10M 실행의 첫 평가(51,200 step)에서 W&B 미디어 목적지 길이 262–268자로 파일 이동 실패. LongPathsEnabled=0. 짧은 경로 성공·260자 일반 경로 FileNotFoundError·확장 경로 성공을 격리 검증했다."
decision = f"\n\n## D{did}. 51,200 step 공통 종료 원인: W&B 영상 경로 길이 제한 — 2026-11-04 Codex (T{tid}, 분석 요청)\n\n- 관측·결론: {summary}\n- 제안만 함: 새 실행의 짧은 -Name 또는 짧은 별도 W&B 디렉터리. 학습 코드 변경·중지·재시작은 승인/수행하지 않았다.\n- 복구 한계: 실패 8개는 첫 checkpoint 256,000 step 이전이라 .pt가 없으며 ResumePlan은 기존 폴더를 건너뛴다.\n- 이전 검증 한계 정정: E77의 영상 검증은 짧은 임시 경로라 실제 main 경로의 제한을 검출하지 못했다.\n- 근거: output/wandb_path_diagnosis_20261104/REPORT.md, diagnosis.json, path_probe.json.\n"
experiment = f"\n\n## E{eid}. 51,200 step 종료 원인 조사·격리 경로 저장 검증(학습 없음) — 2026-11-04 Codex (T{tid})\n\n- 대상 plan: runs/main_amax_20261104_115922_b6_wide_corner_indiv_comm5_pen005. 처음부터·넓은 지도·개인 보상·r_c=5·action_max·mirror 없음·10M·W&B online·영상 저장.\n- 14:31 KST 관측: pgct_max s1–3, ippo s1–3, oracle_max s1–2 failed at 51,200. oracle_max s3 47,104, allk_max s1/s2/s3 45,056/44,032/43,008; 상태 running 및 metrics 증가 확인(CIM 조회는 접근 제한).\n- 결과: {summary} 로컬 eval_00050.mp4는 저장됨. 예외 위치는 trainer log_to_wandb → W&B Video.bind_to_run → shutil.move/open(dst).\n- 실제 수행: 기존 로그·manifest·설치 SDK 열람, 레지스트리 조회, 별도 output 폴더의 표준 라이브러리 파일 저장 검증. W&B Video 추가 재현은 의존성 로딩 중 결과 없이 진단 프로세스만 중단. 장기 학습·새 평가 실행 없음.\n- 코드 관측: --save-video이면 첫 평가부터 eval_video가 생겨 wandb_video_interval=5와 무관하게 업로드 경로가 호출된다(trainer 5192, 5228–5233).\n- 한계: 현재 프로세스에 수정 주입 안 함. 새 경로에서의 실제 W&B 영상 업로드·학습 재실행 검증은 안 함.\n- 근거: output/wandb_path_diagnosis_20261104/REPORT.md, diagnose.py, diagnosis.json, path_probe.py, path_probe.json; 각 job_status.json·metrics.jsonl·err.log.\n"
for name, addition in [("DECISIONS.md", decision), ("EXPERIMENTS.md", experiment)]:
    p = root / name
    p.read_text(encoding="utf-8")  # reread immediately before append
    with p.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(addition)
statepath = root / "RESEARCH_STATE.md"
state = statepath.read_text(encoding="utf-8")
heading, rest = state.split("\n", 1)
new = f"\n\n## 최신 확인 — 2026-11-04 14:31 KST, Codex (T{tid}, 51,200 step 종료 원인)\n\n- **장애(D{did}, E{eid})**: {summary}\n- **실제 상태**: pgct_max s1–3·ippo s1–3·oracle_max s1–2 총 8개 failed at 51,200. oracle_max s3와 allk_max s1–3은 43,008–47,104 step에서 로그 증가 확인. 이전 T185의 전부 정상 실행 중 기록은 과거 스냅샷이다.\n- **복구 제안**: 짧고 새 -Name(예 amax2) 또는 짧은 별도 W&B 경로. 실패 job의 .pt 없음(첫 checkpoint 256,000). ResumePlan은 실패 job 복구가 아니다. 학습 코드·프로세스 변경 없음.\n- 근거: output/wandb_path_diagnosis_20261104/REPORT.md.\n"
statepath.write_text(heading + new + rest, encoding="utf-8", newline="\n")
log = f"\n\n## 원문 기록 T{tid} — 2026-11-04 KST, Codex (정확한 발화 시각은 기록하지 않음)\n\n### 사용자 — 원문\n\n{user}\n\n"
for item in progress:
    log += f"### Assistant — 진행 안내\n\n{item}\n\n"
log += f"### 수행 기록 — 대화 원문과 별도\n\n- 세션 제공 AGENTS.md 전체 및 HANDOFF.md, docs/SESSION_HANDOFF.md, RESEARCH_STATE.md 전체(초기 출력 일부 생략), DECISIONS.md·EXPERIMENTS.md·CONVERSATION_LOG.md의 최근 부분을 읽었다. 실제 분석 범위는 REPORT.md 참조.\n- 별도 output에 진단 스크립트·근거 생성, D{did}/E{eid}/연구 상태 및 이 기록 갱신. 학습 코드 변경·학습 시작·중단·재시작·커밋 없음.\n- 진행 안내에서는 작은 영상 재현을 예고했으나 실제 완료 검증은 일반 파일 경로 저장이다. 추가 영상 API 검증은 결과 없이 진단 프로세스만 중단했다.\n\n### Assistant — 최종 답변\n\n{final}\n"
logpath.read_text(encoding="utf-8")
with logpath.open("a", encoding="utf-8", newline="") as stream:
    stream.write(log.replace("\n", "\r\n"))
print(f"Recorded T{tid}, D{did}, E{eid}")
