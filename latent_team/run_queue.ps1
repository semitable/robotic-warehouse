# Queue launcher: plan IPPO/transfer jobs for one or more candidates and run them a few at a time (D74).
#
# Plans every candidate with train_from_scratch.py --plan-only (one plan folder per candidate, like
# run_candidate_pilot.ps1), then starts jobs so that at most -MaxConcurrent training processes run on this
# machine.  Training processes started elsewhere (for example a pilot that is still running) count too, so
# the queue fills the machine as they finish.  Jobs whose output folder already exists are never started
# again (a crashed job is reported and left alone, so no log or checkpoint is overwritten).
#
# Run this from a PowerShell window you opened yourself (not from the Claude app).  Closing that window
# stops the queue but not the jobs already started.  Nothing here changes the trainer or the environment.
param(
  [string[]] $Candidates = @(),
  [string[]] $Modes = @("ippo"),
  [int[]] $Seeds = @(1, 2),
  [int] $TotalTimesteps = 1000000,
  [int] $MaxConcurrent = 4,
  [int] $PollSeconds = 60,
  [int] $ThreadsPerJob = 4,
  [string] $Name = "",
  [string] $RandomReference = "",
  [switch] $Track,
  [string] $WandbProject = "rware-from-scratch",
  [switch] $SmokeTest,
  [switch] $PlanOnly,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe",
  # T178: extra train_from_scratch.py planning arguments, e.g. @("--reward-probe-ac", "--ac-gate", "v2",
  # "--ac-distance", "action_max", "--ac-config", "latent_team/acv2_nomirror.json").  Empty = unchanged.
  [string[]] $ExtraPlanArgs = @(),
  # T178: start jobs mode by mode in the order of -Modes (all seeds of the first mode first) instead of the
  # plan's seed-by-seed order, so the most important modes finish first.
  [switch] $ModeMajor,
  # T185: resume an existing plan folder (runs/<Name>_<candidate>) without planning again, e.g. after the window
  # that ran the queue was closed. Give the original -Name and -Modes; jobs whose folder exists are skipped and
  # the PIDs already in runs/<Name>_pids.json are kept.
  [string] $ResumePlan = ""
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Name) { $Name = "queue_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
$env:OMP_NUM_THREADS = "$ThreadsPerJob"
$env:MKL_NUM_THREADS = "$ThreadsPerJob"
$env:WANDB_DISABLE_GIT = "true"
$QueueLog = "runs/${Name}_queue.log"
function Log($Text) { $Line = "[{0}] {1}" -f (Get-Date -Format s), $Text; Write-Host $Line; Add-Content -Encoding utf8 $QueueLog $Line }

function Running-Training {
  @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "train_from_scratch\.py" -and $_.CommandLine -match "--run-job" })
}

# 1. plan every candidate (or reuse an existing plan)
$Queue = @()
if (-not $ResumePlan -and -not $Candidates) { throw "Give -Candidates (new plan) or -ResumePlan (existing plan)" }
if ($ResumePlan) {
  $Plan = (Resolve-Path $ResumePlan).Path
  $Manifest = Get-Content (Join-Path $Plan "experiment.json") -Raw | ConvertFrom-Json
  foreach ($Job in $Manifest.jobs) {
    $Queue += [pscustomobject]@{ Job = $Job.name; Plan = $Plan; Mode = $Job.mode; Seed = [int]$Job.seed }
  }
  $Candidates = @()
}
foreach ($Candidate in $Candidates) {
  $Out = "runs/${Name}_$Candidate"
  $PlanArgs = @("latent_team/train_from_scratch.py", "--maps", "e1", "--env-variant", $Candidate, "--modes") + $Modes +
    @("--seeds") + @($Seeds | ForEach-Object { "$_" }) + @("--timesteps", "$TotalTimesteps", "--output-dir", $Out,
    "--stage-label", "${Name}_$Candidate", "--wandb-project", $WandbProject, "--plan-only")
  if ($RandomReference) { $PlanArgs += @("--random-reference", $RandomReference) }
  if ($Track) { $PlanArgs += "--track" }
  if ($SmokeTest) { $PlanArgs += "--smoke-test" }
  $PlanArgs += $ExtraPlanArgs
  & $Python @PlanArgs
  if ($LASTEXITCODE -ne 0) { throw "Planning $Candidate failed with exit code $LASTEXITCODE" }
  $Plan = (Resolve-Path $Out).Path
  $Manifest = Get-Content (Join-Path $Plan "experiment.json") -Raw | ConvertFrom-Json
  foreach ($Job in $Manifest.jobs) {
    $Queue += [pscustomobject]@{ Job = $Job.name; Plan = $Plan; Mode = $Job.mode; Seed = [int]$Job.seed }
  }
}
if ($ModeMajor) {
  $Queue = @($Queue | Sort-Object @{ Expression = { [array]::IndexOf($Modes, $_.Mode) } }, Seed, Plan)
}
Log ("planned {0} jobs: {1}" -f $Queue.Count, (($Queue | ForEach-Object { $_.Job }) -join ", "))
if ($PlanOnly) { return }

# 2. start them a few at a time
$Started = @()
$PidFile = "runs/${Name}_pids.json"
if ($ResumePlan -and (Test-Path $PidFile)) { $Started = @(Get-Content $PidFile -Raw | ConvertFrom-Json) }
foreach ($Item in $Queue) {
  $RunDir = Join-Path $Item.Plan $Item.Job
  if (Test-Path $RunDir) { Log "skip $($Item.Job): output folder already exists (not overwritten)"; continue }
  while ((Running-Training).Count -ge $MaxConcurrent) { Start-Sleep -Seconds $PollSeconds }
  $Log = Join-Path $Item.Plan ("{0}.log" -f $Item.Job)
  $Err = Join-Path $Item.Plan ("{0}.err.log" -f $Item.Job)
  $Process = Start-Process -FilePath $Python -PassThru -WindowStyle Hidden `
    -ArgumentList @("latent_team/train_from_scratch.py", "--run-job", $Item.Job, "--plan", "`"$($Item.Plan)`"") `
    -RedirectStandardOutput $Log -RedirectStandardError $Err
  $Started += [pscustomobject]@{ Job = $Item.Job; PID = $Process.Id; Plan = $Item.Plan }
  $Started | ConvertTo-Json | Set-Content -Encoding utf8 "runs/${Name}_pids.json"
  Log "started $($Item.Job) (PID $($Process.Id)); running now: $((Running-Training).Count)"
  Start-Sleep -Seconds 20   # let the process appear before counting again
}
Log "all jobs started; PIDs in runs/${Name}_pids.json. The queue window can be closed."
