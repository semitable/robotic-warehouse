# IPPO learning-speed pilot for the E1-F replacement candidates (env_candidates.py, D62 / E40).
#
# One plan folder per candidate, IPPO only, seeds 1 and 2, team reward (set by the candidate), 2M steps by default.
# The only question is "does IPPO start delivering, and how fast?" -- pick the environment from that,
# never from probe results.  Every job runs as its own background python process with stdout/stderr logs.
#
# Run this from a PowerShell window you opened yourself (not from the Claude app), so an app update or
# restart cannot stop the jobs.  Nothing here changes the trainer, the environment or existing runs.
param(
  [ValidateSet("b_e1f6_comm3", "a_e1f4_team", "c_e1f6_wide_comm3", "b_e1f6_comm3_nopen", "a_e1f4_team_nopen", "b_e1f6_comm3_pen005",
               "b6_corner_indiv_pen005", "b6_adjacent_team_pen005", "b6_adjacent_indiv_pen005",
               "c1_e1f_team", "c2_e1f8_team", "c3_wide_s3_team")]
  [string[]] $Candidates = @("b_e1f6_comm3", "a_e1f4_team"),   # D63 priority B -> A; add c_e1f6_wide_comm3 if B learns
  [int[]] $Seeds = @(1, 2),                                      # seed 2 swaps the goal sides
  [int] $TotalTimesteps = 2000000,
  [int] $ThreadsPerJob = 4,
  [string] $Name = "",
  [switch] $Track,
  [string] $WandbProject = "rware-from-scratch",
  [switch] $PlanOnly,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Continue"   # python warnings go to stderr; exit codes are checked instead
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Name) { $Name = "pilot_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
$env:OMP_NUM_THREADS = "$ThreadsPerJob"
$env:MKL_NUM_THREADS = "$ThreadsPerJob"
$env:WANDB_DISABLE_GIT = "true"

$Started = @()
foreach ($Candidate in $Candidates) {
  $Out = "runs/${Name}_$Candidate"
  $PlanArgs = @("latent_team/train_from_scratch.py", "--maps", "e1", "--env-variant", $Candidate, "--modes", "ippo",
    "--seeds") + @($Seeds | ForEach-Object { "$_" }) + @("--timesteps", "$TotalTimesteps", "--output-dir", $Out,
    "--stage-label", "${Name}_$Candidate", "--wandb-project", $WandbProject, "--plan-only")
  if ($Track) { $PlanArgs += "--track" }
  if ($SmokeTest) { $PlanArgs += "--smoke-test" }
  & $Python @PlanArgs
  if ($LASTEXITCODE -ne 0) { throw "Planning $Candidate failed with exit code $LASTEXITCODE" }
  if ($PlanOnly) { continue }
  $Plan = Resolve-Path $Out
  $Manifest = Get-Content (Join-Path $Plan "experiment.json") -Raw | ConvertFrom-Json
  foreach ($Job in $Manifest.jobs) {
    $Log = Join-Path $Plan ("{0}.log" -f $Job.name)
    $Err = Join-Path $Plan ("{0}.err.log" -f $Job.name)
    $Process = Start-Process -FilePath $Python -PassThru -WindowStyle Hidden `
      -ArgumentList @("latent_team/train_from_scratch.py", "--run-job", $Job.name, "--plan", "`"$Plan`"") `
      -RedirectStandardOutput $Log -RedirectStandardError $Err
    $Started += [pscustomobject]@{ Job = $Job.name; PID = $Process.Id; Plan = "$Plan" }
    Write-Host ("[{0}] started {1} (PID {2}); log {3}" -f (Get-Date -Format s), $Job.name, $Process.Id, $Log)
  }
}
if ($Started.Count -gt 0) {
  $Started | Format-Table -AutoSize
  $Started | ConvertTo-Json | Set-Content -Encoding utf8 "runs/${Name}_pids.json"
  Write-Host "PIDs saved to runs/${Name}_pids.json"
}
