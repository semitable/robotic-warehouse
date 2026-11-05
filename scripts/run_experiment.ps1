# PGCT experiment queue for the pgct_runner experiments (LBF, VMAS): plan every (method, seed) job of one
# config, then run them as separate processes with at most -MaxConcurrent at a time. Logs: <plan>\<job>.log.
# Finished jobs are skipped, so the same command resumes an interrupted queue (unfinished jobs restart from
# scratch; there is no mid-run resume). Overcooked keeps its own Overcooked\scripts\run_overcooked.ps1.
#
# Examples (repository root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Experiment LBF -Name lbf_smoke -Smoke -NoTrack
#   powershell -ExecutionPolicy Bypass -Command "& '.\scripts\run_experiment.ps1' -Experiment LBF -Name lbf_main -Seeds 1,2,3 -MaxConcurrent 3"
#   powershell -ExecutionPolicy Bypass -Command "& '.\scripts\run_experiment.ps1' -Experiment VMAS -Name vmas_main -Seeds 1,2,3"
# VMAS needs `pip install vmas==1.5.2` (GPLv3; pulls gym 0.26 and pyglet<=1.5.27).
# Long runs: start them yourself in a separate PowerShell window (not from an app that may restart).
param(
  [Parameter(Mandatory = $true)] [ValidateSet("LBF", "VMAS")] [string] $Experiment,
  [string] $Config = "main_4p2t.json",
  [ValidateSet("ippo", "oracle", "all", "pgct_d2", "pgct_id_only", "random_gate")]
  [string[]] $Methods = @("ippo", "oracle", "all", "pgct_d2", "pgct_id_only", "random_gate"),
  [int[]] $Seeds = @(1, 2, 3),
  [Parameter(Mandatory = $true)] [string] $Name,
  [int] $TotalTimesteps = 0,
  [int] $MaxConcurrent = 2,
  [string] $Python = $(if ($env:PGCT_PYTHON) { $env:PGCT_PYTHON } else { "python" }),
  [string[]] $Set = @(),
  [switch] $NoTrack,
  [string] $WandbProject = "",
  [string] $WandbEntity = "",
  [ValidateSet("online", "offline")] [string] $WandbMode = "online",
  [string[]] $WandbTags = @(),
  # Play videos: one greedy episode at every N-th evaluation plus the final one (eval/video on W&B).
  [int] $VideoEvery = 1,
  [int] $VideoKeep = 10,
  [switch] $NoVideo,
  [switch] $Smoke,
  [switch] $PlanOnly
)
$ErrorActionPreference = "Stop"
$Package = @{ LBF = "pgct_lbf"; VMAS = "pgct_vmas" }[$Experiment]
if (-not $WandbProject) { $WandbProject = "pgct-" + $Experiment.ToLower() }
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$env:PYTHONPATH = "$Repo\$Experiment;$Repo\PGCT"
$env:PYTHONIOENCODING = "utf-8"
$Root = Join-Path $Repo "$Experiment\runs\$Name"
$Common = @("-m", "$Package.train", "--config", $Config, "--name", $Name)
if (-not $NoTrack) {
  $Common += @("--track", "--wandb-project", $WandbProject, "--wandb-mode", $WandbMode)
  if ($WandbEntity) { $Common += @("--wandb-entity", $WandbEntity) }
  if ($WandbTags.Count -gt 0) { $Common += @("--wandb-tags") + $WandbTags }
}
if (-not $NoVideo) { $Common += @("--video-every", "$VideoEvery", "--video-keep", "$VideoKeep") }
if ($Smoke) { $Common += "--smoke" }
if ($TotalTimesteps -gt 0) { $Common += @("--total-timesteps", "$TotalTimesteps") }
if ($Set.Count -gt 0) { $Common += @("--set") + $Set }

# 1) Plan all jobs once (also validates the configuration).
& $Python @Common --methods @Methods --seeds @($Seeds | ForEach-Object { "$_" }) --plan-only
if ($LASTEXITCODE -ne 0) { throw "planning failed" }
if ($PlanOnly) { return }

# 2) One process per job.
$Queue = New-Object System.Collections.Queue
foreach ($s in $Seeds) { foreach ($m in $Methods) { $Queue.Enqueue(@($m, $s)) } }
$Running = @()
while ($Queue.Count -gt 0 -or $Running.Count -gt 0) {
  $Running = @($Running | Where-Object { -not $_.HasExited })
  while ($Queue.Count -gt 0 -and $Running.Count -lt $MaxConcurrent) {
    $job = $Queue.Dequeue()
    $jobName = "$($job[0])_s$($job[1])"
    if (Test-Path (Join-Path $Root "$jobName\final.pt")) { Write-Host "skip $jobName (finished)"; continue }
    $log = Join-Path $Root "$jobName.log"
    $jobArgs = $Common + @("--methods", $job[0], "--seeds", "$($job[1])")
    $p = Start-Process -FilePath $Python -ArgumentList $jobArgs -NoNewWindow -PassThru `
      -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    Write-Host "start $jobName (PID $($p.Id)) -> $log"
    $Running += $p
  }
  Start-Sleep -Seconds 5
}
$Unfinished = @(foreach ($s in $Seeds) { foreach ($m in $Methods) {
  if (-not (Test-Path (Join-Path $Root "$($m)_s$s\final.pt"))) { "$($m)_s$s" } } })
if ($Unfinished.Count) {
  Write-Host "UNFINISHED: $($Unfinished -join ', ') (see <job>.log / <job>.log.err; rerun the same command)"
  exit 1
}
Write-Host "queue finished; aggregate with scripts\aggregate_experiment.ps1 -Experiment $Experiment -Name $Name"
