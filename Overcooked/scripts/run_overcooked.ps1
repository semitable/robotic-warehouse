# Overcooked PGCT experiments: plan every (method, seed) job of one config, then run them as separate
# processes with at most -MaxConcurrent at a time. Logs: <plan>\<job>.log. Finished jobs are skipped, so the
# same command resumes an interrupted queue (unfinished jobs restart from scratch; there is no mid-run resume).
#
# Examples (repository root):
#   powershell -ExecutionPolicy Bypass -File Overcooked\scripts\run_overcooked.ps1 -Name oc_smoke -Smoke
#   powershell -ExecutionPolicy Bypass -Command "& '.\Overcooked\scripts\run_overcooked.ps1' -Name oc_main -Seeds 1,2,3 -MaxConcurrent 3"
# Long runs: start them yourself in a separate PowerShell window (not from an app that may restart).
param(
  [string] $Config = "main_4p2t.json",
  [ValidateSet("ippo", "oracle", "all", "pgct_d2", "pgct_id_only", "random_gate")]
  [string[]] $Methods = @("ippo", "oracle", "all", "pgct_d2", "pgct_id_only", "random_gate"),
  [int[]] $Seeds = @(1, 2, 3),
  [Parameter(Mandatory = $true)] [string] $Name,
  [int] $TotalTimesteps = 0,
  [int] $MaxConcurrent = 2,
  [string] $Python = $(if ($env:PGCT_PYTHON) { $env:PGCT_PYTHON } else { "python" }),
  [string[]] $Set = @(),
  [switch] $Smoke,
  [switch] $PlanOnly
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Repo
$env:PYTHONPATH = "$Repo\Overcooked;$Repo\PGCT"
$Root = Join-Path $Repo "Overcooked\runs\$Name"
$Common = @("-m", "pgct_overcooked.train", "--config", $Config, "--name", $Name)
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
Write-Host "queue finished; aggregate with Overcooked\scripts\aggregate.ps1 -Name $Name"
