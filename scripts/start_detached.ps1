# Start an experiment queue (scripts\run_experiment.ps1) OUTSIDE the calling program's process tree, through WMI
# (Win32_Process.Create). The queue then survives closing or updating the Claude/Codex app (E26) and runs without
# a hidden window as the same user (W&B login found). Output: <Experiment>\runs\<Name>\queue.log, one <job>.log per
# job. Finished jobs are skipped, so the same command resumes an interrupted queue. A job folder left by a killed
# job must be moved aside first (logs are appended, never overwritten). -WaitForPid delays the start.
#
#   powershell -ExecutionPolicy Bypass -Command "& '.\scripts\start_detached.ps1' -Experiment LBF -Name lbf_main_5m -Methods pgct_d2,ippo,oracle,all,random_gate -Seeds 1,2,3 -MaxConcurrent 5"
# Stop a queue: end its powershell.exe (PID printed below) and the python.exe job processes (Task Manager).
param(
  [Parameter(Mandatory = $true)] [ValidateSet("LBF", "VMAS")] [string] $Experiment,
  [Parameter(Mandatory = $true)] [string] $Name,
  [string] $Config = "main_4p2t.json",
  [ValidateSet("ippo", "oracle", "all", "pgct_d2", "pgct_id_only", "random_gate")]
  [string[]] $Methods = @("pgct_d2", "ippo", "oracle", "all", "random_gate"),
  [int[]] $Seeds = @(1, 2, 3),
  [int] $MaxConcurrent = 3,
  [int] $TotalTimesteps = 0,
  [string] $Python = $(if ($env:PGCT_PYTHON) { $env:PGCT_PYTHON } else { "C:\Users\95101\anaconda3\envs\marl-policy-sharing\python.exe" }),
  [switch] $NoTrack,
  [switch] $NoVideo,
  [int[]] $WaitForPid = @()
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
$Root = Join-Path $Repo "$Experiment\runs\$Name"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Log = Join-Path $Root "queue.log"
$queue = "& '$Repo\scripts\run_experiment.ps1' -Experiment $Experiment -Name $Name -Config $Config " +
         "-Methods $($Methods -join ',') -Seeds $($Seeds -join ',') -MaxConcurrent $MaxConcurrent -Python '$Python'"
if ($TotalTimesteps -gt 0) { $queue += " -TotalTimesteps $TotalTimesteps" }
if ($NoTrack) { $queue += " -NoTrack" }
if ($NoVideo) { $queue += " -NoVideo" }
$queue += " *>> '$Log'"
if ($WaitForPid.Count -gt 0) {  # start only after these processes (e.g. a running batch) have ended
  $queue = "Wait-Process -Id $($WaitForPid -join ',') -ErrorAction SilentlyContinue; " + $queue
}
$command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$queue`""
# Hidden console (SW_HIDE). A visible queue console was closed on 2026-11-06 11:17 and the close event killed
# every job attached to it (forrtl error 200, E99); a hidden window cannot be closed by accident.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $command; CurrentDirectory = $Repo; ProcessStartupInformation = $startup }
if ($result.ReturnValue -ne 0) { throw "Win32_Process.Create failed with code $($result.ReturnValue)" }
Write-Host "queue started outside this app: PID $($result.ProcessId); log $Log"
