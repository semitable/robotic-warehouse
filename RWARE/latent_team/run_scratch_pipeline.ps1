# Stage 1 (pilot) then, only if every stage-1 job completed, stage 2 (main comparison).
# Stage 1 is NOT an initialization for stage 2: every run starts from random weights. Because the trainer has
# no learning-rate schedule, a stage-1 mode is a prefix of the same stage-2 mode; -SkipStage1 avoids the overlap.
# Use -Detach to run the whole pipeline in a hidden background PowerShell that survives closing this window.
param(
  [ValidateSet("ippo", "pgct", "oracle", "all", "wrong", "mappo", "seps", "hetdps")] [string[]] $Stage1Modes = @("ippo", "oracle"),
  [int] $Stage1Timesteps = 4000000,
  [int[]] $Stage1Seeds = @(1),
  [ValidateSet("ippo", "pgct", "oracle", "all", "wrong", "mappo", "seps", "hetdps")] [string[]] $Stage2Modes = @("ippo", "pgct", "oracle", "all"),
  [int] $Stage2Timesteps = 15000000,
  [int[]] $Stage2Seeds = @(1),
  [string] $Name = "",
  [int] $ThreadsPerJob = 4,
  [switch] $SkipStage1,
  [switch] $Track,
  [string] $WandbProject = "rware-from-scratch",
  [ValidateSet("online", "offline")] [string] $WandbMode = "online",
  [ValidateSet("e1", "e1f")] [string] $EnvVariant = "e1",
  [ValidateSet("canonical", "v2")] [string] $ProbeBank = "canonical",
  [ValidateSet("individual", "team")] [string] $TeamReward = "individual",
  [switch] $Detach,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Name) { $Name = "scratch_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
$Stage1Dir = "runs/${Name}_stage1"
$Stage2Dir = "runs/${Name}_stage2"
$PipelineLog = "runs/${Name}_pipeline.log"

if ($Detach) {
  $Self = $MyInvocation.MyCommand.Path
  $Command = "& '$Self' -Name '$Name' -ThreadsPerJob $ThreadsPerJob -Python '$Python'" +
    " -Stage1Modes $($Stage1Modes -join ',') -Stage1Timesteps $Stage1Timesteps -Stage1Seeds $($Stage1Seeds -join ',')" +
    " -Stage2Modes $($Stage2Modes -join ',') -Stage2Timesteps $Stage2Timesteps -Stage2Seeds $($Stage2Seeds -join ',')" +
    " -WandbProject '$WandbProject' -WandbMode $WandbMode -EnvVariant $EnvVariant -ProbeBank $ProbeBank -TeamReward $TeamReward"
  if ($SmokeTest) { $Command += " -SmokeTest" }
  if ($SkipStage1) { $Command += " -SkipStage1" }
  if ($Track) { $Command += " -Track" }
  # UTF-8 log so that the pipeline status can be read by any tool.
  $Command += " *>&1 | Out-File -FilePath '$PipelineLog' -Encoding utf8"
  $Process = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $Command)
  Write-Host "Pipeline started in background (PID $($Process.Id))."
  Write-Host "Pipeline log: $PipelineLog"
  Write-Host "Stage 1: $Stage1Dir   Stage 2 (starts after stage 1): $Stage2Dir"
  return
}

$Common = @{ Parallel = $true; Wait = $true; ThreadsPerJob = $ThreadsPerJob; Python = $Python; SmokeTest = $SmokeTest;
             Track = $Track; WandbProject = $WandbProject; WandbMode = $WandbMode; EnvVariant = $EnvVariant;
             ProbeBank = $ProbeBank; TeamReward = $TeamReward }
if ($SkipStage1) {
  Write-Host ("[{0}] Stage 1 skipped." -f (Get-Date -Format s))
} else {
  Write-Host ("[{0}] Stage 1 start: {1}, {2} steps -> {3}" -f (Get-Date -Format s), ($Stage1Modes -join ","), $Stage1Timesteps, $Stage1Dir)
  try {
    & "$PSScriptRoot\run_from_scratch.ps1" @Common -Modes $Stage1Modes -Seeds $Stage1Seeds `
      -TotalTimesteps $Stage1Timesteps -OutputDir $Stage1Dir -StageLabel "${Name}_stage1"
  } catch {
    Write-Host ("[{0}] Stage 1 failed; stage 2 not started. {1}" -f (Get-Date -Format s), $_)
    exit 1
  }
  Write-Host ("[{0}] Stage 1 complete." -f (Get-Date -Format s))
}
Write-Host ("[{0}] Stage 2 start: {1}, {2} steps -> {3}" -f (Get-Date -Format s), ($Stage2Modes -join ","), $Stage2Timesteps, $Stage2Dir)
try {
  & "$PSScriptRoot\run_from_scratch.ps1" @Common -Modes $Stage2Modes -Seeds $Stage2Seeds `
    -TotalTimesteps $Stage2Timesteps -OutputDir $Stage2Dir -StageLabel "${Name}_stage2"
} catch {
  Write-Host ("[{0}] Stage 2 failed. {1}" -f (Get-Date -Format s), $_)
  exit 1
}
Write-Host ("[{0}] Pipeline complete." -f (Get-Date -Format s))
