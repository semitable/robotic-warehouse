param(
  [ValidateSet("control", "detour", "bottleneck", "dock")] [string[]] $Profiles = @("detour"),
  [ValidateSet("pgct", "ippo")] [string[]] $Modes = @("pgct", "ippo"),
  [int[]] $Seeds = @(1),
  [int] $TotalTimesteps = 4000000,
  [int] $Requests = 4,
  [string] $InitCheckpoint = "runs/pgct09700_physical_mixing_20260929_155338_079437/stage03_4ag_physical_home0.750/final.pt",
  [string] $OutputDir = "",
  [switch] $Track,
  [switch] $SaveVideo,
  [switch] $DryRun,
  [switch] $SmokeTest,
  [switch] $PreviewOnly,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$TrainArgs = @("examples/train_hard_rware_experiments.py", "--init-checkpoint", $InitCheckpoint,
  "--timesteps", "$TotalTimesteps", "--requests", "$Requests", "--profiles")
$TrainArgs += @($Profiles | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--modes"
$TrainArgs += @($Modes | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--seeds"
$TrainArgs += @($Seeds | ForEach-Object { "$_" })
if ($OutputDir) { $TrainArgs += @("--output-dir", $OutputDir) }
if ($Track) { $TrainArgs += "--track" }
if ($SaveVideo) { $TrainArgs += "--save-video" }
if ($DryRun) { $TrainArgs += "--dry-run" }
if ($SmokeTest) { $TrainArgs += "--smoke-test" }
if ($PreviewOnly) { $TrainArgs += "--preview-only" }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Constrained RWARE experiment failed with exit code $LASTEXITCODE" }
