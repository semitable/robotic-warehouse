param(
  [string] $InitCheckpoint = "runs/pgct09700_physical_mixing_20260929_155338_079437/stage03_4ag_physical_home0.750/final.pt",
  [ValidateSet("e1", "e2", "e3")] [string[]] $Maps = @("e1", "e2", "e3"),
  [ValidateSet("pgct", "ippo")] [string[]] $Modes = @("pgct", "ippo"),
  [int[]] $Seeds = @(1),
  [int] $TotalTimesteps = 4000000,
  [double] $ShelfHomeRatio = 0.5,
  [int] $RequestCount = 4,
  [int] $EvalSeed = 5000001,
  [string] $OutputDir = "",
  [switch] $NoTrack,
  [switch] $NoVideo,
  [switch] $DryRun,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Culture = [Globalization.CultureInfo]::InvariantCulture
$TrainArgs = @("examples/train_pgct_spatial_experiments.py", "--init-checkpoint", $InitCheckpoint,
  "--timesteps", "$TotalTimesteps", "--home-ratio", $ShelfHomeRatio.ToString($Culture),
  "--requests", "$RequestCount", "--eval-seed", "$EvalSeed", "--maps")
$TrainArgs += @($Maps | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--modes"
$TrainArgs += @($Modes | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--seeds"
$TrainArgs += @($Seeds | ForEach-Object { "$_" })
if ($OutputDir) { $TrainArgs += @("--output-dir", $OutputDir) }
if (-not $NoTrack -and -not $DryRun -and -not $SmokeTest) { $TrainArgs += "--track" }
if ($NoVideo) { $TrainArgs += "--no-video" }
if ($DryRun) { $TrainArgs += "--dry-run" }
if ($SmokeTest) { $TrainArgs += "--smoke-test" }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Spatial experiment failed with exit code $LASTEXITCODE" }
