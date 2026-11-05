param(
  [ValidateSet("e1", "e2")] [string[]] $Maps = @("e1"),
  [ValidateSet("oracle", "all", "wrong")] [string[]] $Modes = @("oracle", "all"),
  [int[]] $Seeds = @(1),
  [int] $TotalTimesteps = 4000000,
  [int] $TransferWarmupUpdates = -1,
  [int] $EvalSeed = 5000001,
  [string] $OutputDir = "",
  [string] $ReferenceExperiment = "runs/sp_20261008_120638_983201",
  [switch] $Track,
  [switch] $SaveVideo,
  [switch] $DryRun,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$TrainArgs = @("latent_team/exp_b_oracle_consensus.py", "--timesteps", "$TotalTimesteps",
  "--eval-seed", "$EvalSeed", "--reference-experiment", $ReferenceExperiment, "--maps")
$TrainArgs += @($Maps | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--modes"
$TrainArgs += @($Modes | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--seeds"
$TrainArgs += @($Seeds | ForEach-Object { "$_" })
if ($TransferWarmupUpdates -ge 0) { $TrainArgs += @("--transfer-warmup-updates", "$TransferWarmupUpdates") }
if ($OutputDir) { $TrainArgs += @("--output-dir", $OutputDir) }
if ($Track) { $TrainArgs += "--track" }
if ($SaveVideo) { $TrainArgs += "--save-video" }
if ($DryRun) { $TrainArgs += "--dry-run" }
if ($SmokeTest) { $TrainArgs += "--smoke-test" }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment B failed with exit code $LASTEXITCODE" }
