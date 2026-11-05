param(
  [string] $InitCheckpoint = "runs/recurrent_ippo_consensus/ex5_semantic_cnn_policy_complete_from_scratch_4ag_sr5_square_home1.00_seed1/checkpoint_09700.pt",
  [ValidateSet(4, 6, 8, 16)] [int] $AgentCount = 4,
  [double[]] $ShelfHomeRatios = @(1.0, 0.875, 0.75, 0.5),
  [int[]] $StageTimesteps = @(1000000, 2000000, 3000000, 4000000),
  [double] $LearningRate = 0.0001,
  [double] $EntropyCoef = 0.02,
  [int] $Seed = 1,
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
$TrainArgs = @("examples/train_pgct_physical_mixing_curriculum.py",
  "--init-checkpoint", $InitCheckpoint, "--agent-count", "$AgentCount", "--seed", "$Seed",
  "--learning-rate", $LearningRate.ToString($Culture), "--entropy-coef", $EntropyCoef.ToString($Culture),
  "--home-ratios")
$TrainArgs += @($ShelfHomeRatios | ForEach-Object { $_.ToString($Culture) })
$TrainArgs += "--stage-timesteps"
$TrainArgs += @($StageTimesteps | ForEach-Object { "$_" })
if ($OutputDir) { $TrainArgs += @("--output-dir", $OutputDir) }
if (-not $NoTrack -and -not $SmokeTest -and -not $DryRun) { $TrainArgs += "--track" }
if ($NoVideo) { $TrainArgs += "--no-video" }
if ($DryRun) { $TrainArgs += "--dry-run" }
if ($SmokeTest) { $TrainArgs += "--smoke-test" }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Curriculum failed with exit code $LASTEXITCODE" }
