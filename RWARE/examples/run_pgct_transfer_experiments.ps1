param(
  [string] $InitCheckpoint = "runs/pgct09700_physical_mixing_20260929_155338_079437/stage03_4ag_physical_home0.750/final.pt",
  [ValidateSet("pgct", "ippo", "random", "drop34", "drop34_refill")] [string[]] $Modes = @("pgct", "ippo"),
  [int[]] $Seeds = @(1, 2, 3),
  [int] $TotalTimesteps = 4000000,
  [int] $EvalSeed = 5000001,
  [int] $CriticEpisodes = 5,
  [int] $DiagnosticInterval = 5,
  [int] $PlotInterval = 50,
  [double] $PerformanceTarget = 45,
  [string] $OutputDir = "",
  [string] $ReferenceExperiment = "",
  [string] $ReuseExperiment = "",
  [switch] $NoTrack,
  [switch] $SaveVideo,
  [switch] $DryRun,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$TrainArgs = @("examples/train_pgct_transfer_experiments.py", "--init-checkpoint", $InitCheckpoint,
  "--timesteps", "$TotalTimesteps", "--eval-seed", "$EvalSeed", "--critic-episodes", "$CriticEpisodes",
  "--diagnostic-interval", "$DiagnosticInterval", "--plot-interval", "$PlotInterval", "--performance-target",
  $PerformanceTarget.ToString([Globalization.CultureInfo]::InvariantCulture), "--modes")
$TrainArgs += @($Modes | ForEach-Object { $_.ToLowerInvariant() })
$TrainArgs += "--seeds"
$TrainArgs += @($Seeds | ForEach-Object { "$_" })
if ($OutputDir) { $TrainArgs += @("--output-dir", $OutputDir) }
if ($ReferenceExperiment) { $TrainArgs += @("--reference-experiment", $ReferenceExperiment) }
if ($ReuseExperiment) { $TrainArgs += @("--reuse-experiment", $ReuseExperiment) }
if (-not $NoTrack -and -not $DryRun -and -not $SmokeTest) { $TrainArgs += "--track" }
if ($SaveVideo) { $TrainArgs += "--save-video" }
if ($DryRun) { $TrainArgs += "--dry-run" }
if ($SmokeTest) { $TrainArgs += "--smoke-test" }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Transfer experiment failed with exit code $LASTEXITCODE" }
