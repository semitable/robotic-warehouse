param(
  [ValidateSet("e1", "e2", "e3")] [string[]] $Maps = @("e1"),
  [ValidateSet("ippo", "pgct", "oracle", "all", "wrong", "mappo", "seps", "hetdps")] [string[]] $Modes = @("ippo", "pgct", "oracle", "all"),
  [int[]] $Seeds = @(1),
  [int] $TotalTimesteps = 10000000,
  [double] $HomeRatio = 0.5,
  [double] $LearningRate = 2e-4,
  [double] $EntropyCoef = 0.05,
  [int] $SaveInterval = 250,
  [string] $OutputDir = "",
  [switch] $Parallel,
  [switch] $Wait,
  [int] $ThreadsPerJob = 4,
  [switch] $Track,
  [string] $WandbProject = "rware-from-scratch",
  [ValidateSet("online", "offline")] [string] $WandbMode = "online",
  [string] $StageLabel = "single",
  [ValidateSet("e1", "e1f")] [string] $EnvVariant = "e1",
  [ValidateSet("canonical", "v2")] [string] $ProbeBank = "canonical",
  [ValidateSet("individual", "team")] [string] $TeamReward = "individual",
  [switch] $SaveVideo,
  [switch] $PlanOnly,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
# Python warnings go to stderr; with "Stop" PowerShell 5.1 can treat them as errors. Exit codes are checked.
$ErrorActionPreference = "Continue"
$Inv = [Globalization.CultureInfo]::InvariantCulture
if (-not $OutputDir) { $OutputDir = "runs/scratch_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
$PlanArgs = @("latent_team/train_from_scratch.py", "--timesteps", "$TotalTimesteps",
  "--home-ratio", $HomeRatio.ToString($Inv), "--learning-rate", $LearningRate.ToString($Inv),
  "--entropy-coef", $EntropyCoef.ToString($Inv), "--save-interval", "$SaveInterval",
  "--output-dir", $OutputDir, "--stage-label", $StageLabel, "--wandb-project", $WandbProject,
  "--wandb-mode", $WandbMode, "--env-variant", $EnvVariant, "--probe-bank", $ProbeBank,
  "--team-reward", $TeamReward, "--maps")
$PlanArgs += @($Maps | ForEach-Object { $_.ToLowerInvariant() })
$PlanArgs += "--modes"
$PlanArgs += @($Modes | ForEach-Object { $_.ToLowerInvariant() })
$PlanArgs += "--seeds"
$PlanArgs += @($Seeds | ForEach-Object { "$_" })
if ($Track) { $PlanArgs += "--track" }
if ($SaveVideo) { $PlanArgs += "--save-video" }
if ($SmokeTest) { $PlanArgs += "--smoke-test" }

if (-not $Parallel) {
  if ($PlanOnly) { $PlanArgs += "--plan-only" }
  & $Python @PlanArgs
  if ($LASTEXITCODE -ne 0) { throw "From-scratch training failed with exit code $LASTEXITCODE" }
  return
}

# Parallel: create the plan once, then one background process per job.
& $Python @($PlanArgs + "--plan-only")
if ($LASTEXITCODE -ne 0) { throw "Planning failed with exit code $LASTEXITCODE" }
if ($PlanOnly) { return }
$Plan = Resolve-Path $OutputDir
$Manifest = Get-Content (Join-Path $Plan "experiment.json") -Raw | ConvertFrom-Json
$env:OMP_NUM_THREADS = "$ThreadsPerJob"
$env:MKL_NUM_THREADS = "$ThreadsPerJob"
$env:WANDB_DISABLE_GIT = "true"
$Processes = @()
foreach ($Job in $Manifest.jobs) {
  $Log = Join-Path $Plan ("{0}.log" -f $Job.name)
  $Err = Join-Path $Plan ("{0}.err.log" -f $Job.name)
  $Process = Start-Process -FilePath $Python -PassThru -WindowStyle Hidden `
    -ArgumentList @("latent_team/train_from_scratch.py", "--run-job", $Job.name, "--plan", "`"$Plan`"") `
    -RedirectStandardOutput $Log -RedirectStandardError $Err
  $Processes += $Process
  Write-Host ("Started {0} (PID {1}); log {2}" -f $Job.name, $Process.Id, $Log)
}
Write-Host "Progress: Get-Content <log> -Tail 5   /   comparison: $Python latent_team/train_from_scratch.py --compare-only --plan `"$Plan`""
if (-not $Wait) { return }

# Wait for every job, rebuild the comparison, and fail unless all jobs completed.
$Processes | Wait-Process
& $Python latent_team/train_from_scratch.py --compare-only --plan "$Plan"
$Failed = @()
foreach ($Job in $Manifest.jobs) {
  $StatusPath = Join-Path (Join-Path $Plan $Job.name) "job_status.json"
  $Status = if (Test-Path $StatusPath) { (Get-Content $StatusPath -Raw | ConvertFrom-Json).status } else { "missing" }
  if ($Status -ne "complete") { $Failed += "$($Job.name)=$Status" }
}
if ($Failed.Count -gt 0) { throw ("Jobs not complete: " + ($Failed -join ", ")) }
Write-Host "All jobs complete: $Plan"
