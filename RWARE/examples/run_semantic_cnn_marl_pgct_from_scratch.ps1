param(
  [ValidateSet("none", "oracle", "policy-complete", "policy-physical", "wrong", "unrestricted", "all")]
  [string] $Run = "policy-complete",
  [bool] $CommunicationEnabled = $true,
  [bool] $SaveEvalVideo = $true,
  [int] $TotalTimesteps = 5000000,
  [int] $ProbeInterval = 5,
  [int] $SensorRange = 5,
  [ValidateSet(4, 8, 16)]
  [int] $AgentCount = 4,
  [ValidateRange(0.5, 1.0)]
  [double] $ShelfHomeRatio = 1.0,
  [int] $ProbeBatchSize = 48,
  [ValidateSet("shared")]
  [string] $ProbeTeamConditioning = "shared",
  [int] $Seed = 1,
  [string] $InitCheckpoint = "",
  [switch] $NoTrack,
  [double] $PgctDistanceThreshold = 0.12,
  [double] $PgctPeerLossCoef = 0.01,
  [int] $PgctWarmupUpdates = 100,
  [int] $MinProbeCount = 8,
  [double] $EntropyCoef = 0.05,
  [string] $EnvKwargsJson = "@examples/env_kwargs_stage2_pgct_success_unblock_zones_individual.json",
  [string] $RunNameSuffix = "",
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)

$ErrorActionPreference = "Stop"
if (-not $CommunicationEnabled) { $Run = "none" }
Set-Location (Split-Path $PSScriptRoot -Parent)
$ConfigText = $EnvKwargsJson
if ($ConfigText.StartsWith("@")) {
  $ConfigText = Get-Content -LiteralPath $ConfigText.Substring(1) -Raw
}
$Config = $ConfigText | ConvertFrom-Json
$Config | Add-Member -Force NoteProperty shelf_team_mode "balanced_soft_zones"
$Config | Add-Member -Force NoteProperty shelf_soft_zone_ratio $ShelfHomeRatio
$Config | Add-Member -Force NoteProperty request_queue_size $AgentCount
$Config | Add-Member -Force NoteProperty request_queue_size_per_team ([int]($AgentCount / 2))
$Config | Add-Member -Force NoteProperty normalised_coordinates $true
# ex5: square task observation; graph eligibility is independent of heading.
$Config.PSObject.Properties.Remove("directional_observation")
$Config.PSObject.Properties.Remove("communication_fov_degrees")
# Keep a per-run config on disk: native Windows argument quoting can damage JSON.
$ConfigDir = Join-Path $PSScriptRoot "generated_configs"
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$ConfigPath = Join-Path $ConfigDir ("cnn_{0}.json" -f [guid]::NewGuid().ToString("N"))
$Config | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $ConfigPath -Encoding ASCII

$Common = @(
  "examples\train_recurrent_ippo_consensus.py",
  "--env-id", "rware-multiteam-tiny-4ag-2teams-v0",
  "--agent-count", "$AgentCount",
  "--team-count", "2",
  "--sensor-range", "$SensorRange",
  "--env-kwargs-json", "@$ConfigPath",
  "--seed", "$Seed",
  "--observation-format", "semantic",
  "--obs-encoder", "cnn",
  "--total-timesteps", "$TotalTimesteps",
  "--rollout-steps", "1024",
  "--sequence-length", "64",
  "--learning-rate", "2e-4",
  "--entropy-coef", "$EntropyCoef",
  "--eval-interval", "50",
  "--eval-episodes", "20",
  "--eval-horizon", "500",
  "--save-interval", "100",
  "--probe-interval", "$ProbeInterval",
  "--min-probe-count", "$MinProbeCount",
  "--pgct-min-policy-information", "0",
  "--pgct-gate-style", "threshold",
  "--fixed-canonical-probes",
  "--wandb-project", "rware-curriculum",
  "--wandb-group", "semantic_cnn_pgct_from_scratch",
  "--wandb-log-eval-video",
  "--wandb-video-interval", "5"
)
if (-not $NoTrack) { $Common += "--track" }
if ($SaveEvalVideo) { $Common += "--save-eval-video" }
if ($InitCheckpoint) {
  $Common += @("--init-checkpoint", $InitCheckpoint, "--transfer-components", "all")
}

function Invoke-Experiment {
  param(
    [string] $Name,
    [string[]] $ExtraArgs
  )

  Write-Host ""
  $RatioLabel = $ShelfHomeRatio.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
  $ViewLabel = "square"
  $ExperimentName = "ex5_${Name}_${AgentCount}ag_sr${SensorRange}_${ViewLabel}_home${RatioLabel}_seed${Seed}"
  if (-not [string]::IsNullOrWhiteSpace($RunNameSuffix)) {
    $ExperimentName = "${ExperimentName}_${RunNameSuffix}"
  }

  Write-Host "=== Starting $ExperimentName ==="
  & $Python @Common @ExtraArgs --wandb-name $ExperimentName --exp-name $ExperimentName
  if ($LASTEXITCODE -ne 0) {
    throw "$ExperimentName failed with exit code $LASTEXITCODE"
  }
}

$RunNone = @(
  "--no-communication",
  "--graph-mode", "none",
  "--comm-graph-mode", "none",
  "--probe-source", "objective",
  "--objective-probe-fraction", "1.0",
  "--policy-probe-team-conditioning", "$ProbeTeamConditioning",
  "--policy-probe-batch-size", "$ProbeBatchSize",
  "--pgct-probe-sequence-length", "1",
  "--peer-transfer-mode", "none",
  "--critic-consensus-tau", "0",
  "--consensus-interval", "0"
)

$RunOracle = @(
  "--graph-mode", "oracle",
  "--comm-graph-mode", "complete",
  "--probe-source", "objective",
  "--objective-probe-fraction", "1.0",
  "--policy-probe-team-conditioning", "$ProbeTeamConditioning",
  "--policy-probe-batch-size", "$ProbeBatchSize",
  "--peer-transfer-mode", "pgct",
  "--pgct-probe-sequence-length", "1",
  "--pgct-peer-loss-coef", "$PgctPeerLossCoef",
  "--critic-consensus-tau", "0",
  "--consensus-interval", "0"
)

$RunPolicyComplete = @(
  "--graph-mode", "policy",
  "--comm-graph-mode", "complete",
  "--probe-source", "objective",
  "--objective-probe-fraction", "1.0",
  "--policy-probe-team-conditioning", "$ProbeTeamConditioning",
  "--policy-probe-batch-size", "$ProbeBatchSize",
  "--peer-transfer-mode", "pgct",
  "--pgct-probe-sequence-length", "1",
  "--pgct-warmup-updates", "$PgctWarmupUpdates",
  "--pgct-distance-ema-beta", "0.35",
  "--pgct-distance-temperature", "0.25",
  "--pgct-distance-threshold", "$PgctDistanceThreshold",
  "--pgct-peer-loss-coef", "$PgctPeerLossCoef",
  "--critic-consensus-tau", "0",
  "--consensus-interval", "0"
)

$RunPolicyPhysical = @(
  "--graph-mode", "policy",
  "--comm-graph-mode", "physical",
  "--probe-source", "objective",
  "--objective-probe-fraction", "1.0",
  "--policy-probe-team-conditioning", "$ProbeTeamConditioning",
  "--policy-probe-batch-size", "$ProbeBatchSize",
  "--peer-transfer-mode", "pgct",
  "--pgct-probe-sequence-length", "1",
  "--pgct-warmup-updates", "$PgctWarmupUpdates",
  "--pgct-distance-ema-beta", "0.35",
  "--pgct-distance-temperature", "0.25",
  "--pgct-distance-threshold", "$PgctDistanceThreshold",
  "--pgct-peer-loss-coef", "$PgctPeerLossCoef",
  "--critic-consensus-tau", "0",
  "--consensus-interval", "0"
)

if ($Run -eq "none" -or $Run -eq "all") {
  Invoke-Experiment "semantic_cnn_none_from_scratch" $RunNone
}
if ($Run -eq "oracle" -or $Run -eq "all") {
  Invoke-Experiment "semantic_cnn_oracle_from_scratch" $RunOracle
}
if ($Run -eq "policy-complete" -or $Run -eq "all") {
  Invoke-Experiment "semantic_cnn_policy_complete_from_scratch" $RunPolicyComplete
}
if ($Run -eq "policy-physical" -or $Run -eq "all") {
  Invoke-Experiment "semantic_cnn_policy_physical_from_scratch" $RunPolicyPhysical
}
foreach ($Baseline in @("wrong", "unrestricted")) {
  if ($Run -eq $Baseline -or $Run -eq "all") {
    $BaselineArgs = @($RunOracle | ForEach-Object { if ($_ -eq "oracle") { $Baseline } else { $_ } })
    Invoke-Experiment "semantic_cnn_${Baseline}_from_scratch" $BaselineArgs
  }
}
