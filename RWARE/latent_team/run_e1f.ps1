# E1-F experiment (team goals at the two bottom corners). Two phases, each started separately:
#
#   -Phase pilot : IPPO only, individual reward and team reward (2 processes). Question: do the two teams
#                  learn different carrying directions, and do the probes see it?  Analyse with
#                  latent_team/e1f_team_divergence.py before deciding on the main phase.
#   -Phase main  : IPPO/PGCT/Oracle/All with the canonical bank, plus PGCT with probe bank v2 (5 processes).
#                  Canonical vs v2 PGCT separates the probe effect from the environment effect.
#
# Every run starts from random weights. Nothing here changes the trainer, the environment or existing runs.
param(
  [ValidateSet("pilot", "main")] [string] $Phase = "pilot",
  [int[]] $Seeds = @(1),
  [int] $TotalTimesteps = 0,
  [ValidateSet("individual", "team")] [string] $TeamReward = "individual",
  [string] $Name = "",
  [int] $ThreadsPerJob = 4,
  [switch] $Track,
  [string] $WandbProject = "rware-from-scratch",
  [switch] $PlanOnly,
  [switch] $SmokeTest,
  [string] $Python = "C:\Users\95101\anaconda3\envs\RWARE_MARL\python.exe"
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Name) { $Name = "e1f_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
if ($TotalTimesteps -le 0) { $TotalTimesteps = if ($Phase -eq "pilot") { 5000000 } else { 10000000 } }
$Common = @{ Parallel = $true; ThreadsPerJob = $ThreadsPerJob; Python = $Python; Seeds = $Seeds; Maps = @("e1");
             EnvVariant = "e1f"; TotalTimesteps = $TotalTimesteps; Track = $Track; WandbProject = $WandbProject;
             PlanOnly = $PlanOnly; SmokeTest = $SmokeTest }

if ($Phase -eq "pilot") {
  $Runs = @(
    @{ Out = "runs/${Name}_pilot_individual"; Modes = @("ippo"); Bank = "canonical"; Reward = "individual" },
    @{ Out = "runs/${Name}_pilot_team"; Modes = @("ippo"); Bank = "canonical"; Reward = "team" })
} else {
  $Runs = @(
    @{ Out = "runs/${Name}_main_canonical"; Modes = @("ippo", "pgct", "oracle", "all"); Bank = "canonical"; Reward = $TeamReward },
    @{ Out = "runs/${Name}_main_v2"; Modes = @("pgct"); Bank = "v2"; Reward = $TeamReward })
}
foreach ($Run in $Runs) {
  Write-Host ("[{0}] {1}: {2}, bank {3}, reward {4}, {5} steps" -f (Get-Date -Format s), $Run.Out,
    ($Run.Modes -join ","), $Run.Bank, $Run.Reward, $TotalTimesteps)
  & "$PSScriptRoot\run_from_scratch.ps1" @Common -Modes $Run.Modes -ProbeBank $Run.Bank -TeamReward $Run.Reward `
    -OutputDir $Run.Out -StageLabel "${Name}_$Phase"
}
if (-not $PlanOnly) {
  Write-Host "Analysis (read-only, any time):"
  foreach ($Run in $Runs) { Write-Host "  $Python latent_team/e1f_team_divergence.py --run $($Run.Out)" }
}
