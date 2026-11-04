# User-launched 10M rerun: new W&B project, last 10 videos per job, short W&B paths.
param(
  [string] $Name = "",
  [int] $MaxConcurrent = 4,
  [switch] $PlanOnly
)
if (-not $Name) { $Name = "am2_" + (Get-Date -Format "MMdd_HHmmss") }
$QueueArgs = @{
  Candidates = @("b6_wide_corner_indiv_comm5_pen005")
  Modes = @("pgct_max", "ippo", "oracle_max", "allk_max")
  Seeds = @(1, 2, 3)
  TotalTimesteps = 10000000
  MaxConcurrent = $MaxConcurrent
  ModeMajor = $true
  Track = $true
  WandbProject = "rware-main-amax-v2"
  Name = $Name
  PlanOnly = $PlanOnly
  ExtraPlanArgs = @("--reward-probe-ac", "--ac-gate", "v2", "--ac-distance", "action_max",
                    "--ac-config", "latent_team/acv2_nomirror.json", "--team-reward", "individual",
                    "--save-video", "--video-keep-last", "10")
}
& (Join-Path $PSScriptRoot "run_queue.ps1") @QueueArgs
