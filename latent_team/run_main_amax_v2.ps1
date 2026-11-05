# User-launched 10M rerun: new W&B project, last 10 videos per job, short W&B paths.
param(
  # T193: the wide map collapsed (no pickups, no deliveries by 1.1M); choose the environment explicitly.
  [Parameter(Mandatory = $true)]
  [ValidateSet("b6_corner_indiv_comm5_pen005", "b6_corner_indiv_comm5_req2_pen005", "b6_wide_corner_indiv_comm5_pen005")]
  [string] $Candidate,
  [string] $Name = "",
  [int] $MaxConcurrent = 4,
  # T195: smaller PC runs, e.g. -Seeds 1 -TotalTimesteps 7000000 (defaults keep the 10M x seeds 1-3 grid).
  [int[]] $Seeds = @(1, 2, 3),
  # T196: e.g. -Modes pgct_max,ippo for a one-day check on a single machine.
  [ValidateSet("pgct_max", "ippo", "oracle_max", "allk_max", "oracle_mature_max")]
  [string[]] $Modes = @("pgct_max", "ippo", "oracle_max", "allk_max"),
  [int] $TotalTimesteps = 10000000,
  # T200: no videos by default (metrics and graphs only). -Video turns them on again (last 10 kept locally);
  # -VideoUpload live = every eval video to W&B during training, finish = last 10 at the end.
  [switch] $Video,
  [ValidateSet("live", "finish")]
  [string] $VideoUpload = "live",
  [switch] $PlanOnly,
  # T189: default is seed by seed (seed 1: pgct_max, ippo, oracle_max, allk_max together, then seed 2, ...),
  # so the modes of one seed run side by side. -ModeMajor restores all seeds of one mode first.
  [switch] $ModeMajor
)
if (-not $Name) { $Name = "am2_" + (Get-Date -Format "MMdd_HHmmss") }
$QueueArgs = @{
  Candidates = @($Candidate)
  Modes = $Modes
  Seeds = $Seeds
  TotalTimesteps = $TotalTimesteps
  MaxConcurrent = $MaxConcurrent
  ModeMajor = [bool]$ModeMajor
  Track = $true
  WandbProject = "rware-main-amax-v2"
  Name = $Name
  PlanOnly = $PlanOnly
  ExtraPlanArgs = @("--reward-probe-ac", "--ac-gate", "v2", "--ac-distance", "action_max",
                    "--ac-config", "latent_team/acv2_nomirror.json", "--team-reward", "individual")
}
if ($Video) { $QueueArgs.ExtraPlanArgs += @("--save-video", "--video-keep-last", "10", "--video-upload", $VideoUpload) }
& (Join-Path $PSScriptRoot "run_queue.ps1") @QueueArgs
