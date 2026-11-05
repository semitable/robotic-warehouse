# Summarise an LBF or VMAS plan folder (identification, learning, cost): writes summary.md/json and jobs.csv.
#   powershell -ExecutionPolicy Bypass -File scripts\aggregate_experiment.ps1 -Experiment LBF -Name lbf_main
param(
  [Parameter(Mandatory = $true)] [ValidateSet("LBF", "VMAS")] [string] $Experiment,
  [Parameter(Mandatory = $true)] [string] $Name,
  [string] $Python = $(if ($env:PGCT_PYTHON) { $env:PGCT_PYTHON } else { "python" })
)
$ErrorActionPreference = "Stop"
$Package = @{ LBF = "pgct_lbf"; VMAS = "pgct_vmas" }[$Experiment]
$Repo = Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH = "$Repo\$Experiment;$Repo\PGCT"
$env:PYTHONIOENCODING = "utf-8"
& $Python -m "$Package.aggregate" --plan (Join-Path $Repo "$Experiment\runs\$Name")
