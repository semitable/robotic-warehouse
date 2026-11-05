# Summarise an Overcooked plan folder (identification, learning, cost): writes summary.md/json and jobs.csv.
#   powershell -ExecutionPolicy Bypass -File Overcooked\scripts\aggregate.ps1 -Name oc_main
param(
  [Parameter(Mandatory = $true)] [string] $Name,
  [string] $Python = $(if ($env:PGCT_PYTHON) { $env:PGCT_PYTHON } else { "python" })
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONPATH = "$Repo\Overcooked;$Repo\PGCT"
$env:PYTHONIOENCODING = "utf-8"
& $Python -m pgct_overcooked.aggregate --plan (Join-Path $Repo "Overcooked\runs\$Name")
