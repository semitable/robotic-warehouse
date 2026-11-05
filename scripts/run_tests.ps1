# Run the test suites (shared core, RWARE regression incl. pre-split equivalence, Overcooked, LBF, VMAS).
#   powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 [-Python <python.exe>] [-Only PGCT,Overcooked]
# RWARE real-trainer tests skip themselves when the local reference experiment / 1M checkpoints are absent.
# The RWARE suite needs pyglet<2 for its one rendering test (tests\test_policy_probe_consensus.py).
# LBF: the comparison with the original lbforaging skips itself unless `pip install lbforaging==2.0.0 --no-deps`.
# VMAS: the whole suite skips itself unless `pip install vmas==1.5.2` (GPLv3).
param(
  [string] $Python = $(if ($env:PGCT_PYTHON) { $env:PGCT_PYTHON } else { "python" }),
  [ValidateSet("PGCT", "RWARE", "Overcooked", "LBF", "VMAS")]
  [string[]] $Only = @("PGCT", "RWARE", "Overcooked", "LBF", "VMAS"),
  [string] $BaseTemp = $(Join-Path $env:TEMP "pgct_pytest")
)
$Repo = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $BaseTemp | Out-Null  # pytest creates --basetemp but not its parent
$Failed = @()
foreach ($suite in $Only) {
  Write-Host "== $suite"
  Push-Location (Join-Path $Repo $suite)
  try {
    # Short base folder: some RWARE trainer tests create deep job paths (Windows 260-character limit).
    & $Python -m pytest tests -q -p no:cacheprovider --basetemp (Join-Path $BaseTemp $suite) -W ignore
    if ($LASTEXITCODE -ne 0) { $Failed += $suite }
  } finally { Pop-Location }
}
if ($Failed.Count) { Write-Host "FAILED: $($Failed -join ', ')"; exit 1 }
Write-Host "all selected suites passed"
