# One-time helper for checkouts that already hold untracked RWARE results at the OLD locations
# (<repo>\runs, <repo>\output) after pulling the 2026-11 split. Git moved the tracked files to RWARE\ but
# leaves ignored/untracked files where they were; the RWARE code now reads RWARE\runs and RWARE\output.
#
# Default is a dry run. -Apply moves each top-level entry of runs\ and output\ into RWARE\runs / RWARE\output
# only when no entry of the same name exists there (nothing is overwritten or deleted).
# Do NOT run while trainings write into runs\ (stop or let them finish first).
#   powershell -ExecutionPolicy Bypass -File scripts\migrate_local_layout.ps1          # show the plan
#   powershell -ExecutionPolicy Bypass -File scripts\migrate_local_layout.ps1 -Apply   # move
param([switch] $Apply)
$Repo = Split-Path $PSScriptRoot -Parent
foreach ($name in @("runs", "output")) {
  $old = Join-Path $Repo $name
  $new = Join-Path $Repo "RWARE\$name"
  if (-not (Test-Path $old)) { Write-Host "$old : nothing to move"; continue }
  if (-not (Test-Path $new)) { New-Item -ItemType Directory -Path $new | Out-Null }
  foreach ($item in Get-ChildItem -LiteralPath $old -Force) {
    $target = Join-Path $new $item.Name
    if (Test-Path -LiteralPath $target) { Write-Host "KEEP (target exists): $($item.FullName)"; continue }
    if ($Apply) { Move-Item -LiteralPath $item.FullName -Destination $target; Write-Host "moved $($item.Name) -> $target" }
    else { Write-Host "would move $($item.FullName) -> $target" }
  }
}
if (-not $Apply) { Write-Host "dry run only; add -Apply to move" }
