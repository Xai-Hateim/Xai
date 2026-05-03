# Renames the Desktop folder that CONTAINS this repo (the Arabic-named parent
# of xai-bench-main) to "hateem dodo".
#
# Windows will refuse if anything has this folder open (Cursor, terminals,
# Explorer inside the folder). Close Cursor / other apps using the path, then:
#   Right-click this file -> Run with PowerShell
# or from an EXTERNAL PowerShell (cwd not inside the folder):
#   powershell -ExecutionPolicy Bypass -File "FULL\PATH\TO\rename-parent-folder-to-hateem-dodo.ps1"

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentPath = Split-Path -Parent $repoRoot
$newName = 'hateem dodo'

if (-not (Test-Path (Join-Path $repoRoot '.git'))) {
    # Optional sanity: repo may not use git; still allow rename
}

if ((Split-Path -Leaf $parentPath) -eq $newName) {
    Write-Host "Already named: $newName"
    exit 0
}

Write-Host "Renaming:`n  $parentPath`n  -> $newName"
Rename-Item -LiteralPath $parentPath -NewName $newName
Write-Host "Done. New path: $(Join-Path (Split-Path -Parent $parentPath) $newName)"
