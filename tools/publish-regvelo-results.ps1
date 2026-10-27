$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root '.runtime/regvelo-runs/replicas.json'
$target = Join-Path $root 'dist/regvelo-results.mjs'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing completed result: $source" }
$json = [System.IO.File]::ReadAllText($source)
[System.IO.File]::WriteAllText($target, ('export const regveloReplicas = ' + $json + ';' + [Environment]::NewLine), [System.Text.UTF8Encoding]::new($false))
Write-Host "Published $target"
