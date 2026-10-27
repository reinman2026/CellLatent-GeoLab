param(
    [ValidateRange(1, 10000)][int]$Epochs = 1000,
    [switch]$Serve,
    [string]$PythonPath = 'E:\PycharmProjects\rna_window_response_hlc_snapshot-regnet\ref\regvelo\localtest\.venv_regvelo\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = 'E:\PycharmProjects\rna_window_response_hlc_snapshot-regnet\ref\regvelo\src'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python interpreter does not exist: $PythonPath"
}

# These variables apply to child processes in this run and are restored below.
$previousNumba = $env:NUMBA_DISABLE_JIT
$previousPythonPath = $env:PYTHONPATH
$previousUtf8 = $env:PYTHONUTF8
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
Push-Location -LiteralPath $projectRoot
try {
    $env:NUMBA_DISABLE_JIT = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $taskTemp = Join-Path $projectRoot '.runtime/temp'
    New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    $env:PYTHONPATH = if ($previousPythonPath) { "$sourceRoot;$previousPythonPath" } else { $sourceRoot }
    & $PythonPath tools/train_regvelo_replicas.py --preflight --probe-imports
    if ($LASTEXITCODE -ne 0) { throw 'Actual imports failed; training was not started. See preflight output.' }
    if ($Serve) {
        & $PythonPath tools/serve_regvelo.py --python $PythonPath
        if ($LASTEXITCODE -ne 0) { throw 'The local training server exited with an error.' }
    } else {
        & $PythonPath tools/train_regvelo_replicas.py --epochs $Epochs --samples 30 --workers 3
        if ($LASTEXITCODE -ne 0) { throw 'Complete RegVelo training failed. See the Python traceback above.' }
        & powershell -NoProfile -ExecutionPolicy Bypass -File tools/publish-regvelo-results.ps1
        if ($LASTEXITCODE -ne 0) { throw 'Training finished, but result publication failed.' }
        & node tools/build-offline.mjs
        if ($LASTEXITCODE -ne 0) { throw 'Training finished, but the offline page could not be rebuilt.' }
        Write-Host 'Three complete RegVelo models finished. Open the updated offline HTML.'
    }
} finally {
    $env:NUMBA_DISABLE_JIT = $previousNumba
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONUTF8 = $previousUtf8
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    Pop-Location
}
