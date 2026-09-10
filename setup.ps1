$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv .venv --python 3.11
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$requirementsFile = if (Test-Path -LiteralPath 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.txt' }
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python .venv\Scripts\python.exe -r $requirementsFile
} else {
    .\.venv\Scripts\python.exe -m ensurepip --upgrade
    .\.venv\Scripts\python.exe -m pip install -r $requirementsFile
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
.\.venv\Scripts\python.exe -m meeting_stt.diagnose
if ($LASTEXITCODE -ne 0) { throw 'GPU check failed. See output above.' }
Write-Host 'Ready. Double-click start.vbs to open Meeting Archive.'
