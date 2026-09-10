$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install uv, then run this script again.'
}
if (-not (Test-Path -LiteralPath '.venv-diarization\Scripts\python.exe')) {
    uv venv .venv-diarization --python 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
uv pip install --python .venv-diarization\Scripts\python.exe torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
$requirementsFile = if (Test-Path -LiteralPath 'requirements-diarization.lock.txt') { 'requirements-diarization.lock.txt' } else { 'requirements-diarization.txt' }
uv pip install --python .venv-diarization\Scripts\python.exe -r $requirementsFile --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
.\.venv-diarization\Scripts\python.exe -X utf8 -m meeting_stt.diagnose_diarization
if ($LASTEXITCODE -ne 0) { throw 'Diarization runtime check failed.' }
Write-Host 'Runtime ready. Open the app and use Speaker Setup to download the model.'
