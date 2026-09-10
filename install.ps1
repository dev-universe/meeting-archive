$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not [Environment]::Is64BitOperatingSystem) {
    throw '64-bit Windows is required.'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: winget install --id=astral-sh.uv -e . Then open install.cmd again.'
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'Install the NVIDIA driver for your GPU, restart Windows, then try again.'
}
uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 installation failed.' }
Write-Host 'Installing transcription environment (1/2)...'
& (Join-Path $PSScriptRoot 'setup.ps1')
Write-Host 'Installing speaker analysis environment (2/2)...'
& (Join-Path $PSScriptRoot 'setup-diarization.ps1')
Write-Host 'Ready. Models are downloaded separately from inside the app.'
