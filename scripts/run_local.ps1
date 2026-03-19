$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Write-Error ".env was not found. Copy .env.example to .env and fill in your values first."
}

$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment not found. Run .\scripts\install_local.ps1 first."
}

& $pythonExe -m app
