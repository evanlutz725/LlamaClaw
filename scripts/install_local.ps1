$ErrorActionPreference = "Stop"

$venvPath = ".venv"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' was not found. Install Python for Windows first."
}

if (-not (Test-Path $venvPath)) {
    py -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment Python executable was not found at $pythonExe"
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -e .[dev]

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Fill in your real values before running LlamaClaw."
}

Write-Host ""
Write-Host "Local install complete."
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Run the app with:"
Write-Host "  .\scripts\run_local.ps1"
