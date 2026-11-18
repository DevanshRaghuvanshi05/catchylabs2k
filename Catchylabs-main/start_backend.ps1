<#
.SYNOPSIS
    Starts the CatchyLab FastAPI backend for local development on Windows PowerShell.

.DESCRIPTION
    - Automatically discovers backend directory and existing virtual environment (.venv)
    - Creates virtual environment and installs requirements.txt only if missing
    - Launches Uvicorn with auto-reload enabled
    - Displays local URLs for API, /health, and /docs

.PARAMETER HostAddress
    The interface to bind to. Default: 127.0.0.1 (or 0.0.0.0 to listen on all interfaces)

.PARAMETER Port
    The port to listen on. Default: 8000

.EXAMPLE
    .\start_backend.ps1
    .\start_backend.ps1 -HostAddress 0.0.0.0 -Port 8000
#>

param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "         CatchyLab Backend Local Development Launcher        " -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

# 1. Resolve Backend Directory
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}

if (Test-Path (Join-Path $ScriptDir "Catchylabs-main\app.py")) {
    $BackendDir = Join-Path $ScriptDir "Catchylabs-main"
} elseif (Test-Path (Join-Path $ScriptDir "app.py")) {
    $BackendDir = $ScriptDir
} else {
    Write-Error "Could not locate CatchyLab backend directory. Please run this script from the workspace root."
    exit 1
}

Set-Location $BackendDir
Write-Host "[+] Working Directory: $BackendDir" -ForegroundColor Green

# 2. Check or Create Virtual Environment
$VenvDir = Join-Path $BackendDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvUvicorn = Join-Path $VenvDir "Scripts\uvicorn.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[!] Virtual environment not found. Creating .venv..." -ForegroundColor Yellow
    
    # Try finding system Python
    $SysPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $SysPython) {
        $SysPython = (Get-Command py -ErrorAction SilentlyContinue).Source
    }
    if (-not $SysPython) {
        Write-Error "Python 3.10+ is required but not found in PATH."
        exit 1
    }

    & $SysPython -m venv $VenvDir
    Write-Host "[+] Virtual environment created successfully." -ForegroundColor Green
}

# 3. Check Dependencies
if (-not (Test-Path $VenvUvicorn)) {
    Write-Host "[!] Core dependencies missing in .venv. Installing from requirements.txt..." -ForegroundColor Yellow
    $ReqFile = Join-Path $BackendDir "requirements.txt"
    if (Test-Path $ReqFile) {
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r $ReqFile
        Write-Host "[+] Dependencies installed successfully." -ForegroundColor Green
    } else {
        Write-Error "requirements.txt not found at $ReqFile"
        exit 1
    }
}

# 4. Check Environment Configuration (.env)
$EnvFile = Join-Path $BackendDir ".env"
if (-not (Test-Path $EnvFile)) {
    $EnvExample = Join-Path $BackendDir ".env.example"
    Write-Host "[i] No .env file found. Backend will start in local development mode" -ForegroundColor Yellow
    Write-Host "    (using in-memory SQLite and default local settings)." -ForegroundColor Yellow
    Write-Host "    To connect to Supabase/PostgreSQL, copy .env.example to .env and configure DATABASE_URL." -ForegroundColor DarkGray
}

# 5. Display Endpoints
Write-Host "`n-------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "  Service Status:    Starting Uvicorn..." -ForegroundColor Green
Write-Host "  Local API URL:     http://${HostAddress}:${Port}" -ForegroundColor White
Write-Host "  Health Check:      http://${HostAddress}:${Port}/health" -ForegroundColor White
Write-Host "  Interactive Docs:  http://${HostAddress}:${Port}/docs" -ForegroundColor White
Write-Host "  ReDoc Schema:      http://${HostAddress}:${Port}/redoc" -ForegroundColor White
Write-Host "-------------------------------------------------------------`n" -ForegroundColor Cyan
Write-Host "Press CTRL+C to stop the backend server.`n" -ForegroundColor DarkGray

# 6. Launch Uvicorn Server with reload mode
try {
    & $VenvUvicorn app:app --host $HostAddress --port $Port --reload
} catch {
    Write-Host "`n[!] Uvicorn exited with error: $_" -ForegroundColor Red
    Write-Host "    Run 'python check_backend.py' to diagnose environment issues." -ForegroundColor Yellow
    exit 1
}
