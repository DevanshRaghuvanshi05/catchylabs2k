# run_demo.ps1 -- Windows PowerShell launcher for CatchyLabs read-only demo.
#
# Parameters
#   -CheckOnly          Load .env, validate settings, print OK or the problem, then exit.
#   -WithWorker         Also start the RQ worker in a separate PowerShell window.
#   -ShowKey            Print the CATCHYLAB_API_KEY value when reporting it.
#   -ExpectProject <id> Abort unless DATABASE_URL host contains this Supabase project id.
#   -SkipDbCheck        Skip the db_check.py validation step before starting uvicorn.
#
# Behaviour
#   1. Reads .env from the script folder; existing env vars win over file values.
#   2. If CATCHYLAB_API_KEY is missing, offers to generate one with Python and
#      append it to .env (asks Y/N; never prints the key unless -ShowKey is given).
#   3. Runs db_check.py to validate DATABASE_URL before starting uvicorn (skip with -SkipDbCheck).
#   4. Sets DEMO_MODE=1 for the server process.
#   5. Uses the venv Python if a "venv" folder exists, otherwise uses python on PATH.

[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$WithWorker,
    [switch]$ShowKey,
    [switch]$SkipDbCheck,
    [switch]$NoDemo,
    [string]$ExpectProject = ""
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper: resolve which python executable to use
# ---------------------------------------------------------------------------
function Get-PythonExe {
    $venvPy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPy) { return $venvPy }
    return "python"
}

# ---------------------------------------------------------------------------
# Step 1: Load .env from the script folder.
#         Existing env vars win; blank lines and #-comments are skipped;
#         surrounding single or double quotes are stripped from values.
# ---------------------------------------------------------------------------
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $eqIdx = $line.IndexOf("=")
        if ($eqIdx -lt 1) { return }
        $key   = $line.Substring(0, $eqIdx).Trim()
        $val   = $line.Substring($eqIdx + 1).Trim()
        # Strip surrounding quotes (single or double)
        if (($val.StartsWith('"') -and $val.EndsWith('"')) -or
            ($val.StartsWith("'") -and $val.EndsWith("'"))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        # Existing env var wins
        $current = [System.Environment]::GetEnvironmentVariable($key, "Process")
        if ([string]::IsNullOrEmpty($current)) {
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
} else {
    Write-Host "INFO: No .env file found at $envFile -- relying on shell environment." -ForegroundColor Yellow
}

Write-Host "=== CatchyLabs Read-Only Demo Launcher ===" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Step 2: Ensure CATCHYLAB_API_KEY is present.
#         If missing, offer to generate one with Python secrets.
# ---------------------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($env:CATCHYLAB_API_KEY)) {
    Write-Host ""
    Write-Host "CATCHYLAB_API_KEY is not set in the environment or .env file." -ForegroundColor Yellow
    Write-Host "Would you like to generate a new key using Python secrets.token_urlsafe(32)" -ForegroundColor Yellow
    Write-Host "and append it to .env? [Y/N]" -ForegroundColor Yellow
    $answer = (Read-Host "Answer").Trim().ToUpper()
    if ($answer -eq "Y") {
        $pyExe = Get-PythonExe
        $newKey = & $pyExe -c "import secrets; print(secrets.token_urlsafe(32))"
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($newKey)) {
            Write-Error "ERROR: Failed to generate key with Python. Aborting."
            exit 1
        }
        # Append to .env (create if absent)
        Add-Content -Path $envFile -Value ""
        Add-Content -Path $envFile -Value "CATCHYLAB_API_KEY=$newKey"
        [System.Environment]::SetEnvironmentVariable("CATCHYLAB_API_KEY", $newKey, "Process")
        if ($ShowKey) {
            Write-Host "Generated and saved CATCHYLAB_API_KEY: $newKey" -ForegroundColor Green
        } else {
            Write-Host "Generated and saved CATCHYLAB_API_KEY (use -ShowKey to display it)." -ForegroundColor Green
        }
    } else {
        Write-Error "ERROR: CATCHYLAB_API_KEY is required. Aborting."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Step 3: Validate DATABASE_URL and extract host and user (never prints secrets).
# ---------------------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($env:DATABASE_URL)) {
    Write-Error "ERROR: DATABASE_URL is not set. Aborting."
    exit 1
}

$dbHost = "unknown-host"
$dbUser = ""
# Match standard postgresql://user:pass@host:port/db
# Extract user (before the first colon or @)
if ($env:DATABASE_URL -match '://([^:@]+)') {
    $dbUser = $Matches[1]
}
if ($env:DATABASE_URL -match '@([^:/@?]+)') {
    $dbHost = $Matches[1]
} elseif ($env:DATABASE_URL -match '://([^:/@?]+)') {
    $dbHost = $Matches[1]
}
Write-Host "Database Host : $dbHost" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 3b: -ExpectProject check
#          For Session Pooler URLs the project ref is in the username
#          (postgres.<ref>), not the host. Check both.
# ---------------------------------------------------------------------------
if ($ExpectProject -ne "") {
    $foundInHost = $dbHost -like "*$ExpectProject*"
    $foundInUser = $dbUser -like "*$ExpectProject*"
    if (-not ($foundInHost -or $foundInUser)) {
        Write-Error "ERROR: Neither host '$dbHost' nor user field contains expected project id '$ExpectProject'. Aborting."
        exit 1
    }
    if ($foundInHost) {
        Write-Host "Project check : OK (host contains '$ExpectProject')" -ForegroundColor Green
    } else {
        Write-Host "Project check : OK (user contains '$ExpectProject' -- Session Pooler)" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# Step 4: -CheckOnly exits here after printing status.
# ---------------------------------------------------------------------------
if ($CheckOnly) {
    Write-Host ""
    Write-Host "OK -- all checks passed. Exiting (--CheckOnly)." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------------------
# Step 5: Resolve Python and start server (and optionally the RQ worker).
# ---------------------------------------------------------------------------
$pyExe = Get-PythonExe
Write-Host "Python        : $pyExe" -ForegroundColor Cyan

if ($NoDemo) {
    $env:DEMO_MODE = "0"
    Write-Host "Demo Mode     : OFF (-NoDemo specified -- live research enabled)" -ForegroundColor Yellow
} else {
    $env:DEMO_MODE = "1"
    Write-Host "Demo Mode     : ON (default -- mutating requests blocked)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Step 5a: Run db_check.py to validate DATABASE_URL (unless -SkipDbCheck).
# ---------------------------------------------------------------------------
$dbCheckScript = Join-Path $PSScriptRoot "db_check.py"
if ($SkipDbCheck) {
    Write-Host "DB check      : skipped (-SkipDbCheck)" -ForegroundColor Yellow
} elseif (-not (Test-Path $dbCheckScript)) {
    Write-Host "DB check      : db_check.py not found, skipping" -ForegroundColor Yellow
} else {
    Write-Host "DB check      : running db_check.py ..." -ForegroundColor Cyan
    $dbCheckArgs = @()
    if ($ExpectProject -ne "") {
        $dbCheckArgs += "--expect-project"
        $dbCheckArgs += $ExpectProject
    }
    & $pyExe $dbCheckScript @dbCheckArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: db_check.py reported a problem (see above). Aborting."
        exit 1
    }
    Write-Host "DB check      : passed" -ForegroundColor Green
}

if ($WithWorker) {
    Write-Host "Starting RQ worker in a new window..." -ForegroundColor Yellow
    $workerCmd = "Write-Host '=== CatchyLab RQ Worker ===' -ForegroundColor Cyan; & '$pyExe' worker.py"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $workerCmd
}

Write-Host "Starting uvicorn on http://0.0.0.0:8000 with DEMO_MODE=1..." -ForegroundColor Yellow
& $pyExe -m uvicorn app:app --host 0.0.0.0 --port 8000
