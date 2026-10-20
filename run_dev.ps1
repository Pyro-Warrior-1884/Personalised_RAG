# Launch script for Personalised RAG (Windows PowerShell).
#
# Activates the virtual environment, runs Django system checks, refuses to
# start if a server is already listening on port 8000, starts the development
# server, and opens the default browser.
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$Port = if ($env:PORT) { $env:PORT } else { 8000 }
$Url = "http://127.0.0.1:$Port/"

# --- Virtual environment -------------------------------------------------
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "No .venv found. Create one first:"
    Write-Host "  py -m venv .venv"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}
& $venvActivate

# --- Refuse to start a duplicate server ----------------------------------
$listening = $false
try {
    $response = Invoke-WebRequest -Uri $Url -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
    if ($response) { $listening = $true }
} catch {
    $listening = $false
}
if ($listening) {
    Write-Host "A server is already listening on $Url"
    Write-Host "Open that URL in your browser, or stop the other server first."
    exit 1
}

# --- Sanity checks --------------------------------------------------------
Write-Host "==> Running Django system checks"
python manage.py check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --- Start the server -----------------------------------------------------
Write-Host "==> Starting development server at $Url"
Write-Host "    Press Ctrl+C to stop it."
Start-Process $Url
python manage.py runserver "127.0.0.1:$Port"
