# PowerShell script to run training_model.py with proper path handling
# This script handles paths with spaces correctly

# Get the script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainingScript = Join-Path $scriptPath "training_model.py"

# Check if virtual environment exists
$venvPath = Join-Path $scriptPath "venv"
$venvActivate = Join-Path $venvPath "Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $venvActivate
} else {
    Write-Host "Virtual environment not found. Using system Python." -ForegroundColor Yellow
}

# Change to script directory
Set-Location $scriptPath

# Run the training script
Write-Host "Starting training..." -ForegroundColor Green
python "$trainingScript"

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPress any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

