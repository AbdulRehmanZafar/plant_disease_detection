<#
    run.ps1 — start the Streamlit demo with the project's virtual environment.

    Usage:
        powershell -ExecutionPolicy Bypass -File .\run.ps1

    Extra arguments are passed to streamlit, e.g.:
        .\run.ps1 --server.port 8600
#>

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv found. Run setup.ps1 first:" -ForegroundColor Red
    Write-Host "    powershell -ExecutionPolicy Bypass -File .\setup.ps1"
    exit 1
}

if (-not (Test-Path "models\plant_disease_resnet50.keras") -and
    -not (Test-Path "models\resnet_weights.npz")) {
    Write-Host "No model found in models\." -ForegroundColor Yellow
    Write-Host "Download the trained weights first:" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\python.exe scripts\get_weights.py"
    Write-Host ""
    Write-Host "Continuing anyway - the app will show instructions in the browser."
    Write-Host ""
    Start-Sleep -Seconds 2
}

Write-Host "Starting the demo... the browser opens at http://localhost:8501" -ForegroundColor Green
& $venvPython -m streamlit run app.py @args
