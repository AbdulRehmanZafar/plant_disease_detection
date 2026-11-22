<#
    setup.ps1 — one-command environment setup on Windows.

    What it does
      1. Finds a Python between 3.10 and 3.12 (TensorFlow has no wheels for 3.13/3.14)
      2. Creates a .venv in this folder
      3. Installs requirements.txt
      4. Runs the fast NumPy self-test

    Usage (from this folder, in PowerShell):
        powershell -ExecutionPolicy Bypass -File .\setup.ps1

    If step 1 fails, install Python 3.12 first:
        winget install -e --id Python.Python.3.12
#>

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host " Plant Disease Detection - environment setup" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green

# ---------------------------------------------------------------- 1. Python
$python = $null
foreach ($ver in @("3.12", "3.11", "3.10")) {
    try {
        $out = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $python = $out.Trim()
            Write-Host "[1/4] Using Python $ver -> $python" -ForegroundColor Cyan
            break
        }
    } catch { }
}

if (-not $python) {
    Write-Host "[1/4] No Python 3.10-3.12 found." -ForegroundColor Red
    Write-Host ""
    Write-Host "  TensorFlow 2.20 does NOT support Python 3.13/3.14, so the system"
    Write-Host "  Python on this machine cannot run the project. Install 3.12:"
    Write-Host ""
    Write-Host "      winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Then close this window, open a new PowerShell, and run setup.ps1 again."
    exit 1
}

# ---------------------------------------------------------------- 2. venv
if (Test-Path ".venv") {
    Write-Host "[2/4] .venv already exists - reusing it" -ForegroundColor Cyan
} else {
    Write-Host "[2/4] Creating .venv (this takes a few seconds)" -ForegroundColor Cyan
    & $python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "venv python not found at $venvPython" }

# ---------------------------------------------------------------- 3. deps
Write-Host "[3/4] Installing dependencies (~600 MB, first run only)..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "pip install failed. Most common causes:" -ForegroundColor Red
    Write-Host "  * no internet / campus proxy   -> try a mobile hotspot"
    Write-Host "  * Python is 3.13+              -> install 3.12 (see above)"
    Write-Host "  * antivirus blocking wheels    -> pause it and retry"
    exit 1
}

# ---------------------------------------------------------------- 4. test
Write-Host "[4/4] Running the fast self-test..." -ForegroundColor Cyan
& $venvPython "tests\selftest.py"
$testExit = $LASTEXITCODE

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host " Done. Next steps:" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Get the trained weights (92 MB):" -ForegroundColor Yellow
Write-Host "      .\.venv\Scripts\python.exe scripts\get_weights.py"
Write-Host ""
Write-Host "  Verify everything:" -ForegroundColor Yellow
Write-Host "      .\.venv\Scripts\python.exe scripts\check_setup.py"
Write-Host ""
Write-Host "  Run the demo:" -ForegroundColor Yellow
Write-Host "      .\.venv\Scripts\python.exe -m streamlit run app.py"
Write-Host ""

if ($testExit -ne 0) {
    Write-Host "The self-test reported failures - read the output above." -ForegroundColor Red
    exit 1
}
