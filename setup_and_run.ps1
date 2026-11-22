# ============================================================================
#  setup_and_run.ps1  --  one script that does everything.
#
#  Normally you never run this directly: double-click RUN.bat instead.
#
#  It performs five steps, in order, and stops with a plain-language message if
#  anything goes wrong:
#     1. find a Python that TensorFlow supports (3.10 - 3.12)
#     2. create / repair the .venv
#     3. install the libraries (first time only, ~5 minutes)
#     4. fetch the 92 MB model (first time only)
#     5. start the app in the browser
#
#  Safe to run again at any time: every step is skipped if it is already done.
#
#  Switches (optional, for troubleshooting):
#     -DetectOnly   print what is installed and exit (changes nothing)
#     -Repair       delete .venv and rebuild it from scratch
#     -Reinstall    force pip install even if it looks done
# ============================================================================

param(
    [switch]$DetectOnly,
    [switch]$Repair,
    [switch]$Reinstall
)

# "Continue" on purpose: pip, winget, python and streamlit all write progress
# and warnings to stderr. With "Stop", PowerShell 5.1 turns those stderr lines
# into terminating errors and the install dies halfway through. Every failure
# that matters is checked explicitly through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$MIN_MINOR = 10          # lowest supported Python 3.x
$MAX_MINOR = 12          # highest supported Python 3.x (TensorFlow has no 3.13/3.14 wheels)
$VENV      = Join-Path $PSScriptRoot ".venv"
$VENV_PY   = Join-Path $VENV "Scripts\python.exe"

# ----------------------------------------------------------------------------
# pretty output helpers
# ----------------------------------------------------------------------------
function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }

function Step($n, $total, $text) {
    Write-Host ""
    Write-Host "[$n/$total] $text" -ForegroundColor Cyan
}

function Ok($text)   { Write-Host "        $text" -ForegroundColor Green }
function Info($text) { Write-Host "        $text" -ForegroundColor DarkGray }

function Fail($headline, $lines) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host " $headline" -ForegroundColor Red
    Write-Host "==================================================================" -ForegroundColor Red
    foreach ($line in $lines) { Write-Host " $line" -ForegroundColor Yellow }
    Write-Host ""
    Say "Nothing was damaged. Fix the point above and run RUN.bat again." "White"
    exit 1
}

# ----------------------------------------------------------------------------
# Python discovery
# ----------------------------------------------------------------------------
function Get-PythonVersion($exe) {
    try {
        $out = & $exe -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
    } catch { }
    return $null
}

function Find-SupportedPython {
    $found = @()

    # 1) the py launcher (installed with python.org builds)
    foreach ($minor in 12, 11, 10) {
        try {
            $exe = & py "-3.$minor" -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) {
                $ver = Get-PythonVersion $exe.Trim()
                if ($ver) { $found += [pscustomobject]@{ Exe = $exe.Trim(); Version = $ver } }
            }
        } catch { }
    }

    # 2) common install locations, in case the launcher is missing
    $roots = @(
        "$env:LOCALAPPDATA\Programs\Python",
        "$env:ProgramFiles\Python",
        "${env:ProgramFiles(x86)}\Python",
        "C:\"
    )
    foreach ($root in $roots) {
        foreach ($minor in 12, 11, 10) {
            foreach ($name in @("Python3$minor", "Python3$minor-32")) {
                $candidate = Join-Path (Join-Path $root $name) "python.exe"
                if (Test-Path $candidate) {
                    $ver = Get-PythonVersion $candidate
                    if ($ver) { $found += [pscustomobject]@{ Exe = $candidate; Version = $ver } }
                }
            }
        }
    }

    # 3) whatever "python" resolves to on PATH
    try {
        $onPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        if ($onPath) {
            $ver = Get-PythonVersion $onPath
            if ($ver) { $found += [pscustomobject]@{ Exe = $onPath; Version = $ver } }
        }
    } catch { }

    $supported = $found | Where-Object {
        $parts = $_.Version.Split(".")
        [int]$parts[0] -eq 3 -and [int]$parts[1] -ge $MIN_MINOR -and [int]$parts[1] -le $MAX_MINOR
    } | Sort-Object Version -Descending | Select-Object -First 1

    return [pscustomobject]@{ Supported = $supported; All = $found }
}

function Get-AnyPythonVersion {
    try {
        $onPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        if ($onPath) { return Get-PythonVersion $onPath }
    } catch { }
    return $null
}

# ----------------------------------------------------------------------------
# Step 1 - Python
# ----------------------------------------------------------------------------
Step 1 5 "Looking for Python 3.10 - 3.12"

$scan = Find-SupportedPython
$systemVersion = Get-AnyPythonVersion

if ($scan.Supported) {
    $pyExe = $scan.Supported.Exe
    Ok "found Python $($scan.Supported.Version)"
    Info $pyExe
} else {
    if ($systemVersion) {
        Info "this laptop has Python $systemVersion, which TensorFlow cannot use"
    }

    if ($DetectOnly) {
        Say "  -> no supported Python. Install it with:" "Yellow"
        Say "     winget install -e --id Python.Python.3.12" "Yellow"
        exit 2
    }

    Say "        no supported Python found - trying to install Python 3.12 automatically" "Yellow"
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        try {
            & winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
        } catch { }
        # winget exits 0 even when it decides to do nothing, so re-scan
        Start-Sleep -Seconds 3
        $scan = Find-SupportedPython
        if ($scan.Supported) {
            $pyExe = $scan.Supported.Exe
            Ok "installed Python $($scan.Supported.Version)"
        }
    }

    if (-not $scan.Supported) {
        Fail "Python 3.12 is required and could not be installed automatically." @(
            "",
            "Do this once, manually:",
            "",
            "   1. Open https://www.python.org/downloads/release/python-3129/",
            "   2. Download 'Windows installer (64-bit)'",
            "   3. Run it and TICK the box 'Add python.exe to PATH'",
            "   4. Close this window, then double-click RUN.bat again",
            "",
            "If you have Microsoft Store: the 'Python 3.12' app from the Store also works.",
            "",
            "Why not the Python 3.13 / 3.14 that is already here?",
            "TensorFlow 2.20 publishes no packages for them, so installation",
            "cannot succeed on 3.13 or 3.14."
        )
    }
}

# ----------------------------------------------------------------------------
# Step 2 - virtual environment
# ----------------------------------------------------------------------------
Step 2 5 "Preparing the project environment"

if ($Repair -and (Test-Path $VENV)) {
    Say "        -Repair given: deleting the old .venv" "Yellow"
    Remove-Item -Recurse -Force $VENV -ErrorAction SilentlyContinue
}

$needsRebuild = $false
if (Test-Path $VENV_PY) {
    $venvVersion = Get-PythonVersion $VENV_PY
    if (-not $venvVersion) {
        $needsRebuild = $true
        Info "existing .venv is broken -> rebuilding"
    } else {
        $parts = $venvVersion.Split(".")
        if ([int]$parts[1] -lt $MIN_MINOR -or [int]$parts[1] -gt $MAX_MINOR) {
            $needsRebuild = $true
            Info "existing .venv uses Python $venvVersion (unsupported) -> rebuilding"
        } else {
            Ok "using the existing .venv (Python $venvVersion)"
        }
    }
}

if ($needsRebuild) {
    Remove-Item -Recurse -Force $VENV -ErrorAction SilentlyContinue
}

if (-not (Test-Path $VENV_PY)) {
    Info "creating .venv (a few seconds)"
    & $pyExe -m venv $VENV
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VENV_PY)) {
        Fail "Could not create the virtual environment." @(
            "The Python installation looks incomplete.",
            "",
            "  - Reinstall Python 3.12 and make sure the 'pip' and 'venv'",
            "    components are selected in the installer.",
            "  - Then run RUN.bat again."
        )
    }
    Ok "created .venv"
}

$venvVersion = Get-PythonVersion $VENV_PY
if (-not $venvVersion) {
    Fail "The virtual environment is not usable." @(
        "Delete the folder .venv inside the project and run RUN.bat again.",
        "If that fails, reinstall Python 3.12."
    )
}

# ----------------------------------------------------------------------------
# Step 3 - libraries
# ----------------------------------------------------------------------------
Step 3 5 "Checking the libraries (TensorFlow is big - first time only)"

$tfDir = Join-Path $VENV "Lib\site-packages\tensorflow"
$depsDone = (Test-Path $tfDir) -and (Test-Path (Join-Path $VENV "Lib\site-packages\streamlit"))

if ($depsDone -and -not $Reinstall) {
    Ok "already installed - skipping the ~600 MB download"
} elseif ($DetectOnly) {
    Say "  -> libraries NOT installed yet" "Yellow"
} else {
    Info "this takes 3-8 minutes and needs internet"
    & $VENV_PY -m pip install --upgrade pip --quiet --disable-pip-version-check
    & $VENV_PY -m pip install -r requirements.txt --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        Fail "Installing the libraries failed." @(
            "Most common causes, in order:",
            "",
            "  1. No internet or a blocked connection",
            "     -> try a mobile hotspot, then run RUN.bat again",
            "",
            "  2. Antivirus / campus firewall blocking the download",
            "     -> pause the antivirus for 10 minutes and retry",
            "",
            "  3. A half-finished earlier attempt",
            "     -> run:  powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -Repair",
            "",
            "The pip error message is printed above this box."
        )
    }
    Ok "libraries installed"
}

# ----------------------------------------------------------------------------
# Step 4 - model weights
# ----------------------------------------------------------------------------
Step 4 5 "Checking the trained model (92 MB)"

$kerasModel = Join-Path $PSScriptRoot "models\plant_disease_resnet50.keras"
$npzModel   = Join-Path $PSScriptRoot "models\resnet_weights.npz"

if ((Test-Path $kerasModel) -or (Test-Path $npzModel)) {
    Ok "model already present - skipping the download"
} elseif ($DetectOnly) {
    Say "  -> model NOT downloaded yet" "Yellow"
} else {
    $attempt = 0
    $downloaded = $false
    while ($attempt -lt 3 -and -not $downloaded) {
        $attempt++
        Info "download attempt $attempt of 3"
        try {
            & $VENV_PY "$PSScriptRoot\scripts\get_weights.py"
            if ($LASTEXITCODE -eq 0 -and ((Test-Path $kerasModel) -or (Test-Path $npzModel))) {
                $downloaded = $true
            }
        } catch { }
        if (-not $downloaded -and $attempt -lt 3) {
            Info "retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }

    if (-not $downloaded) {
        Fail "The model could not be downloaded." @(
            "The weights file is 92 MB, from GitHub.",
            "",
            "  1. Check the internet (open github.com in a browser)",
            "  2. Run RUN.bat again - already-downloaded parts are kept",
            "  3. Or bring the file on a USB stick and run:",
            "",
            "     .\.venv\Scripts\python.exe scripts\get_weights.py --from `"E:\resnet_weights.npz`"",
            "",
            "  4. Or train your own: python train.py --data <split folder>"
        )
    }
    Ok "model ready"
}

# ----------------------------------------------------------------------------
# Step 5 - health check, then start
# ----------------------------------------------------------------------------
if ($DetectOnly) {
    Say ""
    Say "Detect-only run finished. Nothing was changed." "Cyan"
    exit 0
}

Step 5 5 "Final check, then starting the app"

# The check prints its own PASS/FAIL table - let it print straight to the
# console. Capturing native stderr in PowerShell 5.1 wraps every line in
# "NativeCommandError" decoration, which looks broken to a student.
Info "running the 9-point check (about 20 seconds)"
& $VENV_PY "$PSScriptRoot\scripts\check_setup.py"
if ($LASTEXITCODE -eq 0) {
    Ok "all nine checks passed"
} else {
    Say "        some checks failed - read the FAIL lines above" "Yellow"
    Say "        the app will still start; fix them before the exhibition" "Yellow"
}

# pick a free port so a stale copy of the app cannot block the demo
$port = 8501
foreach ($candidate in 8501, 8502, 8600, 8601) {
    $busy = $false
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $candidate)
        $busy = $true
        $client.Close()
    } catch { $busy = $false }
    if (-not $busy) { $port = $candidate; break }
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  READY" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  The app opens in your browser at:" -ForegroundColor White
Write-Host "      http://localhost:$port" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Upload a leaf photo to get a diagnosis." -ForegroundColor White
Write-Host "  Keep this black window open while you demo." -ForegroundColor White
Write-Host "  Press Ctrl + C here to stop the app." -ForegroundColor White
Write-Host ""

& $VENV_PY -m streamlit run app.py --server.port $port

Write-Host ""
Say "The app has stopped. You can close this window." "Gray"
exit 0
