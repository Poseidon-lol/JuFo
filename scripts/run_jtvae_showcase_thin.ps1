param(
    [int]$Rows = 1200,
    [string]$Device = "cuda:0",
    [switch]$NoAmp
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    if ($env:VIRTUAL_ENV) {
        $venvPy = Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"
        if (Test-Path $venvPy) {
            return @{ Path = $venvPy; UseLauncher = $false }
        }
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @{ Path = $pythonCmd.Source; UseLauncher = $false }
    }
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return @{ Path = $pyCmd.Source; UseLauncher = $true }
    }
    throw "No Python executable found. Activate .venv or install Python."
}

function Invoke-Py {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [bool]$UseLauncher,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    if ($UseLauncher) {
        & $PythonPath -3 @Args
    } else {
        & $PythonPath @Args
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Args -join ' ')"
    }
}

$pyInfo = Resolve-Python
$PythonPath = $pyInfo.Path
$UseLauncher = [bool]$pyInfo.UseLauncher

Write-Host "[1/2] Building lightweight showcase dataset..." -ForegroundColor Cyan
Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
    "scripts/build_jtvae_showcase_dataset.py",
    "--max-rows", "$Rows"
)

Write-Host "[2/2] Starting JT-VAE showcase training..." -ForegroundColor Cyan
$cmd = @(
    "-m", "src.main",
    "train-generator",
    "--config", "configs/gen_conf_showcase_thin.yaml",
    "--device", $Device
)

if ($NoAmp) {
    $cmd += "--no-amp"
} else {
    $cmd += "--amp"
}

Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args $cmd

Write-Host "Done. Dashboard: experiments/showcase_thin/live_decode_dashboard.html" -ForegroundColor Green
