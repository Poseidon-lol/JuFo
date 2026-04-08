param(
    [int]$Iterations = 50,
    [int]$SeedRows = 300,
    [int]$Seed = 42,
    [string]$Device = "auto",
    [string]$SurrogateDir = "models/surrogate_3d_full",
    [string]$GeneratorCkpt = "models/generator/jtvae_epoch_120.pt",
    [string]$SurrogateBootstrapConfig = "configs/train_conf_3d_full_showcase_bootstrap.yaml",
    [string]$GeneratorBootstrapConfig = "configs/gen_conf_showcase_bootstrap.yaml",
    [int]$GeneratorBootstrapRows = 1200,
    [string]$BootstrapSurrogateDir = "models/surrogate_3d_showcase_bootstrap",
    [string]$BootstrapGeneratorDir = "models/generator_showcase_bootstrap",
    [switch]$SkipPlot
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

function Test-LfsPointerFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    try {
        $line = Get-Content -Path $Path -TotalCount 1 -ErrorAction Stop
        return [bool]($line -like "version https://git-lfs.github.com/spec/v1*")
    } catch {
        return $false
    }
}

function Resolve-SurrogateCheckpoint {
    param([string]$PathSpec)
    if (-not (Test-Path $PathSpec)) {
        return $null
    }
    $item = Get-Item -LiteralPath $PathSpec
    if ($item.PSIsContainer) {
        $primary = Join-Path $item.FullName "schnet_full.pt"
        if (Test-Path $primary) {
            return $primary
        }
        $candidate = Get-ChildItem -Path $item.FullName -Filter "schnet_full*.pt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
        return $null
    }
    return $item.FullName
}

function Resolve-LatestJtvaeCheckpoint {
    param([string]$Directory)
    if (-not (Test-Path $Directory)) {
        return $null
    }
    $candidate = Get-ChildItem -Path $Directory -Filter "jtvae_epoch_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($candidate) {
        return $candidate.FullName
    }
    return $null
}

$pyInfo = Resolve-Python
$PythonPath = $pyInfo.Path
$UseLauncher = [bool]$pyInfo.UseLauncher

Write-Host "[Preflight] Python: $PythonPath" -ForegroundColor DarkCyan
Write-Host "[Preflight] Checking Python dependencies (torch, pandas)..." -ForegroundColor DarkCyan
try {
    Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @("-c", "import torch, pandas")
} catch {
    throw "Missing Python dependencies in current interpreter. Install at least: torch pandas"
}

Write-Host "[1/6] Build lightweight active-loop seed..." -ForegroundColor Cyan
Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
    "scripts/build_active_loop_showcase_seed.py",
    "--rows", "$SeedRows",
    "--seed", "$Seed"
)

Write-Host "[2/6] Validate checkpoints (and bootstrap if needed)..." -ForegroundColor Cyan
$SurrogateDirRuntime = $SurrogateDir
$GeneratorCkptRuntime = $GeneratorCkpt

$surrogateCkpt = Resolve-SurrogateCheckpoint -PathSpec $SurrogateDirRuntime
$needsSurrogateBootstrap = $false
if (-not $surrogateCkpt) {
    Write-Warning "Surrogate checkpoint not found in '$SurrogateDirRuntime'."
    $needsSurrogateBootstrap = $true
} elseif (Test-LfsPointerFile -Path $surrogateCkpt) {
    Write-Warning "Surrogate checkpoint is a Git-LFS pointer: $surrogateCkpt"
    $needsSurrogateBootstrap = $true
}
if ($needsSurrogateBootstrap) {
    if (-not (Test-Path $SurrogateBootstrapConfig)) {
        throw "Missing surrogate bootstrap config: $SurrogateBootstrapConfig"
    }
    Write-Host "[bootstrap] Training thin full-SchNet surrogate..." -ForegroundColor Yellow
    Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
        "-m", "src.main", "train-surrogate-3d-full",
        "--config", "$SurrogateBootstrapConfig",
        "--device", "$Device",
        "--no-amp"
    )
    $SurrogateDirRuntime = $BootstrapSurrogateDir
    $surrogateCkpt = Resolve-SurrogateCheckpoint -PathSpec $SurrogateDirRuntime
    if (-not $surrogateCkpt) {
        throw "Bootstrap surrogate training finished but no schnet_full*.pt was found in '$SurrogateDirRuntime'."
    }
    if (Test-LfsPointerFile -Path $surrogateCkpt) {
        throw "Bootstrap surrogate checkpoint is still an LFS pointer: $surrogateCkpt"
    }
}

$needsGeneratorBootstrap = $false
if (-not (Test-Path $GeneratorCkptRuntime)) {
    Write-Warning "Generator checkpoint not found: $GeneratorCkptRuntime"
    $needsGeneratorBootstrap = $true
} elseif (Test-LfsPointerFile -Path $GeneratorCkptRuntime) {
    Write-Warning "Generator checkpoint is a Git-LFS pointer: $GeneratorCkptRuntime"
    $needsGeneratorBootstrap = $true
} else {
    $adjacentVocab = Join-Path (Split-Path -Parent $GeneratorCkptRuntime) "fragment_vocab.json"
    if (-not (Test-Path $adjacentVocab)) {
        Write-Warning "Generator fragment vocab missing next to checkpoint: $adjacentVocab"
        $needsGeneratorBootstrap = $true
    } elseif (Test-LfsPointerFile -Path $adjacentVocab) {
        Write-Warning "Generator fragment vocab is a Git-LFS pointer: $adjacentVocab"
        $needsGeneratorBootstrap = $true
    }
}
if ($needsGeneratorBootstrap) {
    if (-not (Test-Path $GeneratorBootstrapConfig)) {
        throw "Missing generator bootstrap config: $GeneratorBootstrapConfig"
    }
    Write-Host "[bootstrap] Building thin JT-VAE training dataset..." -ForegroundColor Yellow
    Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
        "scripts/build_jtvae_showcase_dataset.py",
        "--max-rows", "$GeneratorBootstrapRows"
    )
    Write-Host "[bootstrap] Training thin JT-VAE generator..." -ForegroundColor Yellow
    Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
        "-m", "src.main", "train-generator",
        "--config", "$GeneratorBootstrapConfig",
        "--device", "$Device",
        "--no-amp"
    )
    $latestGen = Resolve-LatestJtvaeCheckpoint -Directory $BootstrapGeneratorDir
    if (-not $latestGen) {
        throw "Bootstrap generator training finished but no jtvae_epoch_*.pt was found in '$BootstrapGeneratorDir'."
    }
    if (Test-LfsPointerFile -Path $latestGen) {
        throw "Bootstrap generator checkpoint is still an LFS pointer: $latestGen"
    }
    $bootstrapVocab = Join-Path $BootstrapGeneratorDir "fragment_vocab.json"
    if (-not (Test-Path $bootstrapVocab)) {
        throw "Bootstrap generator vocab missing: $bootstrapVocab"
    }
    if (Test-LfsPointerFile -Path $bootstrapVocab) {
        throw "Bootstrap generator vocab is still an LFS pointer: $bootstrapVocab"
    }
    $GeneratorCkptRuntime = $latestGen
}

Write-Host "Using surrogate: $SurrogateDirRuntime" -ForegroundColor DarkCyan
Write-Host "Using generator: $GeneratorCkptRuntime" -ForegroundColor DarkCyan

Write-Host "[3/6] Baseline run (ohne RL)..." -ForegroundColor Cyan
Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
    "-m", "src.main", "active-loop",
    "--config", "configs/active_learn_showcase_thin_baseline.yaml",
    "--iterations", "$Iterations",
    "--surrogate-dir", "$SurrogateDirRuntime",
    "--generator-ckpt", "$GeneratorCkptRuntime",
    "--use-pseudo-dft",
    "--device", "$Device",
    "--surrogate-device", "$Device",
    "--generator-device", "$Device"
)

Write-Host "[4/6] PPO run..." -ForegroundColor Cyan
Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
    "-m", "src.main", "active-loop",
    "--config", "configs/active_learn_showcase_thin_ppo.yaml",
    "--iterations", "$Iterations",
    "--surrogate-dir", "$SurrogateDirRuntime",
    "--generator-ckpt", "$GeneratorCkptRuntime",
    "--use-pseudo-dft",
    "--device", "$Device",
    "--surrogate-device", "$Device",
    "--generator-device", "$Device"
)
Write-Host "PPO live dashboard: experiments/showcase_ppo/ppo/active_loop_live_dashboard.html" -ForegroundColor DarkCyan

if ($SkipPlot) {
    Write-Host "[5/6] Skipping plot generation (--SkipPlot)." -ForegroundColor Yellow
    Write-Host "Done (runs completed, no figure generated)." -ForegroundColor Green
    return
}

Write-Host "[5/6] Build comparison figure..." -ForegroundColor Cyan
$plotPng = "experiments/showcase_ppo/ppo_vs_baseline_hitrate.png"
$plotSvg = "experiments/showcase_ppo/ppo_vs_baseline_hitrate.svg"
try {
    Invoke-Py -PythonPath $PythonPath -UseLauncher $UseLauncher -Args @(
        "scripts/plot_ppo_vs_baseline_hitrate.py",
        "--baseline", "experiments/showcase_ppo/baseline/active_learning_history.csv",
        "--ppo", "experiments/showcase_ppo/ppo/active_learning_history.csv",
        "--output", "$plotPng",
        "--score-threshold", "-2.0"
    )
    Write-Host "[6/6] Done." -ForegroundColor Green
    if (Test-Path $plotPng) {
        Write-Host "Figure: $plotPng" -ForegroundColor Green
    } elseif (Test-Path $plotSvg) {
        Write-Host "Figure: $plotSvg" -ForegroundColor Green
    } else {
        Write-Warning "Plot command finished but no figure file was found."
    }
} catch {
    Write-Warning "Runs completed, but plotting failed."
    Write-Warning "You can still inspect CSV outputs in experiments/showcase_ppo/*/active_learning_history.csv"
    return
}
