param(
    [int]$Timesteps = 500000,
    [int]$EvalEpisodes = 20,
    [int]$Seed = 42,
    [double]$GnnBeta = 2.0,
    [double]$GnnThreshold = 0.001,
    [int]$GnnMcSamples = 4,
    [double]$GnnDropoutP = 0.05,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RlDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $RlDir
$OutDir = Join-Path $RlDir "outputs\extrapolation_fix"
$LogDir = Join-Path $RlDir "outputs\overnight_logs"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "gnn_penalty_overnight_$Stamp.log"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $RepoRoot

function Write-Log {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $Line
    Add-Content -Path $LogPath -Value $Line
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )

    Write-Log "START: $Name"
    Write-Log "COMMAND: $Python $($ArgsList -join ' ')"

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @ArgsList 2>&1 | ForEach-Object {
            $Line = $_.ToString()
            Write-Host $Line
            Add-Content -Path $LogPath -Value $Line
        }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode"
    }

    Write-Log "DONE: $Name"
}

try {
    Write-Log "Overnight GNN penalty SAC run"
    Write-Log "Repo root: $RepoRoot"
    Write-Log "Log file:  $LogPath"

    $DependencyCheck = @"
import importlib
import sys

print(f"python: {sys.executable}")
missing = []
for name in ("torch", "torch_geometric", "stable_baselines3"):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}: {version}")
    except Exception as exc:
        missing.append((name, type(exc).__name__, str(exc)))

if missing:
    print("missing dependencies:")
    for name, err_type, msg in missing:
        print(f"  {name}: {err_type}: {msg}")
    raise SystemExit(1)

print("dependencies ok")
"@

    Run-Step "Dependency/import check" @("-c", $DependencyCheck)

    Run-Step "Train and evaluate SAC with edge-state GNN penalty" @(
        "-m", "rl_dynamic_control.scripts.train_gnn_penalty_sac",
        "--timesteps", "$Timesteps",
        "--eval-episodes", "$EvalEpisodes",
        "--seed", "$Seed",
        "--gnn-beta", "$GnnBeta",
        "--gnn-threshold", "$GnnThreshold",
        "--gnn-mc-samples", "$GnnMcSamples",
        "--gnn-dropout-p", "$GnnDropoutP"
    )

    Run-Step "Generate extrapolation/reward comparison plots" @(
        "-m", "rl_dynamic_control.scripts.plot_extrapolation_fix"
    )

    Write-Log "SUCCESS: overnight run finished"
    Write-Log "Summary CSV: $OutDir\extrapolation_fix_results.csv"
    Write-Log "Plots dir:    $OutDir"
}
catch {
    Write-Log "FAILED: $($_.Exception.Message)"
    Write-Log "Check log: $LogPath"
    exit 1
}
