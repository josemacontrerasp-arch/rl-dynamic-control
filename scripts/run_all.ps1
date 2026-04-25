<#
.SYNOPSIS
    Runs the full GNN validation pipeline unattended.

.DESCRIPTION
    Executes, in sequence, and fails fast on any error:
        1. pytest test_flowsheet_graph.py       — regression check
        2. scripts/verify_dropout_fix.py        — dropout fix verify
        3. scripts/train_gnn_lcom_fix.py        — LCOM weighting experiment
        4. scripts/train_gnn_target_swap.py     — LCOM target-swap experiment

    * Any non-zero exit code from any step halts the pipeline.
    * Each step's stdout/stderr is tee'd to outputs/gnn_sweep/run_all_<step>.log.
    * A master log (outputs/gnn_sweep/run_all.log) captures timestamps,
      pass/fail status per step, and a summary at the end.
    * Script's own exit code is 0 only if every step succeeded.

.USAGE
    # From the rl_dynamic_control/ directory:
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/run_all.ps1

    # or in an already-active PowerShell (from rl_dynamic_control/):
    .\scripts\run_all.ps1

.NOTES
    Expects the .venv to already be on PATH (activated) — if not, the
    python calls will use the system python.  Activate beforehand with:
        .\.venv\Scripts\Activate.ps1
#>

$ErrorActionPreference = 'Stop'

# --- Paths --------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot   # rl_dynamic_control/
$OutDir   = Join-Path $RepoRoot 'outputs\gnn_sweep'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$MasterLog = Join-Path $OutDir 'run_all.log'
"" | Out-File $MasterLog -Encoding utf8  # truncate

function Write-Log {
    param([string]$Message)
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $line = "[$ts] $Message"
    Write-Host $line
    $line | Out-File $MasterLog -Append -Encoding utf8
}

function Invoke-Step {
    <#
        Runs a command, tees its combined output to a per-step log, and
        throws if its exit code is non-zero.  We use Start-Process + Wait
        so $LASTEXITCODE is reliable even through pipelines.
    #>
    param(
        [string]$Name,
        [string]$LogFile,
        [string]$Cmd,
        [string[]]$Arguments
    )

    Write-Log ""
    Write-Log "========================================================"
    Write-Log "STEP: $Name"
    Write-Log "CMD:  $Cmd $($Arguments -join ' ')"
    Write-Log "LOG:  $LogFile"
    Write-Log "========================================================"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # Run foreground so output streams live; tee everything to the step log.
    $proc = Start-Process -FilePath $Cmd `
                          -ArgumentList $Arguments `
                          -NoNewWindow `
                          -PassThru `
                          -Wait `
                          -RedirectStandardOutput $LogFile `
                          -RedirectStandardError  ($LogFile + '.err')

    $sw.Stop()
    $code = $proc.ExitCode
    $mins = [math]::Round($sw.Elapsed.TotalMinutes, 1)

    # Cat the step log to master log for a flat record
    Get-Content $LogFile | Out-File $MasterLog -Append -Encoding utf8
    if (Test-Path ($LogFile + '.err')) {
        $errContent = Get-Content ($LogFile + '.err')
        if ($errContent) {
            "--- stderr ---"              | Out-File $MasterLog -Append -Encoding utf8
            $errContent                   | Out-File $MasterLog -Append -Encoding utf8
        }
    }

    if ($code -ne 0) {
        Write-Log "STEP FAILED: $Name  (exit=$code, elapsed=${mins}m)"
        Write-Log "Check $LogFile and $($LogFile + '.err') for details."
        throw "Step '$Name' failed with exit code $code"
    }

    Write-Log "STEP OK:     $Name  (elapsed=${mins}m)"
}

# Force UTF-8 on Python's stdio so Unicode characters (→, ², ≥, ✓ …)
# in our print statements don't crash under Windows' default cp1252
# console. Belt-and-braces: the scripts also call sys.stdout.reconfigure.
$env:PYTHONIOENCODING = 'utf-8'

# --- Header --------------------------------------------------------
Write-Log "RUN_ALL pipeline starting"
Write-Log "Repo root : $RepoRoot"
Write-Log "Output dir: $OutDir"
Write-Log "Python    : $((Get-Command python).Source)"
Write-Log "PYTHONIOENCODING=$env:PYTHONIOENCODING"

Set-Location $RepoRoot
$OverallSw = [System.Diagnostics.Stopwatch]::StartNew()
$AllOk = $false
try {

    # ── Step 1: pytest regression check ───────────────────────────
    Invoke-Step -Name     "1/4 pytest regression" `
                -LogFile  (Join-Path $OutDir 'run_all_1_pytest.log') `
                -Cmd      "python" `
                -Arguments @("-m", "pytest", "test_flowsheet_graph.py", "-v")

    # ── Step 2: dropout fix verification (~15–20 min) ─────────────
    Invoke-Step -Name     "2/4 verify_dropout_fix" `
                -LogFile  (Join-Path $OutDir 'run_all_2_verify_dropout.log') `
                -Cmd      "python" `
                -Arguments @("scripts/verify_dropout_fix.py")

    # ── Step 3: LCOM weighting experiment (~14 min) ───────────────
    #    Kept in the pipeline even though the 2026-04-21 run
    #    empirically falsified the loss-weighting fix (R²_LCOM went
    #    0.249 → 0.213 under w=[1,1,3]).  Re-running it preserves the
    #    "full story" on disk so future-me can see the sequence of
    #    attempted fixes; it also costs ~14 min, so feel free to
    #    comment out for faster iteration once target-swap lands.
    Invoke-Step -Name     "3/4 train_gnn_lcom_fix" `
                -LogFile  (Join-Path $OutDir 'run_all_3_lcom_fix.log') `
                -Cmd      "python" `
                -Arguments @("scripts/train_gnn_lcom_fix.py")

    # ── Step 4: LCOM target-swap experiment (~14 min) ─────────────
    #    Option A per the LCOM diagnosis: swap LCOM out of the target
    #    vector for methanol production (t/yr), derive LCOM at
    #    inference as TAC_pred / production_pred.
    Invoke-Step -Name     "4/4 train_gnn_target_swap" `
                -LogFile  (Join-Path $OutDir 'run_all_4_target_swap.log') `
                -Cmd      "python" `
                -Arguments @("scripts/train_gnn_target_swap.py")

    $AllOk = $true
}
catch {
    Write-Log "PIPELINE ABORTED: $_"
}
finally {
    $OverallSw.Stop()
    $total = [math]::Round($OverallSw.Elapsed.TotalMinutes, 1)
    Write-Log ""
    Write-Log "========================================================"
    if ($AllOk) {
        Write-Log "PIPELINE FINISHED: all 4 steps OK  (total ${total}m)"
        Write-Log "========================================================"
        Write-Log ""
        Write-Log "Artefacts to review:"
        Write-Log "  $OutDir\verify_dropout_fix.csv"
        Write-Log "  $OutDir\lcom_fix_results.csv"
        Write-Log "  $OutDir\target_swap_results.csv"
        Write-Log "  $OutDir\run_all.log  (full combined log)"
        exit 0
    } else {
        Write-Log "PIPELINE FAILED  (total ${total}m)"
        Write-Log "See $MasterLog for the full trace."
        Write-Log "========================================================"
        exit 1
    }
}
