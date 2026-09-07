<#
.SYNOPSIS
    Wait for the running 12-fold sweep to finish, then power the machine off.

.DESCRIPTION
    Written 2026-08-18 to give the box a ~5 hour break after the A0 bs=64 sweep,
    which had been running ~3.5 days of continuous uptime.

    Runs on the WINDOWS side on purpose: the shutdown sequence includes
    `wsl --shutdown`, so a watcher living inside WSL would kill itself partway
    through. It polls WSL from outside instead.

    Behaviour (as chosen 2026-08-18):
      - Powers off REGARDLESS of whether the sweep succeeded or failed.
      - 60s grace after the orchestrator exits, so MLflow finishes writing
        meta.yaml / metrics before the filesystem goes away.
      - Clean `wsl --shutdown` first, then Windows shutdown.

.NOTES
    ABORT: create the file scripts\ABORT_SHUTDOWN (any contents) and the watcher
    exits without powering off, within one poll interval.

    Also aborts on its own if it has been running longer than -MaxHours, so a
    stuck sweep can never trigger a surprise shutdown a day later.
#>

[CmdletBinding()]
param(
    # Matched against the WSL process list. The orchestrator for this sweep.
    [string]$OrchestratorPattern = 'run_12fold_cv.sh',
    [int]$PollSeconds = 120,
    [int]$GraceSeconds = 60,
    # Windows shutdown countdown, giving a visible warning if anyone is at the desk.
    [int]$ShutdownDelaySeconds = 60,
    [int]$MaxHours = 12
)

$ErrorActionPreference = 'Continue'

$repo     = '\\wsl.localhost\ubuntu-22.04\home\miles\FE_WSTS'
$logPath  = Join-Path $repo 'logs\shutdown_watcher.log'
$abortPath = Join-Path $repo 'scripts\ABORT_SHUTDOWN'

function Write-Log {
    param([string]$Message)
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Output $line
    try { Add-Content -Path $logPath -Value $line -Encoding utf8 } catch { }
}

function Test-SweepRunning {
    # Returns $true while either the orchestrator OR any train.py is alive.
    # Checking both means we do not fire during the gap between two folds.
    #
    # The [r] / [s] bracket trick matters: `bash -c "pgrep -f 'run_12fold_cv.sh'"`
    # has the pattern in its OWN command line, so a naive pattern self-matches and
    # the count never reaches zero -- the watcher would poll forever and never fire.
    # Wrapping the first character in a character class matches the real process
    # but not the literal query string.
    $orchPattern = '[' + $OrchestratorPattern.Substring(0, 1) + ']' + $OrchestratorPattern.Substring(1)
    $cmd = "pgrep -f '$orchPattern' | wc -l; pgrep -f 'src/[t]rain.py' | wc -l"
    $out = & wsl.exe -d ubuntu-22.04 -- bash -c $cmd 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $out) {
        # WSL not answering. Treat as "still running" rather than risk an early
        # shutdown on a transient hiccup -- these were frequent under memory pressure.
        Write-Log 'WARN: could not query WSL; assuming sweep still running'
        return $true
    }
    $nums = @($out | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })
    if ($nums.Count -lt 2) { Write-Log 'WARN: unexpected WSL output; assuming running'; return $true }
    return (($nums[0] + $nums[1]) -gt 0)
}

Write-Log "=== watcher started (pid $PID) ==="
Write-Log "orchestrator pattern : $OrchestratorPattern"
Write-Log "poll ${PollSeconds}s | grace ${GraceSeconds}s | shutdown delay ${ShutdownDelaySeconds}s | max ${MaxHours}h"
Write-Log "abort by creating: $abortPath"

$deadline = (Get-Date).AddHours($MaxHours)

while ($true) {
    if (Test-Path $abortPath) {
        Write-Log 'ABORT file present - exiting without shutdown.'
        exit 0
    }
    if ((Get-Date) -gt $deadline) {
        Write-Log "Exceeded MaxHours=$MaxHours - exiting WITHOUT shutdown (safety guard)."
        exit 0
    }
    if (-not (Test-SweepRunning)) {
        Write-Log 'Sweep no longer running.'
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Log "Grace period ${GraceSeconds}s (letting MLflow flush)..."
Start-Sleep -Seconds $GraceSeconds

# Last abort check -- the grace window is the final chance to call it off.
if (Test-Path $abortPath) {
    Write-Log 'ABORT file appeared during grace - exiting without shutdown.'
    exit 0
}

# Record what the sweep actually produced, so the morning starts with a summary
# even though the machine went down.
try {
    Write-Log 'Final run inventory:'
    $py = '/home/miles/miniconda3/envs/WSTS_env/bin/python'
    $inv = & wsl.exe -d ubuntu-22.04 -- bash -c "cd /home/miles/FE_WSTS && $py scripts/inspect_mlflow_runs.py --filter bs64 --compare batch_size" 2>$null
    foreach ($l in $inv) { Write-Log "  $l" }
} catch {
    Write-Log "WARN: inventory failed: $_"
}

Write-Log 'Running wsl --shutdown for a clean VM stop...'
& wsl.exe --shutdown 2>$null
Start-Sleep -Seconds 10

Write-Log "Triggering Windows shutdown in ${ShutdownDelaySeconds}s. (cancel with: shutdown /a)"
& shutdown.exe /s /t $ShutdownDelaySeconds /c "Sweep complete - scheduled rest period."
Write-Log '=== watcher done ==='
