# Runs every benchmark suite at below-normal priority with 4 threads, one JSON
# file per suite in research/results/. Usage: .\scripts\run-benchmarks.ps1 [-Quick] [-Exe path\to\pe.exe]
param([switch]$Quick, [string]$Exe = "")
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Exe) { $Exe = Join-Path $root 'target\release\pe.exe' }
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}
$env:RAYON_NUM_THREADS = '4'
New-Item -ItemType Directory -Force research\results | Out-Null
$suites = 'eval', 'equity', 'cfr', 'analytic', 'river', 'abstraction', 'exploit', 'multiway', 'players', 'turn', 'ranges', 'review'
foreach ($s in $suites) {
    $t = [Diagnostics.Stopwatch]::StartNew()
    $args = @('bench', $s, '--out', "research/results/bench-$s.json")
    if ($Quick) { $args += '--quick' }
    & $Exe @args
    "{0} finished in {1:N0} s (exit {2})" -f $s, $t.Elapsed.TotalSeconds, $LASTEXITCODE
}
"all suites done"
