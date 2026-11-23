# Runs cargo at below-normal priority with a limited job count, offline.
# Usage:  .\scripts\cargo-gentle.ps1 build --release --package pe-cli
# (Use --package, not -p: PowerShell would bind -p to one of its own parameters.)
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CargoArgs)

$selfContained = Join-Path $env:USERPROFILE '.rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\self-contained'
foreach ($p in @((Join-Path $env:USERPROFILE '.cargo\bin'), $selfContained)) {
    if ((Test-Path $p) -and (($env:PATH -split ';') -notcontains $p)) { $env:PATH = "$p;$env:PATH" }
}
# Child processes (rustc, test binaries) inherit a below-normal priority class.
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}
if (-not $env:CARGO_BUILD_JOBS) { $env:CARGO_BUILD_JOBS = '4' }
if (-not $env:RAYON_NUM_THREADS) { $env:RAYON_NUM_THREADS = '4' }
$env:CARGO_NET_OFFLINE = 'true'
& cargo @CargoArgs
exit $LASTEXITCODE
