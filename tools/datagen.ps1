<#
.SYNOPSIS
Runs NNUE data generation: one engine process per worker, each playing
self-play games with the "gendata" command (src/datagen.h) into its own file.

.DESCRIPTION
Worker w writes <OutDir>\worker-<ww>.bin (and .params beside it), with seed
SeedBase + w, logging to <OutDir>\worker-<ww>.log. A worker whose file
already exists resumes after its last complete game, so a stopped run is
continued by starting it again with the same arguments. With -Wait the
script returns when every worker has finished (Hours elapsed, or Games
played); otherwise it returns at once and the workers run on.

.EXAMPLE
# The round 1 pilot: 24 workers for 1.5 hours.
& tools\datagen.ps1 -Engine engines\datagen-v3.exe -OutDir C:\FunComp\data\pilot -Hours 1.5
# Round 2: a network for the games, and adjudication.
& tools\datagen.ps1 -Engine engines\datagen-v3.exe -OutDir C:\FunComp\data\round2 -Hours 24 -SeedBase 4000 `
    -EvalFile C:/FunComp/data/nets/candidate.nnue -Extra 'adjscore=800 adjplies=8 adjphase=8 adjtb=1 adjpct=90'
#>
param(
    [Parameter(Mandatory)] [string] $Engine,
    [Parameter(Mandatory)] [string] $OutDir,
    [double] $Hours = 0,
    [int] $Games = 0,
    [int] $Workers = 24,
    # The first worker's number: workers First .. First + Workers - 1 (to add workers beside ones already running).
    [int] $First = 0,
    # A hard limit on each worker's working set, in MB (0: none). The tablebases are mapped files: every page a
    # worker touches stays in its working set, counted as memory in use, until all of memory is; with a limit the
    # pages beyond it wait in the standby list, which any program gets at once (2026-10-10, 6-piece tables, 24
    # workers at 512: the rate unchanged at 7.5M positions an hour, memory in use under 50% instead of 99%).
    [int] $MaxWorkingSetMB = 0,
    [int] $SeedBase = 1000,
    [string] $Syzygy = 'C:/FunComp/syzygy',
    [int] $SoftNodes = 10000,
    [int] $HardNodes = 40000,
    [int] $MaxOpenScore = 525,
    [int] $Hash = 16,
    # A network for the games (the engine's EvalFile); empty: the hand-written eval.
    [string] $EvalFile = '',
    # More gendata options, e.g. 'adjscore=800 adjplies=8 adjphase=8 adjtb=1 adjpct=90'.
    [string] $Extra = '',
    # More UCI options before the games, 'Name=Value' each (round 4: NnueScale=189, its generator's scale, on a
    # build made with -DTUNE=ON).
    [string[]] $Options = @(),
    [switch] $Wait
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Engine)) { throw "Not found: $Engine" }
if ($Hours -le 0 -and $Games -le 0) { throw 'Give -Hours or -Games' }
$enginePath = (Resolve-Path $Engine).Path
New-Item -ItemType Directory -Force $OutDir | Out-Null
$dir = (Resolve-Path $OutDir).Path

$procs = @()
for ($w = $First; $w -lt $First + $Workers; $w++) {
    $name = 'worker-{0:d2}' -f $w
    $out = Join-Path $dir "$name.bin"
    $cmds = Join-Path $dir "$name.in"
    $limits = @()
    if ($Hours -gt 0) { $limits += "hours=$Hours" }
    if ($Games -gt 0) { $limits += "games=$Games" }
    # The UCI pool's own hash is unused by gendata: keep it small.
    if ($Extra) { $limits += $Extra }
    $lines = @("setoption name SyzygyPath value $Syzygy", 'setoption name Hash value 1')
    if ($EvalFile) { $lines += "setoption name EvalFile value $EvalFile" }
    foreach ($o in $Options) { $k, $v = $o -split '=', 2; $lines += "setoption name $k value $v" }
    $lines += @(
        ("gendata out=$($out -replace '\\', '/') worker=$w seed=$($SeedBase + $w) " +
         "softnodes=$SoftNodes hardnodes=$HardNodes maxopenscore=$MaxOpenScore hash=$Hash " +
         ($limits -join ' ')),
        'quit'
    )
    $lines | Set-Content -Encoding ascii $cmds
    $procs += Start-Process -FilePath $enginePath -RedirectStandardInput $cmds `
        -RedirectStandardOutput (Join-Path $dir "$name.log") `
        -RedirectStandardError (Join-Path $dir "$name.err") -WindowStyle Hidden -PassThru
}
if ($MaxWorkingSetMB -gt 0) {
    Add-Type -Namespace FunComp -Name WorkingSet -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetProcessWorkingSetSizeEx(IntPtr process, IntPtr min, IntPtr max, uint flags);
'@
    foreach ($p in $procs) {
        # Flags: the hard minimum off (2), the hard maximum on (4). It lasts for the process.
        if (-not [FunComp.WorkingSet]::SetProcessWorkingSetSizeEx($p.Handle, [IntPtr](64MB), [IntPtr]($MaxWorkingSetMB * 1MB), 6)) {
            Write-Warning "No working-set limit on process $($p.Id)"
        }
    }
}
Write-Host "Started $Workers workers (from $First) into $dir (ids $(($procs | ForEach-Object Id) -join ' '))"
if ($Wait) {
    $procs | Wait-Process
    Write-Host 'All workers finished.'
}
