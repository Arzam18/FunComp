<#
.SYNOPSIS
Plays an SPRT match between two engines with fastchess.

.DESCRIPTION
Stops as soon as the sequential probability ratio test decides between
H0 (the new engine is Elo0 or worse than the base) and H1 (it is Elo1 or
better), with 5% error rates both ways. Elo is normalized Elo, as on
OpenBench and fishtest.

Use -Elo0 0 -Elo1 5 for a change meant to gain strength, and
-Elo0 -5 -Elo1 0 for one that only has to do no harm (a simplification,
a speedup that changes the bench, a refactor).

Engine options go in as Name=Value strings. Every parameter in params.h
can be set this way, so one build can be played against itself with a
heuristic switched off or a margin changed. fastchess only sends options
the engine advertises (it skips the rest with a warning), so such runs
need a build configured with -DTUNE=ON. The script checks this and
refuses to start otherwise. (A build without TUNE has the parameters as
constants, FROZEN_PARAMS: told to change one, it stops with an error.)

.EXAMPLE
# Is the new build better than the pre-pruning engine?
tools\sprt.ps1 -New build\chess.exe -Base engines\v0.1-nopruning.exe -Book books\UHO_Lichess_4852_v1.epd

.EXAMPLE
# Does razoring earn its keep? The base is the same build with it off.
tools\sprt.ps1 -New build\chess.exe -Base build\chess.exe -BaseOptions 'RazorMaxDepth=0' -Book books\UHO_Lichess_4852_v1.epd

.EXAMPLE
# A larger futility margin, as a non-regression test.
tools\sprt.ps1 -New build\chess.exe -Base build\chess.exe -NewOptions 'FpBase=150' -Elo0 -5 -Elo1 0 -Book books\UHO_Lichess_4852_v1.epd
#>
param(
    [Parameter(Mandatory)] [string] $New,
    [Parameter(Mandatory)] [string] $Base,
    # Opening positions, .epd or .pgn. Without a book both engines would
    # replay the same few games from the start position.
    [Parameter(Mandatory)] [string] $Book,
    [string] $Fastchess = 'fastchess',
    # Time control: seconds + increment per move.
    [string] $TC = '8+0.08',
    [int] $Hash = 16,
    [int] $Concurrency = [Math]::Max(1, [Environment]::ProcessorCount - 2),
    [double] $Elo0 = 0,
    [double] $Elo1 = 5,
    # Cap in game pairs: 20000 pairs = 40000 games, about 10 hours at 24
    # concurrent games. A clear result stops the SPRT long before. A change
    # still undecided at the cap is somewhere around 0-5 Elo, and whether
    # to keep it is a judgement call (its Elo estimate, and what it costs
    # in complexity).
    [int] $Rounds = 20000,
    [string[]] $NewOptions = @(),
    [string[]] $BaseOptions = @(),
    [string] $Pgn = 'sprt.pgn'
)

$ErrorActionPreference = 'Stop'

foreach ($exe in @($New, $Base, $Book)) {
    if (-not (Test-Path $exe)) { throw "Not found: $exe" }
}
if (-not (Get-Command $Fastchess -ErrorAction SilentlyContinue)) {
    throw "fastchess not found ('$Fastchess'). Put it on PATH or pass -Fastchess <path>."
}

# fastchess silently drops options an engine doesn't advertise, which
# would turn the test into a long match between identical engines.
function Assert-Options([string] $exe, [string[]] $options) {
    if (-not $options) { return }
    $advertised = (@('uci', 'quit') | & (Resolve-Path $exe).Path) |
        Where-Object { $_ -match '^option name (\S+)' } | ForEach-Object { $Matches[1] }
    foreach ($o in $options) {
        $name = ($o -split '=', 2)[0]
        if ($advertised -notcontains $name) {
            throw "$exe doesn't advertise option '$name'. Use a build configured with -DTUNE=ON."
        }
    }
}
Assert-Options $New $NewOptions
Assert-Options $Base $BaseOptions

$format = if ($Book -match '\.pgn$') { 'pgn' } else { 'epd' }

function Engine-Args([string] $cmd, [string] $name, [string[]] $options) {
    $a = @('-engine', "cmd=$((Resolve-Path $cmd).Path)", "name=$name")
    foreach ($o in $options) { $a += "option.$o" }
    return $a
}

$fcArgs = @()
$fcArgs += Engine-Args $New 'new' $NewOptions
$fcArgs += Engine-Args $Base 'base' $BaseOptions
$fcArgs += @(
    '-each', "tc=$TC", "option.Hash=$Hash",
    '-openings', "file=$((Resolve-Path $Book).Path)", "format=$format", 'order=random',
    '-rounds', $Rounds, '-games', 2, '-repeat',
    '-concurrency', $Concurrency,
    '-sprt', "elo0=$Elo0", "elo1=$Elo1", 'alpha=0.05', 'beta=0.05', 'model=normalized',
    # Adjudicate dead draws and lost positions to save time; both are
    # standard OpenBench settings.
    '-draw', 'movenumber=40', 'movecount=8', 'score=10',
    '-resign', 'movecount=3', 'score=600',
    '-pgnout', "file=$Pgn",
    # Match state, saved as the match runs; an interrupted match resumes
    # with: fastchess -config file=<this file>
    '-config', "outname=$([IO.Path]::ChangeExtension($Pgn, '.json'))",
    '-recover'
)

Write-Host "$Fastchess $($fcArgs -join ' ')"
# fastchess writes warnings to stderr; PowerShell 5.1 would treat them as
# terminating errors under "Stop".
$ErrorActionPreference = 'Continue'
& $Fastchess @fcArgs
