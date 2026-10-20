<#
.SYNOPSIS
Plays our engine against a set of rated opponents with fastchess, for a
rough absolute rating.

.DESCRIPTION
Each opponent plays -Rounds game pairs against the engine (colors
reversed within a pair) from a balanced opening book. Unlike an SPRT,
the result is an absolute estimate, so two things differ from sprt.ps1:
balanced openings (unbalanced books like UHO stretch Elo differences),
and concurrency limited to physical cores, since time lost to
hyper-threading would bias the result rather than cancel out.

Opponent ratings are CCRL Blitz (2'+1" on an i7-4770K). This machine is
faster and the time control differs, so treat the resulting number as
an estimate on that scale, not a CCRL rating.

.EXAMPLE
tools\gauntlet.ps1 -Engine engines\best.exe
#>
param(
    [string] $Engine = 'engines\best.exe',
    # Options for FunComp itself, as "option.Name=value" (e.g. its EvalFile).
    [string[]] $EngineOptions = @(),
    # Opponents as "file=CCRL rating", optionally followed by ";name=..."
    # (the name in the PGN; default the file's base name) and any
    # ";option.Name=value" the opponent needs, e.g. a network file.
    [string[]] $Opponents = @(
        'engines\stash-15.3.exe=2173',
        'engines\stash-17.0.exe=2297',
        'engines\stash-19.0.exe=2473',
        'engines\stash-21.0.exe=2713',
        'engines\stash-25.0.exe=2933'
    ),
    [string] $Book = 'books\8moves_v3.pgn',
    [string] $Fastchess = 'tools\fastchess\fastchess-windows-x86-64\fastchess.exe',
    [string] $TC = '10+0.1',
    [int] $Hash = 512,
    [int] $Concurrency = 12,
    [int] $Rounds = 50,          # game pairs per opponent
    [string] $Pgn = 'matches\gauntlet.pgn'
)

$ErrorActionPreference = 'Stop'

$fcArgs = @('-engine', "cmd=$((Resolve-Path $Engine).Path)", 'name=FunComp') + $EngineOptions
foreach ($o in $Opponents) {
    $parts = $o -split ';'
    $file, $rating = $parts[0] -split '=', 2
    $name = [IO.Path]::GetFileNameWithoutExtension($file)
    $extra = @()
    foreach ($p in $parts[1..($parts.Count)]) {
        if (-not $p) { continue }
        if ($p -like 'name=*') { $name = $p.Substring(5) } else { $extra += $p }
    }
    $fcArgs += @('-engine', "cmd=$((Resolve-Path $file).Path)", "name=$name") + $extra
}
$format = if ($Book -match '\.pgn$') { 'pgn' } else { 'epd' }
$fcArgs += @(
    '-tournament', 'gauntlet', '-seeds', 1,
    '-each', "tc=$TC", "option.Hash=$Hash",
    '-openings', "file=$((Resolve-Path $Book).Path)", "format=$format", 'order=random',
    '-rounds', $Rounds, '-games', 2, '-repeat',
    '-concurrency', $Concurrency,
    '-draw', 'movenumber=40', 'movecount=8', 'score=10',
    '-resign', 'movecount=3', 'score=600',
    '-pgnout', "file=$Pgn",
    '-config', "outname=$([IO.Path]::ChangeExtension($Pgn, '.json'))",
    '-recover'
)

Write-Host "$Fastchess $($fcArgs -join ' ')"
# fastchess writes warnings to stderr; PowerShell 5.1 would treat them as
# terminating errors under "Stop".
$ErrorActionPreference = 'Continue'
& $Fastchess @fcArgs
