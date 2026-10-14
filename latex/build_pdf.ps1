$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$cache = Join-Path $PSScriptRoot '.latex-cache'
New-Item -ItemType Directory -Force $cache | Out-Null
$env:TEXMFCACHE = $cache
$env:TEXMFVAR = $cache
$env:LC_ALL = 'C'
$env:LANG = 'C'
$previousErrorPolicy = $ErrorActionPreference
try {
    # Windows PowerShell treats latexmk's informational stderr as ErrorRecords.
    # Use the process exit code to distinguish a build failure from such messages.
    $ErrorActionPreference = 'Continue'
    if (-not (Test-Path 'figures/framework.pdf') -or (Get-Item 'figures/framework.tex').LastWriteTime -gt (Get-Item 'figures/framework.pdf').LastWriteTime) {
        Push-Location 'figures'
        & lualatex -interaction=nonstopmode -halt-on-error framework.tex 2>&1 | ForEach-Object { $_.ToString() }
        $figureExitCode = $LASTEXITCODE
        Pop-Location
        if ($figureExitCode -ne 0) { throw "Framework build failed ($figureExitCode)" }
    }
    & latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex 2>&1 | ForEach-Object { $_.ToString() }
    $buildExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorPolicy
}
if ($buildExitCode -ne 0) { throw "LaTeX build failed ($buildExitCode)" }
