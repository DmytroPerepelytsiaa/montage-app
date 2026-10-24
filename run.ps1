$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$pythonw = Join-Path $root '.venv\Scripts\pythonw.exe'
$python = Join-Path $root '.venv\Scripts\python.exe'
$launchArgs = @('-m', 'dima_auto_editor') + $args

if (Test-Path -LiteralPath $pythonw) {
    Start-Process -FilePath $pythonw -ArgumentList $launchArgs -WorkingDirectory $root
    exit 0
}

if (Test-Path -LiteralPath $python) {
    Push-Location -LiteralPath $root
    try {
        & $python @launchArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

Write-Host ''
Write-Host 'Dima Auto Editor is not installed yet: the .venv folder was not found.' -ForegroundColor Yellow
Write-Host 'Install it once with:'
Write-Host '    powershell -ExecutionPolicy Bypass -File install.ps1' -ForegroundColor Cyan
Write-Host 'Then start the app with run.bat or run.ps1.'
exit 1
