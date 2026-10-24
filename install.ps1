[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$SkipRife
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$minimumVersion = [version]'3.10'
$rifeRelease = '20221029'
$rifeUrl = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/$rifeRelease/rife-ncnn-vulkan-$rifeRelease-windows.zip"
$rifeDir = Join-Path $root 'tools\rife'

function Write-Step([string]$message) {
    Write-Host ''
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Fail([string]$message) {
    Write-Host ''
    Write-Host $message -ForegroundColor Red
    exit 1
}

function Get-PythonVersion([string[]]$command) {
    try {
        $exe = $command[0]
        $prefix = @($command | Select-Object -Skip 1)
        $output = & $exe @prefix -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) { return [version]($output | Select-Object -Last 1).Trim() }
    }
    catch { }
    return $null
}

function Find-Python {
    foreach ($candidate in @(@('py', '-3'), @('python'), @('python3'))) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $version = Get-PythonVersion $candidate
        if ($version) { return @{ Command = $candidate; Version = $version } }
    }
    return $null
}

function Invoke-Checked([string]$description, [scriptblock]$action) {
    & $action
    if ($LASTEXITCODE -ne 0) { Fail "$description failed (exit code $LASTEXITCODE)." }
}

Set-Location -LiteralPath $root

Write-Step 'Looking for Python 3.10 or newer'
$python = Find-Python
if (-not $python) {
    Fail ("Python was not found.`n" +
          "Install Python 3.11+ from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')`n" +
          "or run:  winget install Python.Python.3.12`n" +
          "Then open a new PowerShell window and run this script again.")
}
if ($python.Version -lt $minimumVersion) {
    Fail ("Python $($python.Version) is too old; version $minimumVersion or newer is required.`n" +
          "Install a newer one with:  winget install Python.Python.3.12")
}
Write-Host "Found Python $($python.Version) ($($python.Command -join ' '))"

if (Test-Path -LiteralPath $venvPython) {
    Write-Step 'Virtual environment .venv already exists - reusing it'
}
else {
    Write-Step 'Creating virtual environment .venv'
    $exe = $python.Command[0]
    $prefix = @($python.Command | Select-Object -Skip 1)
    Invoke-Checked 'Creating the virtual environment' { & $exe @prefix -m venv (Join-Path $root '.venv') }
}

Write-Step 'Upgrading pip'
Invoke-Checked 'Upgrading pip' { & $venvPython -m pip install --upgrade pip }

Write-Step 'Installing requirements.txt'
Invoke-Checked 'Installing requirements' { & $venvPython -m pip install -r (Join-Path $root 'requirements.txt') }

if ($Dev) {
    Write-Step 'Installing development requirements (pytest)'
    Invoke-Checked 'Installing development requirements' { & $venvPython -m pip install -r (Join-Path $root 'requirements-dev.txt') }
}

Write-Step 'Checking FFmpeg'
$systemFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($systemFfmpeg) {
    Write-Host "System FFmpeg found: $($systemFfmpeg.Source)"
}
else {
    Write-Host 'No system FFmpeg on PATH. That is fine: the bundled imageio-ffmpeg binary works out of the box.'
    Write-Host 'Optional system-wide install:  winget install Gyan.FFmpeg' -ForegroundColor DarkGray
}
$bundled = $null
try { $bundled = & $venvPython -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>$null }
catch { }
if ($LASTEXITCODE -eq 0 -and $bundled) {
    Write-Host "Bundled FFmpeg: $bundled"
}
elseif (-not $systemFfmpeg) {
    Fail 'No usable FFmpeg was found. Run:  winget install Gyan.FFmpeg  and then this script again.'
}

function Install-Rife {
    $rifeExe = Join-Path $rifeDir 'rife-ncnn-vulkan.exe'
    if (Test-Path -LiteralPath $rifeExe) {
        Write-Host "RIFE already installed: $rifeExe"
        return
    }
    $download = Join-Path ([System.IO.Path]::GetTempPath()) ("rife_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $download | Out-Null
    try {
        $zip = Join-Path $download 'rife.zip'
        Write-Host "Downloading RIFE (about 430 MB, only the needed 18 MB are kept)..."
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $rifeUrl -OutFile $zip -UseBasicParsing
        Expand-Archive -LiteralPath $zip -DestinationPath $download
        $unpacked = Join-Path $download "rife-ncnn-vulkan-$rifeRelease-windows"
        New-Item -ItemType Directory -Force -Path $rifeDir | Out-Null
        foreach ($item in @('rife-ncnn-vulkan.exe', 'vcomp140.dll', 'LICENSE', 'rife-v4.6')) {
            Copy-Item -LiteralPath (Join-Path $unpacked $item) -Destination $rifeDir -Recurse -Force
        }
        Write-Host "RIFE installed: $rifeExe"
    }
    catch {
        Write-Host "Could not install RIFE ($($_.Exception.Message)). The app still works and uses FFmpeg for frame generation." -ForegroundColor Yellow
    }
    finally {
        Remove-Item -LiteralPath $download -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($SkipRife) {
    Write-Step 'Skipping RIFE (frame generation falls back to FFmpeg)'
}
else {
    Write-Step 'Installing RIFE neural frame generation (optical flow)'
    Install-Rife
}

Write-Step 'Verifying the installation'
Invoke-Checked 'Importing the application' { & $venvPython -c "import PySide6, numpy, dima_auto_editor; print('OK: PySide6', PySide6.__version__, '| numpy', numpy.__version__)" }

Write-Host ''
Write-Host 'Installation complete. Start the app by double-clicking run.bat' -ForegroundColor Green
Write-Host '(or run:  powershell -ExecutionPolicy Bypass -File run.ps1)'
