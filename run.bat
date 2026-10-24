@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m dima_auto_editor %*
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m dima_auto_editor %*
    exit /b %ERRORLEVEL%
)

echo.
echo Dima Auto Editor is not installed yet: the .venv folder was not found.
echo.
echo Install it once with:
echo     powershell -ExecutionPolicy Bypass -File install.ps1
echo.
echo Then start the app again with run.bat
echo.
pause
exit /b 1
