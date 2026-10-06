@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --distpath . --workpath packaging\build packaging\RamanLauncher.spec
exit /b %errorlevel%
