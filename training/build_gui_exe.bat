@echo off
setlocal

set ROOT=%~dp0..
set PY=%ROOT%\unsloth_env\Scripts\python.exe

if not exist "%PY%" (
  echo Could not find Python at %PY%
  echo Activate/create the venv first.
  exit /b 1
)

"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install pyinstaller
"%PY%" -m PyInstaller --noconfirm --onefile --windowed --name UnslothTrainerGUI "%~dp0gui_app.py"

echo.
echo Build complete. EXE is in: %~dp0dist\UnslothTrainerGUI.exe
endlocal
