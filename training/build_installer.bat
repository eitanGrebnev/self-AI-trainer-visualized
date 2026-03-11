@echo off
setlocal

set ROOT=%~dp0
set EXE=%ROOT%dist\UnslothTrainerGUI.exe

if not exist "%EXE%" (
  echo Missing GUI executable: %EXE%
  echo Build it first with build_gui_exe.bat
  exit /b 1
)

where iscc >nul 2>&1
if %errorlevel% neq 0 (
  echo Inno Setup compiler (iscc) not found in PATH.
  echo Install Inno Setup and add ISCC to PATH, then rerun.
  exit /b 1
)

iscc "%ROOT%installer.iss"
if %errorlevel% neq 0 (
  echo Installer build failed.
  exit /b 1
)

echo.
echo Installer build complete.
echo Check output in this folder for UnslothTrainerStudioSetup.exe
endlocal
