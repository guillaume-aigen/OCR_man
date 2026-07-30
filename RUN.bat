@echo off
rem Process everything in INPUT\ and write results to OUTPUT\.
rem Double-click this file, or run it from a terminal with extra options:
rem     RUN.bat --no-llm
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   The environment is not set up yet.
  echo   Open a terminal in this folder and run:
  echo.
  echo       python SETUP.py
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" RUN_ME.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if "%EXITCODE%"=="0" (
  echo   Done. Results are in: %cd%\OUTPUT
) else if "%EXITCODE%"=="4" (
  echo   Nothing was processed. Put your scanned PDFs or EPUBs here first:
  echo   %cd%\INPUT
) else (
  echo   Finished with errors. See %cd%\WORK\ocr_man.log
)
echo.
pause
exit /b %EXITCODE%
