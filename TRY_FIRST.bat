@echo off
rem Quality check: process only the first 20 pages of each document in INPUT\.
rem Run this before committing to a full book so you can look at the output
rem first. Takes about two minutes per document instead of hours.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   The environment is not set up yet. Run:  python SETUP.py
  echo.
  pause
  exit /b 1
)

echo.
echo   Processing the first 20 pages of each document as a quality check.
echo.
".venv\Scripts\python.exe" RUN_ME.py --pages 20 %*
set EXITCODE=%ERRORLEVEL%

if "%EXITCODE%"=="4" (
  echo.
  echo   Nothing was processed. Put your scanned PDFs or EPUBs here first:
  echo   %cd%\INPUT
  echo.
  pause
  exit /b 4
)

echo.
echo   Sample output is in: %cd%\OUTPUT
echo.
echo   Open the .md file and check:
echo     - is the text in the right order (not jumping between columns)?
echo     - are paragraphs whole, not chopped up?
echo     - are page headers/footers gone from the body text?
echo.
echo   If it looks good, just run RUN.bat for the whole thing. It continues
echo   from the pages already done instead of redoing them.
echo.
pause
