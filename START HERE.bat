@echo off
title ML Workshop - Lesson Launcher
cd /d "%~dp0"

rem No hard-coded paths: prefer the `py` launcher, fall back to `python`.
set "PYCMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYCMD=py"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 set "PYCMD=python"
)
if not defined PYCMD (
    echo ERROR: no Python found. Install Python 3.12+ from https://python.org
    echo and re-run this file.
    pause
    exit /b 1
)

echo ============================================================
echo    ML WORKSHOP - starting the lesson notebook...
echo.
echo    First run sets up everything automatically (Python
echo    packages, the right PyTorch build for your GPU) - this
echo    can take several minutes. Later runs start in seconds.
echo.
echo    1. Wait for your browser to open by itself
echo    2. In the browser left panel, open:
echo       lessons  -  01_foundations  -  lesson1_tiny_net...
echo    3. KEEP THIS BLACK WINDOW OPEN while you work
echo    4. When done: close the browser tab, then close this window
echo ============================================================
echo.

rem bootstrap finds/installs uv, detects the GPU, syncs dependencies
%PYCMD% tools\bootstrap.py || (
    echo.
    echo Bootstrap failed - see the messages above.
    pause
    exit /b 1
)

rem after bootstrap, uv is guaranteed: on PATH or in a well-known dir
set "UV=uv"
where uv >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
    if exist "%LOCALAPPDATA%\hermes\bin\uv.exe" set "UV=%LOCALAPPDATA%\hermes\bin\uv.exe"
)

"%UV%" run jupyter lab

echo.
echo Jupyter has stopped.
pause
