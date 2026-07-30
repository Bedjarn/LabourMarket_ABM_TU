@echo off
setlocal EnableExtensions

rem Path-independent Windows launcher for the Streamlit dashboard.
set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE="

rem Prefer a project environment, then active/common Conda installations.
call :try_python "%PROJECT_ROOT%.venv\Scripts\python.exe"
if defined CONDA_PREFIX call :try_python "%CONDA_PREFIX%\python.exe"
call :try_python "%USERPROFILE%\anaconda3\python.exe"
call :try_python "%USERPROFILE%\miniconda3\python.exe"
call :try_python "%ProgramData%\anaconda3\python.exe"
call :try_python "%ProgramData%\miniconda3\python.exe"

rem Also support regular per-user Python installations and the Python Launcher.
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :try_python "%%~fD\python.exe"
for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do call :try_python "%%P"
for /f "delims=" %%P in ('where python.exe 2^>nul') do call :try_python "%%P"

if not defined PYTHON_EXE (
    echo.
    echo ERROR: Python 3.10 or newer was not found.
    echo Install Python from https://www.python.org/downloads/ and enable
    echo "Add python.exe to PATH" during installation. Then run this file again.
    echo.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -c "import streamlit, mesa, numpy, pandas, matplotlib, openpyxl" >nul 2>&1
if errorlevel 1 (
    echo Required Python packages are missing.
    choice /C YN /N /M "Install them now from requirements.txt? [Y/N] "
    if errorlevel 2 (
        echo Run this command when you are ready:
        echo "%PYTHON_EXE%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
        pause
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
    if errorlevel 1 (
        echo.
        echo ERROR: Installing the required packages failed.
        pause
        exit /b 1
    )
)

if /I "%~1"=="--check" (
    echo Dashboard dependencies are available.
    exit /b 0
)

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "%PROJECT_ROOT%model\run_live_dashboard.py"
if errorlevel 1 (
    echo.
    echo ERROR: The dashboard stopped with an error.
    pause
    exit /b 1
)
exit /b 0

:try_python
if defined PYTHON_EXE exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0
