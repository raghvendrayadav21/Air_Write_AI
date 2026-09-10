@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  AirWrite AI – Quick Setup & Run Script
REM  Double-click this file OR run it from Command Prompt
REM ============================================================

echo.
echo ===================================================
echo   AIRWRITE AI - Setup ^& Launch
echo   Write in the Air. Recognize with AI.
echo ===================================================
echo.

REM ── Find Python ─────────────────────────────────────────────
set PYTHON_CMD=

REM Check common user install locations first
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    goto :found_python
)

REM Try PATH
for %%P in (python3.11 python3.12 python3.10 python3 python) do (
    where %%P >nul 2>&1
    if !errorlevel! == 0 (
        set PYTHON_CMD=%%P
        goto :found_python
    )
)

echo ERROR: Python 3.10+ not found.
echo.
echo Please install Python from: https://python.org/downloads/
echo Make sure to tick "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
REM ── Check installed packages & directly run ──────────────────
echo [OK] Python found: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2


REM ── Check for trained model ───────────────────────────────────
if not exist "model\airwrite_model.h5" (
    echo ===================================================
    echo   WARNING: Model not found!
    echo   model\airwrite_model.h5 is missing.
    echo ===================================================
    echo.
    echo The app will open but PREDICTION will be disabled
    echo until you train the model.
    echo.
    set /p train_now="Train the model now? (y/n, default=n): "
    if /i "!train_now!" == "y" (
        echo.
        echo ===================================================
        echo   Training CNN model...
        echo   This will take 5-20 minutes.
        echo ===================================================
        echo.
        "%PYTHON_CMD%" train_model.py
        echo.
        if exist "model\airwrite_model.h5" (
            echo [OK] Model trained and saved!
        ) else (
            echo [WARN] Model training may not have completed. Check output above.
        )
        echo.
    )
)

REM ── Launch AirWrite AI ────────────────────────────────────────
echo ===================================================
echo   Launching AirWrite AI...
echo.
echo   Keyboard shortcuts:
echo     C = Clear canvas
echo     P = Predict character
echo     Q = Quit
echo ===================================================
echo.

"%PYTHON_CMD%" main.py

echo.
echo AirWrite AI exited.
pause
