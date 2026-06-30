@echo off
title RAG System - Clean Start
color 0A
echo ============================================================
echo  RAG System - Clean ^& Rebuild
echo ============================================================
echo.

cd /d "%~dp0"

REM --- Paths ---
set VENV=.venv
set PIP=%VENV%\Scripts\pip.exe
set UV=%VENV%\Scripts\uvicorn.exe

REM --- Check Python dynamically ---
set PYTHON=python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" (
        set PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe
    ) else (
        echo [ERROR] Python not found in system PATH or default AppData path.
        echo Please install Python 3.10 and check the environment.
        pause & exit /b 1
    )
)
echo [OK] Python: %PYTHON%

REM --- Clean __pycache__ ---
echo.
echo [CLEAN] Removing __pycache__ ...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q "*.pyc" 2>nul
echo [OK] Cleaned.

REM --- Setup venv ---
echo.
if not exist "%VENV%\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    "%PYTHON%" -m venv "%VENV%"
    if %ERRORLEVEL% NEQ 0 (echo [ERROR] venv creation failed & pause & exit /b 1)
    echo [OK] venv created.
) else (
    echo [OK] venv exists.
)

REM --- Upgrade pip ---
echo.
echo [INSTALL] Upgrading pip...
"%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip

REM --- Install all dependencies ---
echo [INSTALL] Installing and upgrading packages to latest version...
"%PIP%" install --upgrade ^
    "fastapi" ^
    "uvicorn[standard]" ^
    "python-multipart" ^
    "langchain" ^
    "langchain-core" ^
    "langchain-community" ^
    "langchain-openai" ^
    "langchain-huggingface" ^
    "langchain-chroma" ^
    "pydantic" ^
    "pydantic-settings" ^
    "python-dotenv" ^
    "chromadb" ^
    "pymupdf" ^
    "sentence-transformers" ^
    "openai" ^
    "redis" ^
    "python-jose[cryptography]" ^
    "passlib[bcrypt]" ^
    "httpx" ^
    "prometheus-client" ^
    "opentelemetry-api" ^
    "opentelemetry-sdk"


if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package installation failed!
    pause & exit /b 1
)
echo [OK] All packages installed.

REM --- Create data dir ---
if not exist "data\chroma" mkdir "data\chroma"

REM --- Launch ---
echo.
echo ============================================================
echo  Starting RAG API Server
echo  Frontend : http://localhost:8080/
echo  Swagger  : http://localhost:8080/docs
echo  API token: dev-token
echo ============================================================
echo.
echo Press Ctrl+C to stop.
echo.

set PYTHONPATH=%~dp0src
"%UV%" rag_system.api.app:app --host 0.0.0.0 --port 8080 --reload

pause
