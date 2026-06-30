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
echo [INSTALL] Installing packages (first run may take 3-5 min)...
"%PIP%" install --quiet ^
    "fastapi==0.111.0" ^
    "uvicorn[standard]==0.30.1" ^
    "python-multipart==0.0.9" ^
    "langchain==0.3.25" ^
    "langchain-core==0.3.65" ^
    "langchain-community==0.3.24" ^
    "langchain-openai==0.2.14" ^
    "langchain-huggingface==0.1.2" ^
    "langchain-chroma==0.1.4" ^
    "pydantic==2.11.7" ^
    "pydantic-settings==2.9.1" ^
    "python-dotenv==1.1.0" ^
    "chromadb==0.5.23" ^
    "pymupdf==1.27.2" ^
    "sentence-transformers==3.0.1" ^
    "openai==1.84.0" ^
    "redis==5.2.1" ^
    "python-jose[cryptography]==3.3.0" ^
    "passlib[bcrypt]==1.7.4" ^
    "httpx==0.27.0" ^
    "prometheus-client==0.21.1" ^
    "opentelemetry-api==1.28.0" ^
    "opentelemetry-sdk==1.28.0"

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
