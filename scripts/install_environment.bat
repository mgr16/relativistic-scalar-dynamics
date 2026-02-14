@echo off
REM Script de instalación automática para el entorno PSYOP (Windows)
REM Automated installation script for PSYOP environment (Windows)
REM 
REM Este script crea un entorno conda con todas las dependencias necesarias
REM This script creates a conda environment with all necessary dependencies

setlocal enabledelayedexpansion

REM Configuration
set ENV_NAME=psyop-dolfinx
set PYTHON_VERSION=3.10

echo ========================================
echo PSYOP - Installation Script (Windows)
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda not found!
    echo Please install Miniconda or Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('conda --version') do set CONDA_VERSION=%%i
echo [OK] Conda found: %CONDA_VERSION%
echo.

REM Check if environment already exists
conda env list | findstr /b "%ENV_NAME% " >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo WARNING: Environment '%ENV_NAME%' already exists
    set /p REMOVE="Do you want to remove it and reinstall? (y/N): "
    if /i "!REMOVE!"=="y" (
        echo Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo Skipping environment creation. Activating existing environment...
        echo.
        echo To activate the environment, run:
        echo   conda activate %ENV_NAME%
        pause
        exit /b 0
    )
)

echo Step 1/4: Creating conda environment '%ENV_NAME%' with Python %PYTHON_VERSION%...
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)
echo [OK] Environment created
echo.

echo Step 2/4: Installing DOLFINx (this may take a few minutes)...
conda install -n %ENV_NAME% -c conda-forge dolfinx -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install DOLFINx
    pause
    exit /b 1
)
echo [OK] DOLFINx installed
echo.

echo Step 3/4: Installing additional dependencies...
conda install -n %ENV_NAME% -c conda-forge gmsh numpy scipy matplotlib pytest pytest-cov pyyaml -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Additional dependencies installed
echo.

echo Step 4/4: Installing PSYOP package in development mode...
call conda activate %ENV_NAME%
pip install -e .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PSYOP package
    pause
    exit /b 1
)
echo [OK] PSYOP package installed
echo.

echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To use PSYOP:
echo   1. Activate the environment: conda activate %ENV_NAME%
echo   2. Run a simulation: python main.py --config config_example.json --output results
echo   3. Run tests: pytest
echo.
echo To verify the installation:
echo   python tests\test_packaging_layout.py
echo.
pause
