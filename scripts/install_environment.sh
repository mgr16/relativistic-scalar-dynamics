#!/bin/bash
# Script de instalación automática para el entorno PSYOP
# Automated installation script for PSYOP environment
# 
# Este script crea un entorno conda con todas las dependencias necesarias
# This script creates a conda environment with all necessary dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="psyop-dolfinx"
PYTHON_VERSION="3.10"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PSYOP - Installation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda not found!${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}WARNING: Environment '${ENV_NAME}' already exists${NC}"
    read -p "Do you want to remove it and reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
    else
        echo -e "${YELLOW}Skipping environment creation. Activating existing environment...${NC}"
        echo ""
        echo -e "${GREEN}To activate the environment, run:${NC}"
        echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
        exit 0
    fi
fi

echo -e "${BLUE}Step 1/4: Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}...${NC}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
echo -e "${GREEN}✓ Environment created${NC}"
echo ""

echo -e "${BLUE}Step 2/4: Installing DOLFINx (this may take a few minutes)...${NC}"
conda install -n ${ENV_NAME} -c conda-forge dolfinx -y
echo -e "${GREEN}✓ DOLFINx installed${NC}"
echo ""

echo -e "${BLUE}Step 3/4: Installing additional dependencies...${NC}"
conda install -n ${ENV_NAME} -c conda-forge \
    gmsh \
    numpy \
    scipy \
    matplotlib \
    pytest \
    pytest-cov \
    pyyaml \
    -y
echo -e "${GREEN}✓ Additional dependencies installed${NC}"
echo ""

echo -e "${BLUE}Step 4/4: Installing PSYOP package in development mode...${NC}"
# Activate environment and install package
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
pip install -e .
echo -e "${GREEN}✓ PSYOP package installed${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}To use PSYOP:${NC}"
echo -e "  1. Activate the environment: ${BLUE}conda activate ${ENV_NAME}${NC}"
echo -e "  2. Run a simulation: ${BLUE}python main.py --config config_example.json --output results${NC}"
echo -e "  3. Run tests: ${BLUE}pytest${NC}"
echo ""
echo -e "${YELLOW}To verify the installation:${NC}"
echo -e "  ${BLUE}python tests/test_packaging_layout.py${NC}"
echo ""
