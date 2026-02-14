#!/bin/bash
#
# Script de instalación automatizada para PSYOP
# Configura el entorno conda con DOLFINx y todas las dependencias necesarias
#
# Uso: bash setup_environment.sh
#

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Nombre del entorno
ENV_NAME="psyop-dolfinx"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PSYOP - Script de Instalación${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Verificar que conda está instalado
echo -e "${YELLOW}[1/6] Verificando instalación de conda...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda no está instalado.${NC}"
    echo "Por favor instala Anaconda o Miniconda desde:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓ Conda encontrado: $(conda --version)${NC}"
echo ""

# Verificar si el entorno ya existe
echo -e "${YELLOW}[2/6] Verificando entorno existente...${NC}"
if conda env list | grep -qw "^${ENV_NAME}"; then
    echo -e "${YELLOW}⚠ El entorno '${ENV_NAME}' ya existe.${NC}"
    read -p "¿Desea eliminarlo y crear uno nuevo? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${YELLOW}Eliminando entorno existente...${NC}"
        conda env remove -n ${ENV_NAME} -y
        echo -e "${GREEN}✓ Entorno eliminado${NC}"
    else
        echo -e "${YELLOW}Usando entorno existente.${NC}"
        echo -e "${YELLOW}Si hay problemas, ejecuta: conda env remove -n ${ENV_NAME}${NC}"
    fi
fi
echo ""

# Crear el entorno conda
echo -e "${YELLOW}[3/6] Creando entorno conda '${ENV_NAME}'...${NC}"
if ! conda env list | grep -qw "^${ENV_NAME}"; then
    echo "Esto puede tomar varios minutos..."
    conda create -n ${ENV_NAME} python=3.10 -y
    echo -e "${GREEN}✓ Entorno creado${NC}"
else
    echo -e "${GREEN}✓ Entorno ya existe${NC}"
fi
echo ""

# Activar el entorno y obtener el path de conda
echo -e "${YELLOW}[4/6] Instalando DOLFINx y dependencias...${NC}"
echo "Esto puede tomar varios minutos..."

# Obtener el conda base y usar conda run para ejecutar comandos
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activar el entorno
conda activate ${ENV_NAME}

# Instalar DOLFINx y dependencias principales
echo -e "${BLUE}Instalando DOLFINx (incluye mpi4py y petsc4py)...${NC}"
conda install -c conda-forge dolfinx -y

# Instalar dependencias adicionales
echo -e "${BLUE}Instalando dependencias adicionales...${NC}"
conda install -c conda-forge gmsh numpy scipy matplotlib pytest pytest-cov pyyaml -y

echo -e "${GREEN}✓ Dependencias instaladas${NC}"
echo ""

# Instalar el paquete en modo desarrollo
echo -e "${YELLOW}[5/6] Instalando paquete psyop en modo desarrollo...${NC}"
pip install -e .
echo -e "${GREEN}✓ Paquete psyop instalado${NC}"
echo ""

# Verificar la instalación
echo -e "${YELLOW}[6/6] Verificando instalación...${NC}"
echo ""

# Verificar imports críticos
python -c "
import sys
try:
    import dolfinx
    print('✓ DOLFINx importado correctamente')
except ImportError as e:
    print('✗ Error importando DOLFINx:', str(e))
    sys.exit(1)

try:
    import gmsh
    print('✓ Gmsh importado correctamente')
except ImportError as e:
    print('⚠ Gmsh no disponible (opcional):', str(e))

try:
    import psyop
    print('✓ Paquete psyop importado correctamente')
except ImportError as e:
    print('✗ Error importando psyop:', str(e))
    sys.exit(1)

print('')
print('Versión DOLFINx:', dolfinx.__version__)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ¡Instalación completada con éxito!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Para usar el entorno, ejecuta:${NC}"
    echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
    echo ""
    echo -e "${BLUE}Para ejecutar el programa:${NC}"
    echo -e "  ${YELLOW}python main.py --config config_example.json --output results${NC}"
    echo ""
    echo -e "${BLUE}Para ejecutar los tests:${NC}"
    echo -e "  ${YELLOW}pytest tests/ -v${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Error en la verificación${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Por favor revisa los errores arriba."
    exit 1
fi
