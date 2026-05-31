#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="psyop-dolfinx"
PYTHON_VERSION="3.10"
INSTALL_DEV="false"
SKIP_PIP="false"
YES_FLAG=""

print_help() {
  cat <<'EOF'
Uso:
  scripts/setup_conda_env.sh [opciones]

Opciones:
  --env-name <nombre>        Nombre del entorno conda (default: psyop-dolfinx)
  --python <version>         Versión de Python (default: 3.10)
  --install-dev              Instala extras de desarrollo (.[dev])
  --skip-pip-install         No ejecuta pip install -e .
  --yes                      Modo no interactivo para conda
  -h, --help                 Muestra esta ayuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --install-dev)
      INSTALL_DEV="true"
      shift
      ;;
    --skip-pip-install)
      SKIP_PIP="true"
      shift
      ;;
    --yes)
      YES_FLAG="-y"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Opción no reconocida: $1"
      print_help
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda no está disponible en PATH."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/5] Verificando/creando entorno '$ENV_NAME' con Python $PYTHON_VERSION..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "El entorno '$ENV_NAME' ya existe."
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" $YES_FLAG
fi

echo "[2/5] Instalando dependencias base desde conda-forge..."
conda install -n "$ENV_NAME" --override-channels -c conda-forge $YES_FLAG \
  fenics-dolfinx gmsh numpy scipy matplotlib pyyaml pytest mpi4py petsc4py

echo "[3/5] Actualizando pip/setuptools/wheel en el entorno..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_PIP" == "false" ]]; then
  echo "[4/5] Instalando paquete local en modo editable..."
  if [[ "$INSTALL_DEV" == "true" ]]; then
    conda run -n "$ENV_NAME" python -m pip install -e "$ROOT_DIR[dev]"
  else
    conda run -n "$ENV_NAME" python -m pip install -e "$ROOT_DIR"
  fi
else
  echo "[4/5] Saltando instalación editable por --skip-pip-install."
fi

echo "[5/5] Verificando imports críticos..."
conda run -n "$ENV_NAME" python -c "import dolfinx, gmsh, mpi4py, petsc4py, numpy, scipy, yaml; print('OK: imports críticos disponibles')"

echo
echo "Entorno listo."
echo "Para activarlo: conda activate $ENV_NAME"
echo "Para ejecutar:  psyop run --config config_example.json --output results"
