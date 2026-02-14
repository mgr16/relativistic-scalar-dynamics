# Guía Rápida de Instalación - PSYOP

## 📋 Requisitos Previos

Antes de instalar PSYOP, necesitas tener **Conda** instalado en tu sistema:

- **Miniconda** (recomendado, ligero): https://docs.conda.io/en/latest/miniconda.html
- **Anaconda** (completo): https://www.anaconda.com/download

### Verificar si Conda está instalado

```bash
conda --version
```

Si ves un número de versión, ¡estás listo para continuar!

## 🚀 Instalación en 1 Paso

### Linux / macOS

```bash
bash scripts/install_environment.sh
```

### Windows

```batch
scripts\install_environment.bat
```

## ✅ ¿Qué hace el script?

El script automáticamente:

1. ✅ Crea un entorno conda llamado `psyop-dolfinx` con Python 3.10
2. ✅ Instala **DOLFINx** (la dependencia crítica para elementos finitos)
3. ✅ Instala todas las dependencias: mpi4py, petsc4py, gmsh, numpy, scipy, matplotlib, pytest, PyYAML
4. ✅ Instala el paquete PSYOP en modo desarrollo

**Tiempo estimado**: 5-15 minutos (dependiendo de tu conexión a internet)

## 🎯 Después de la Instalación

### 1. Activar el entorno

```bash
conda activate psyop-dolfinx
```

### 2. Verificar la instalación

```bash
python tests/test_packaging_layout.py
```

Si ves mensajes de éxito (✓), ¡la instalación fue correcta!

### 3. Ejecutar tu primera simulación

```bash
python main.py --config config_example.json --output results
```

## 📚 Recursos Adicionales

- **README completo**: Consulta `README.md` para detalles técnicos completos
- **Documentación**: Carpeta `docs/` contiene documentación detallada
- **Ejemplos**: Archivo `config_example.json` muestra configuraciones de ejemplo

## 🆘 Problemas Comunes

### "conda: command not found"

**Solución**: Necesitas instalar Conda primero
- Descarga Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Sigue las instrucciones de instalación para tu sistema operativo

### "Environment already exists"

**Solución**: Ya tienes un entorno con ese nombre

```bash
# Opción 1: Eliminar y reinstalar
conda env remove -n psyop-dolfinx
bash scripts/install_environment.sh

# Opción 2: Usar el entorno existente
conda activate psyop-dolfinx
pip install -e .
```

### Error durante la instalación

**Solución**: Revisa la sección "Solución de Problemas" en el README.md

## 💡 Comandos Útiles

```bash
# Ver todos los entornos conda
conda env list

# Activar el entorno PSYOP
conda activate psyop-dolfinx

# Desactivar el entorno
conda deactivate

# Ejecutar tests
pytest

# Ver ayuda del programa principal
python main.py --help
```

## 📞 Soporte

Si encuentras problemas:
1. Consulta la sección "Solución de Problemas" en README.md
2. Revisa los logs del script de instalación
3. Abre un issue en el repositorio con detalles del error

---

**¡Bienvenido a PSYOP!** Estás listo para simular campos escalares en relatividad general. 🚀
