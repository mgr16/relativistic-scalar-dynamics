# Reproducir el manuscrito F3

Este directorio contiene el manuscrito PRD de dos columnas y todos sus
artefactos generados. El orden canónico es:

1. tabla de números;
2. macros LaTeX;
3. figuras;
4. PDF;
5. manifest SHA-256 y tests.

Ningún paso lee `results/` ni regenera artefactos congelados F0--F2. En
particular, **no ejecutar `scripts/interior_production.py`** como parte del
build del paper.

## Entorno probado

- Python 3.10.19
- NumPy 2.2.6
- Matplotlib 3.10.8
- Tectonic 0.16.9
- bundle Tectonic `default_bundle_v33.tar`
- RevTeX 4.2e, resuelto desde ese bundle

La instalación mínima usada fue:

```bash
mamba install -n rsd-dolfinx -c conda-forge tectonic
```

`texlive-core` de conda-forge no es suficiente por sí solo: la variante
probada no incluye las clases LaTeX ni un `tlmgr` funcional. Una distribución
TeX completa con `revtex4-2` puede usar `latexmk`, pero no es el camino
reproducible auditado aquí.

## Regeneración

Desde la raíz del repositorio:

```bash
mamba activate rsd-dolfinx
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rsd-matplotlib-cache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/rsd-xdg-cache"

python scripts/paper_numbers.py
python scripts/paper_tex_numbers.py
python scripts/paper_figures.py
```

La primera máquina necesita poblar el cache de Tectonic una vez:

```bash
cd paper
tectonic -X compile main.tex \
  --bundle https://relay.fullyjustified.net/default_bundle_v33.tar
cd ..
```

Después, el build de release usa sólo ese bundle cacheado y fija la época del
PDF. Los dos builds consecutivos auditados dieron el mismo SHA-256.

```bash
cd paper
SOURCE_DATE_EPOCH=1783814400 tectonic -X compile main.tex \
  --bundle https://relay.fullyjustified.net/default_bundle_v33.tar \
  --only-cached
cd ..
```

El resultado esperado es `paper/main.pdf`: 7 páginas letter, PDF 1.5. La
fecha visible del manuscrito está fijada en `main.tex`; `SOURCE_DATE_EPOCH`
fija los metadatos internos del PDF.

## Verificación

No regenerar el manifest para ocultar un drift. Primero ejecutar todos los
checks; actualizarlo sólo cuando el cambio sea intencional y revisado.

```bash
python scripts/paper_numbers.py --check
python scripts/paper_tex_numbers.py --check
python scripts/paper_figures.py --check
python scripts/paper_manifest.py --check
python -m pytest -m "not slow"
git diff --check
```

El cierre C5 espera 257 entradas canónicas, 97 macros, 5 figuras en PDF+PNG,
un PDF de 7 páginas, 32 artefactos en el manifest y 216 tests rápidos (más 7
slow deseleccionados). El manifest lista la clausura de fuentes y outputs con
rutas relativas y hashes SHA-256; no contiene timestamps ni rutas del host.

## Cambios de metadata y release

La byline se tomó del nombre Git verificable. Afiliación, ORCID y email se
omitieron porque no había una fuente local fiable. Si se agregan antes del
depósito, hay que recompilar `main.pdf`, repetir el QA visual y actualizar el
manifest.

La propuesta de tag es `v3.3.0-paper`. Crear el tag, cambiar la versión del
paquete y depositar en arXiv/Zenodo u otro servicio son acciones reservadas a
Marco y no forman parte de estos scripts.
