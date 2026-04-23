# Auditoría rápida del proyecto (2026-04-23)

## Alcance

Se realizó una revisión técnica de estructura, documentación, empaquetado y capacidad de ejecución básica del repositorio, con foco en:

- Instalación local y dependencias
- Salud de pruebas automáticas
- Coherencia entre documentación y código
- Riesgos técnicos inmediatos

## Comprobaciones ejecutadas

1. `pytest -q`
2. `python -m pip install -r requirements.txt`
3. `pytest -q tests/test_basic_python.py tests/test_packaging_layout.py tests/test_structure.py tests/test_basic.py tests/test_main_simple.py tests/test_standalone_logic.py`
4. `python -m compileall -q src main.py scripts`

## Hallazgos

### 1) Dependencia fuerte de NumPy para colección de pruebas

La colección de pruebas falla si `numpy` no está disponible en el entorno. Esto afecta múltiples módulos de tests durante la fase de importación.

**Impacto:** no se puede ejecutar la suite completa en entornos restringidos sin acceso a paquetes binarios.

**Recomendación:** separar pruebas "core sin FEM" y pruebas numéricas con marcador/skip condicional más explícito para degradación controlada.

### 2) Instalación bloqueada por conectividad/proxy del entorno

La instalación de `requirements.txt` no fue posible por errores de conexión al índice de paquetes (`ProxyError`), por lo que no se pudieron validar rutas dependientes de `numpy/scipy`.

**Impacto:** limita la validación en CI/entornos cerrados si no existe caché de wheels o mirror interno.

**Recomendación:** documentar instalación offline/air-gapped (wheelhouse, mirror privado o lockfile reproducible).

### 3) Suite mínima de estructura sí pasa

Las pruebas de layout/paquete (`test_packaging_layout.py`) pasaron correctamente, y la compilación sintáctica del código (`compileall`) también.

**Impacto:** la base de empaquetado/importación está razonablemente estable.

**Recomendación:** ampliar estas pruebas "livianas" para cubrir CLI y carga de configuración en modo sin solver.

### 4) Inconsistencia menor en tests

`tests/test_structure.py` existe pero está vacío.

**Impacto:** ruido en mantenimiento; sugiere deuda técnica menor.

**Recomendación:** eliminar archivo vacío o poblarlo con checks estructurales reales.

## Estado general

**Resultado de la revisión:** estado **aceptable** para estructura de paquete y sintaxis, con **riesgo operativo medio** en reproducibilidad de entorno debido a dependencias científicas pesadas y falta de estrategia offline documentada.

## Prioridades sugeridas (ordenadas)

1. Definir estrategia reproducible de instalación en entornos restringidos (documentación + lock/wheels).
2. Etiquetar/segmentar explícitamente tests por nivel de dependencia (core, numpy/scipy, dolfinx).
3. Añadir smoke test de CLI desacoplado del solver pesado.
4. Limpiar artefactos de tests vacíos (`test_structure.py`).

## ¿Cómo mejorarlo? (científico + computacional)

### A) Mejoras científicas (física numérica)

1. **Validación por soluciones manufacturadas (MMS)**
   - Construir 2-3 casos analíticos para (φ, Π) en fondo plano y medir convergencia de orden.
   - Reportar errores L2/L∞ por refinamiento en espacio y tiempo.
   - **Meta:** observar pendiente cercana al orden esperado del esquema (espacio/tiempo).

2. **Banco de pruebas de conservación y balance**
   - Integrar energía total y flujo por frontera para verificar balance:
     - `dE/dt + Flux_boundary ≈ 0` (según condiciones físicas configuradas).
   - Guardar presupuesto de energía por paso para detectar inestabilidades tempranas.
   - **Meta:** error relativo acumulado por debajo de un umbral definido por resolución.

3. **Análisis sistemático de reflexión en frontera**
   - Barrer radio del dominio, resolución y variantes de BC de salida.
   - Medir coeficiente de reflexión efectivo con una señal de prueba estandarizada.
   - **Meta:** curva reflexión vs. frecuencia para justificar configuración por defecto.

4. **Calibración QNM con referencias**
   - Comparar frecuencias y tasas de amortiguamiento contra benchmarks publicados.
   - Separar incertidumbre numérica (malla, dt, ventana FFT/Prony) de la física.
   - **Meta:** tabla de error relativo de QNM para 1-2 escenarios canónicos.

5. **Estudios de sensibilidad paramétrica**
   - Diseñar barridos para `m_squared`, `lambda_coupling`, amplitud y ancho inicial.
   - Priorizar diseño de experimentos reproducible (semillas, config versionada).
   - **Meta:** mapas de respuesta física con intervalos de confianza numérica.

### B) Mejoras computacionales (software/arquitectura)

1. **Matriz de tests por niveles**
   - Definir suites: `unit-core`, `unit-numpy`, `integration-dolfinx`, `slow-hpc`.
   - Añadir `pytest.importorskip(...)` o marcadores para fallar con mensajes claros.
   - **Meta:** CI rápida (<5 min) para core + pipeline extendido para HPC.

2. **Entorno reproducible**
   - Publicar lockfiles/ambientes (`conda-lock` o equivalente) y guía offline.
   - Mantener imagen base de CI con dependencias científicas preinstaladas.
   - **Meta:** “time-to-first-run” predecible en entornos con y sin internet.

3. **Observabilidad y trazabilidad de corridas**
   - Estandarizar `manifest.json` con versión de config, hash de malla y parámetros efectivos.
   - Añadir métricas por paso (dt efectivo, norma de residuo, energía, flux, walltime).
   - **Meta:** diagnóstico post-mortem sin rerun.

4. **Rendimiento**
   - Perfilado con casos representativos (CPU time por ensamblado, solve, I/O).
   - Reducir I/O sin perder ciencia: checkpoints y muestreo adaptativo.
   - **Meta:** mejorar throughput (sim-time/hour) con benchmark reproducible.

5. **Calidad de código**
   - Activar tipado gradual en módulos críticos (`solver`, `physics`, `analysis`).
   - Endurecer lint y static checks en pre-commit/CI.
   - **Meta:** reducir regresiones por cambios en interfaces internas.

### C) Hoja de ruta sugerida (6 semanas)

- **Semana 1-2:** segmentación de tests + entorno reproducible + smoke tests CLI.
- **Semana 3-4:** MMS + balance energético + baseline de reflexión en frontera.
- **Semana 5-6:** calibración QNM + perfilado/rendimiento + reporte técnico final.

### D) Entregables concretos recomendados

1. `docs/validation/mms_report.md` (órdenes de convergencia).
2. `docs/validation/energy_budget.md` (balance energía/flujo).
3. `benchmarks/performance_baseline.md` (tiempo/memoria por caso).
4. Workflow CI separado (`core.yml` y `hpc.yml`) con criterios de aceptación.

## Implementación ejecutada en este ciclo

- ✅ **Segmentación inicial de tests por dependencia**:
  - Tests dependientes de NumPy pasan a `importorskip("numpy")` para evitar errores de colección en entornos mínimos.
  - Se añadió el marcador `requires_numpy` en `pytest.ini`.
- ✅ **Smoke test de CLI**:
  - Se añadió `tests/test_cli_smoke.py` para validar generación de configuración de ejemplo.
- ✅ **CI separada por niveles**:
  - Se agregaron workflows dedicados:
    - `core.yml` para lint + suite sin DOLFINx/slow.
    - `hpc.yml` para pruebas DOLFINx bajo contenedor especializado.
