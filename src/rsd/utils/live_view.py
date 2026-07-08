#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización en vivo del campo escalar con PyVista.

Pensado para demos y debugging en serie (1 rank MPI): `LiveViewer` construye
una sola vez el grid VTK a partir del espacio de funciones y actualiza los
valores nodales de φ en cada frame con `plotter.update()`. La fábrica
`create_live_viewer` degrada con elegancia: si pyvista no está instalado,
no hay display, o la corrida es paralela, loggea un warning y devuelve None
para que la simulación continúe sin ventana.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

from rsd.utils.logger import get_logger

_INSTALL_HINT = "conda install -c conda-forge pyvista"


class LiveViewer:
    """Ventana interactiva PyVista que muestra φ durante la evolución.

    Muestra un corte por el plano z=0 (con fallback a nube de puntos si el
    plano no intersecta la malla), con barra de color fija calibrada con el
    primer frame y el tiempo t en pantalla.

    Args:
        V: Espacio de funciones escalar (dolfinx) donde vive φ
        off_screen: Render sin ventana (para tests/CI); None respeta el
            global de pyvista (p.ej. PYVISTA_OFF_SCREEN=true)
        window_size: Tamaño de la ventana en píxeles
        cmap: Mapa de colores (divergente: φ oscila alrededor de 0)
    """

    def __init__(
        self,
        V,
        off_screen: Optional[bool] = None,
        window_size: Tuple[int, int] = (1024, 768),
        cmap: str = "coolwarm",
    ):
        import pyvista as pv
        from dolfinx.plot import vtk_mesh

        self._logger = get_logger(__name__)
        self._off_screen = bool(pv.OFF_SCREEN) if off_screen is None else bool(off_screen)
        self._shown = False
        self._failed = False
        self._clim: Optional[Tuple[float, float]] = None

        topology, cell_types, geometry = vtk_mesh(V)
        self.grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        self.grid.point_data["phi"] = np.zeros(self.grid.n_points)

        # Corte z=0 para ver el interior del dominio; la geometría del corte
        # es fija, así que solo hay que re-interpolar los escalares por frame
        slc = self.grid.slice(normal="z", origin=(0.0, 0.0, 0.0))
        self._use_slice = slc.n_points > 0
        self._surface = slc if self._use_slice else self.grid

        self.plotter = pv.Plotter(
            off_screen=self._off_screen, window_size=list(window_size)
        )
        if self._use_slice:
            self._actor = self.plotter.add_mesh(
                self._surface,
                scalars="phi",
                cmap=cmap,
                scalar_bar_args={"title": "phi"},
            )
        else:
            self._actor = self.plotter.add_mesh(
                self._surface,
                scalars="phi",
                cmap=cmap,
                style="points",
                point_size=6,
                render_points_as_spheres=True,
                scalar_bar_args={"title": "phi"},
            )
        self._text = self.plotter.add_text("t = 0.000", font_size=12)
        self.plotter.view_xy()

    def _set_time_text(self, text: str) -> None:
        # pyvista >= ~0.44 devuelve CornerAnnotation (SetText(pos, txt));
        # versiones previas devuelven vtkTextActor (SetInput)
        if hasattr(self._text, "SetText"):
            self._text.SetText(2, text)  # 2 = upper_left
        else:
            self._text.SetInput(text)

    @property
    def failed(self) -> bool:
        """True si la vista se desactivó tras un error (p.ej. ventana cerrada)."""
        return self._failed

    def update(self, phi, t: float) -> None:
        """Actualiza la vista con el campo φ al tiempo t.

        No propaga errores: ante cualquier falla (típicamente el usuario
        cerró la ventana) se desactiva con un warning y la evolución sigue.
        """
        if self._failed:
            return
        try:
            values = np.asarray(phi.x.array[: self.grid.n_points], dtype=np.float64)
            self.grid.point_data["phi"] = values
            if self._use_slice:
                new_slice = self.grid.slice(normal="z", origin=(0.0, 0.0, 0.0))
                self._surface.point_data["phi"] = new_slice.point_data["phi"]

            # Barra de color fija: rango simétrico calibrado con el primer frame
            if self._clim is None:
                vmax = float(np.max(np.abs(values))) if values.size else 0.0
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                self._clim = (-vmax, vmax)
                self._actor.mapper.scalar_range = self._clim
                self.plotter.update_scalar_bar_range(self._clim, name="phi")

            self._set_time_text(f"t = {t:.3f}")

            if not self._shown:
                if not self._off_screen:
                    self.plotter.show(
                        title="RSD live: phi",
                        interactive_update=True,
                        auto_close=False,
                    )
                self._shown = True
            if self._off_screen:
                self.plotter.render()
            else:
                self.plotter.update()
        except Exception as exc:
            self._failed = True
            self._logger.warning(f"Live view desactivada tras error: {exc}")

    def close(self) -> None:
        """Cierra la ventana (idempotente, nunca lanza)."""
        try:
            self.plotter.close()
        except Exception:
            pass


def create_live_viewer(V, comm=None, **kwargs) -> Optional[LiveViewer]:
    """Crea un LiveViewer si el entorno lo permite; si no, devuelve None.

    Degradación elegante: cualquier impedimento (pyvista ausente, entorno
    headless, MPI paralelo, fallo de inicialización) produce un warning y
    None, nunca una excepción — la simulación debe continuar sin ventana.

    Args:
        V: Espacio de funciones escalar donde vive φ
        comm: Comunicador MPI de la malla (para detectar corridas paralelas)
        **kwargs: Pasados a LiveViewer (off_screen, window_size, cmap)
    """
    logger = get_logger(__name__)

    if comm is not None and comm.size > 1:
        if comm.rank == 0:
            logger.warning(
                "--live requiere ejecución en serie (1 rank MPI); "
                f"corrida con {comm.size} ranks, continuando sin visualización"
            )
        return None

    try:
        import pyvista
    except ImportError:
        logger.warning(
            f"--live: pyvista no está instalado ({_INSTALL_HINT}); "
            "continuando sin visualización"
        )
        return None

    off_screen = kwargs.get("off_screen")
    if off_screen is None:
        off_screen = bool(pyvista.OFF_SCREEN)
    headless = (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )
    if headless and not off_screen:
        logger.warning(
            "--live: no se detectó display (entorno headless); "
            "continuando sin visualización"
        )
        return None

    try:
        return LiveViewer(V, **kwargs)
    except Exception as exc:
        logger.warning(
            f"--live: no se pudo inicializar la vista ({exc}); "
            "continuando sin visualización"
        )
        return None
