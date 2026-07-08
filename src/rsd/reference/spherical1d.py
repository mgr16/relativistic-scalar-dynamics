#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oráculo 1D: Klein-Gordon esférico sobre Schwarzschild en Kerr-Schild.

Resuelve el sector radial del mismo sistema (φ, Π) que el solver 3D para un
modo multipolar φ = u(t, r)·Y_lm, con las MISMAS convenciones 3+1
(Π = (∂_t u − β^r ∂_r u)/α, K_ij = -½£_n γ_ij):

    ∂_t u = α Π + β^r ∂_r u
    ∂_t Π = (1/w) ∂_r(w α γ^rr ∂_r u) − α l(l+1)/r² u
            + β^r ∂_r Π + α K Π − α V'(u)

con w(r) = r²√(1+2M/r) la densidad radial de volumen. Fondo Schwarzschild
en coordenadas Kerr-Schild (horizon-penetrating, M=0 ⇒ plano):

    α = (1+2M/r)^{-1/2},  β^r = (2M/r)/(1+2M/r),  γ^rr = α²,
    K = 2Mα³(1+3M/r)/r²   (idéntico al valor citado en docs/math).

Propósito (Fase 0 del programa de investigación):
- Referencia independiente de alta resolución para validar el código 3D
  (ringdown QNM contra Leaver, perfiles del interior).
- Exploración barata del sector esférico de la hipótesis H2: asintótica
  del campo hacia la singularidad (r→0) y dominación cinética sobre el
  término de potencial.

Numérica: diferencias finitas de 2.º orden sobre malla radial logarítmica
(o uniforme), SSP-RK3 (idéntico al 3D), filtro de paso bajo tipo
Kreiss-Oliger (D⁴ adimensional, post-paso). Velocidad característica
coordenada acotada por 1 exactamente (el rayo nulo entrante de Kerr-Schild
cumple dr/dt = −1 para todo r), así que dt = cfl·min(Δr).

Fronteras:
- interior (r_min): outflow puro si r_min está dentro del horizonte
  (ambas características salen del dominio); derivadas laterales + ghost
  por extrapolación cuadrática para el término de flujo. Con M=0 actúa
  como borde absorbente (no físico cerca de r=0; usar pulsos salientes).
- exterior (r_max): advección saliente de (u−u∞) y Π con velocidad
  característica local λ_out = α²(1−2M/r), variante esférica (término q/r).

Solo NumPy: corre sin DOLFINx (ruta Core CI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from rsd.utils.logger import get_logger

logger = get_logger(__name__)

# numpy >= 2.0 renombró trapz -> trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

VALID_GRIDS = ("log", "uniform")
VALID_DIRECTIONS = ("static", "ingoing", "outgoing", "ingoing_curved")


@dataclass
class OracleResult:
    """Series de tiempo y snapshots de una evolución del oráculo."""

    ts: np.ndarray
    probes: Dict[float, np.ndarray]          # radio -> u(t) interpolado
    energies: np.ndarray
    snapshot_ts: List[float] = field(default_factory=list)
    snapshots_u: List[np.ndarray] = field(default_factory=list)
    snapshots_Pi: List[np.ndarray] = field(default_factory=list)
    # energía de Killing y sus flujos de borde (positivos al salir);
    # E_K(t) − E_K(0) + ∫(F_in + F_out)dt → 0 (docs/math/killing_energy.md)
    energies_killing: np.ndarray = field(default_factory=lambda: np.empty(0))
    flux_inner_killing: np.ndarray = field(default_factory=lambda: np.empty(0))
    flux_outer_killing: np.ndarray = field(default_factory=lambda: np.empty(0))


class SphericalOracle1D:
    """KG esférico (modo l) sobre Schwarzschild-KS por diferencias finitas.

    Args:
        M: masa del agujero negro (0 = espacio plano)
        l: número multipolar del modo (potencial no lineal solo con l=0)
        r_min, r_max: dominio radial (r_min < 2M ⇒ interior incluido)
        n_points: número de nodos radiales
        grid: "log" (uniforme en ln r, resuelve r→0) o "uniform"
        potential_type / potential_params: igual que el solver 3D
        cfl: factor CFL sobre dt = cfl·min(Δr) (velocidad máxima = 1)
        ko_eps: amplitud del filtro D⁴ post-paso (0 = sin filtro)
        u_infinity: valor asintótico del campo (la BC saliente advecta u−u∞)
    """

    def __init__(
        self,
        M: float = 1.0,
        l: int = 0,
        r_min: float = 1.0,
        r_max: float = 100.0,
        n_points: int = 2000,
        grid: str = "log",
        potential_type: str = "zero",
        potential_params: Optional[dict] = None,
        cfl: float = 0.4,
        ko_eps: float = 0.02,
        u_infinity: float = 0.0,
    ):
        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")
        if l < 0:
            raise ValueError(f"l must be >= 0, got {l}")
        if not (0 < r_min < r_max):
            raise ValueError(f"need 0 < r_min < r_max, got [{r_min}, {r_max}]")
        if n_points < 16:
            raise ValueError(f"n_points must be >= 16, got {n_points}")
        if grid not in VALID_GRIDS:
            raise ValueError(f"grid must be one of {VALID_GRIDS}, got {grid!r}")
        if not (0 < cfl <= 1):
            raise ValueError(f"cfl must be in (0, 1], got {cfl}")
        if not (0 <= ko_eps < 1):
            raise ValueError(f"ko_eps must be in [0, 1), got {ko_eps}")

        self.M = float(M)
        self.l = int(l)
        self.cfl = float(cfl)
        self.ko_eps = float(ko_eps)
        self.u_infinity = float(u_infinity)

        from rsd.physics.potential import get_potential

        self.potential = get_potential(potential_type, **(potential_params or {}))
        self._V_is_zero = potential_type == "zero"
        if not self._V_is_zero and self.l != 0:
            # V'(u·Y_lm) no factoriza en Y_lm salvo para l=0; un potencial no
            # lineal acoplaría multipolos y este oráculo evoluciona uno solo.
            logger.warning(
                "potencial no nulo con l>0: la reducción 1D solo es exacta "
                "para potenciales lineales en u (cuadrático); interprete con cuidado"
            )

        # --- malla radial ---
        if grid == "log":
            self.r = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_points))
            self.r[0], self.r[-1] = r_min, r_max  # exactos en los extremos
        else:
            self.r = np.linspace(r_min, r_max, n_points)
        self.n = n_points
        self.dr_min = float(np.min(np.diff(self.r)))

        # --- coeficientes de fondo (estacionarios: se precomputan una vez) ---
        r = self.r
        with np.errstate(divide="ignore"):
            two_m_over_r = 2.0 * self.M / r
        self.alpha = 1.0 / np.sqrt(1.0 + two_m_over_r)
        self.beta = two_m_over_r / (1.0 + two_m_over_r)
        self.gamma_rr_inv = self.alpha**2
        self.w = r**2 * np.sqrt(1.0 + two_m_over_r)
        if self.M > 0:
            self.K = 2.0 * self.M * self.alpha**3 * (1.0 + 3.0 * self.M / r) / r**2
        else:
            self.K = np.zeros_like(r)
        # coeficiente de difusión del flujo: D = α γ^rr = α³
        self.D = self.alpha**3
        # w·D en los puntos medios para el término conservativo
        r_mid = 0.5 * (r[:-1] + r[1:])
        self.wD_mid = self._wD(r_mid)
        # ancho de celda dual (denominador de la divergencia del flujo)
        self.dr_dual = np.empty(self.n)
        self.dr_dual[1:-1] = 0.5 * (r[2:] - r[:-2])
        self.dr_dual[0] = r[1] - r[0]
        self.dr_dual[-1] = r[-1] - r[-2]

        self._setup_derivative_stencils()
        self._setup_inner_ghost()

        # velocidad saliente local para la BC exterior
        self.lam_out_boundary = float(
            self.alpha[-1] ** 2 * (1.0 - 2.0 * self.M / r[-1])
        )

        # --- estado ---
        self.u = np.full(self.n, self.u_infinity, dtype=float)
        self.Pi = np.zeros(self.n, dtype=float)
        self.current_time = 0.0

        logger.info(
            f"SphericalOracle1D: M={self.M}, l={self.l}, r∈[{r_min}, {r_max}], "
            f"N={n_points} ({grid}), dt_CFL={self.compute_dt():.3e}"
        )

    # ------------------------------------------------------------------
    # geometría y stencils
    # ------------------------------------------------------------------

    def _wD(self, r: np.ndarray) -> np.ndarray:
        """w·D = r³/(r+2M) (analítico; w=r^{3/2}(r+2M)^{1/2}, D=α³)."""
        return r**3 / (r + 2.0 * self.M)

    def _setup_derivative_stencils(self) -> None:
        """Coeficientes de la primera derivada centrada (malla no uniforme)."""
        r = self.r
        hm = r[1:-1] - r[:-2]   # h⁻
        hp = r[2:] - r[1:-1]    # h⁺
        self._c_m = -hp / (hm * (hm + hp))
        self._c_0 = (hp - hm) / (hp * hm)
        self._c_p = hm / (hp * (hm + hp))
        # derivada lateral de 2.º orden en i=0 (puntos 0,1,2) — upwind exacto
        # en el borde interior outflow (la información viene de r mayores)
        h1 = r[1] - r[0]
        h2 = r[2] - r[1]
        self._b0 = np.array([
            -(2.0 * h1 + h2) / (h1 * (h1 + h2)),
            (h1 + h2) / (h1 * h2),
            -h1 / (h2 * (h1 + h2)),
        ])
        # derivada lateral de 2.º orden en i=N-1 (puntos N-3,N-2,N-1)
        g1 = r[-2] - r[-3]
        g2 = r[-1] - r[-2]
        self._bN = np.array([
            g2 / (g1 * (g1 + g2)),
            -(g1 + g2) / (g1 * g2),
            (g1 + 2.0 * g2) / (g2 * (g1 + g2)),
        ])

    def _setup_inner_ghost(self) -> None:
        """Ghost interior por extrapolación cuadrática (outflow)."""
        r = self.r
        self._r_ghost = 2.0 * r[0] - r[1]
        if self._r_ghost <= 0.0:
            self._r_ghost = 0.5 * r[0]
        x0, x1, x2 = r[0], r[1], r[2]
        xg = self._r_ghost
        # pesos de Lagrange de la cuadrática por (x0,x1,x2) evaluada en xg
        self._ghost_w = np.array([
            (xg - x1) * (xg - x2) / ((x0 - x1) * (x0 - x2)),
            (xg - x0) * (xg - x2) / ((x1 - x0) * (x1 - x2)),
            (xg - x0) * (xg - x1) / ((x2 - x0) * (x2 - x1)),
        ])
        self._wD_ghost_mid = self._wD(0.5 * (self._r_ghost + r[0]))
        self._dr_ghost = r[0] - self._r_ghost

    # ------------------------------------------------------------------
    # operadores espaciales
    # ------------------------------------------------------------------

    def _deriv_r(self, f: np.ndarray) -> np.ndarray:
        """∂_r f, centrada en el interior y lateral 2.º orden en los bordes."""
        df = np.empty_like(f)
        df[1:-1] = self._c_m * f[:-2] + self._c_0 * f[1:-1] + self._c_p * f[2:]
        df[0] = self._b0 @ f[:3]
        df[-1] = self._bN @ f[-3:]
        return df

    def _flux_divergence(self, u: np.ndarray) -> np.ndarray:
        """(1/w) ∂_r(w D ∂_r u) en forma conservativa (flujos en caras)."""
        flux = self.wD_mid * (u[1:] - u[:-1]) / np.diff(self.r)
        div = np.empty(self.n)
        div[1:-1] = (flux[1:] - flux[:-1]) / self.dr_dual[1:-1]
        # cara fantasma interior: flujo con u extrapolado cuadráticamente
        u_ghost = self._ghost_w @ u[:3]
        flux_ghost = self._wD_ghost_mid * (u[0] - u_ghost) / self._dr_ghost
        div[0] = (flux[0] - flux_ghost) / self.dr_dual[0]
        # el valor en i=N-1 lo sobreescribe la BC exterior
        div[-1] = 0.0
        return div / self.w

    def rhs(self, u: np.ndarray, Pi: np.ndarray) -> tuple:
        """L(u, Π) del sistema de primer orden (sin filtro)."""
        du_r = self._deriv_r(u)
        dPi_r = self._deriv_r(Pi)

        L_u = self.alpha * Pi + self.beta * du_r
        L_Pi = (
            self._flux_divergence(u)
            + self.beta * dPi_r
            + self.alpha * self.K * Pi
        )
        if self.l > 0:
            L_Pi -= self.alpha * self.l * (self.l + 1) / self.r**2 * u
        if not self._V_is_zero:
            L_Pi -= self.alpha * self.potential.derivative_np(u)

        # BC exterior: advección saliente esférica de (u−u∞) y Π
        lam = self.lam_out_boundary
        rN = self.r[-1]
        L_u[-1] = -lam * (du_r[-1] + (u[-1] - self.u_infinity) / rN)
        L_Pi[-1] = -lam * (dPi_r[-1] + Pi[-1] / rN)
        return L_u, L_Pi

    # ------------------------------------------------------------------
    # integración temporal
    # ------------------------------------------------------------------

    def compute_dt(self) -> float:
        """dt CFL: velocidad característica coordenada ≤ 1 exactamente."""
        return self.cfl * self.dr_min

    def step(self, dt: float) -> None:
        """Un paso SSP-RK3 (mismo esquema que el solver 3D) + filtro D⁴."""
        u0, Pi0 = self.u, self.Pi

        k_u, k_Pi = self.rhs(u0, Pi0)
        u1 = u0 + dt * k_u
        Pi1 = Pi0 + dt * k_Pi

        k_u, k_Pi = self.rhs(u1, Pi1)
        u2 = 0.75 * u0 + 0.25 * (u1 + dt * k_u)
        Pi2 = 0.75 * Pi0 + 0.25 * (Pi1 + dt * k_Pi)

        k_u, k_Pi = self.rhs(u2, Pi2)
        self.u = u0 / 3.0 + 2.0 / 3.0 * (u2 + dt * k_u)
        self.Pi = Pi0 / 3.0 + 2.0 / 3.0 * (Pi2 + dt * k_Pi)

        if self.ko_eps > 0:
            self._apply_filter(self.u, offset=self.u_infinity)
            self._apply_filter(self.Pi)
        self.current_time += dt

    def _apply_filter(self, f: np.ndarray, offset: float = 0.0) -> None:
        """Filtro paso-bajo D⁴ adimensional (índices), amplitud ko_eps/16."""
        g = f - offset
        d4 = g[:-4] - 4.0 * g[1:-3] + 6.0 * g[2:-2] - 4.0 * g[3:-1] + g[4:]
        f[2:-2] -= (self.ko_eps / 16.0) * d4

    # ------------------------------------------------------------------
    # condiciones iniciales
    # ------------------------------------------------------------------

    def set_initial_gaussian(
        self,
        A: float = 1e-3,
        r0: float = 8.0,
        width: float = 2.0,
        direction: str = "static",
    ) -> None:
        """Cascarón gaussiano u = u∞ + A·exp(−(r−r0)²/w²) con momento opcional.

        direction:
            "static": Π = 0 (el pulso se parte en mitades)
            "ingoing"/"outgoing": relación de onda esférica plana
                Π = ±(∂_r u_pert + u_pert/r)  (idéntica al código 3D)
            "ingoing_curved": relación consistente con Kerr-Schild, donde el
                rayo nulo entrante cumple exactamente dr/dt = −1, de modo que
                u ≈ g(t+r)/r ⇒ Π = ((1−β)∂_r u_pert + u_pert/r)/α.
                Con M=0 se reduce a "ingoing".
        """
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}"
            )
        r = self.r
        pert = A * np.exp(-((r - r0) ** 2) / width**2)
        dpert = -2.0 * (r - r0) / width**2 * pert
        self.u = self.u_infinity + pert
        if direction == "static":
            self.Pi = np.zeros_like(r)
        elif direction == "ingoing":
            self.Pi = dpert + pert / r
        elif direction == "outgoing":
            self.Pi = -(dpert + pert / r)
        else:  # ingoing_curved
            self.Pi = ((1.0 - self.beta) * dpert + pert / r) / self.alpha
        self.current_time = 0.0

    # ------------------------------------------------------------------
    # diagnósticos
    # ------------------------------------------------------------------

    def energy(self) -> float:
        """E = ∫ w [½Π² + ½γ^rr(∂_r u)² + ½ l(l+1)u²/r² + V(u)] dr."""
        return float(_trapezoid(self.w * self.energy_density(), self.r))

    def energy_killing(
        self, u: Optional[np.ndarray] = None, Pi: Optional[np.ndarray] = None
    ) -> float:
        """E_K = ∫ w (α ρ + β^r Π ∂_r u) dr — energía de Killing (ξ = ∂_t).

        A diferencia de energy(), su balance cierra con flujos puros de
        superficie también sobre la foliación estacionaria de Kerr-Schild
        (sin términos de volumen β/K); ver docs/math/killing_energy.md.
        """
        u = self.u if u is None else u
        Pi = self.Pi if Pi is None else Pi
        du_r = self._deriv_r(u)
        eps_K = self.alpha * self.energy_density(u, Pi) + self.beta * Pi * du_r
        return float(_trapezoid(self.w * eps_K, self.r))

    def killing_boundary_flux(
        self, u: Optional[np.ndarray] = None, Pi: Optional[np.ndarray] = None
    ) -> tuple:
        """Flujo de energía de Killing que SALE por cada borde (F_in, F_out).

        F(r) = α w T^r_t con T^r_t = (αΠ + β^r u′)·α·((2M/r)Π + α u′);
        el balance exacto es E_K(t) − E_K(0) + ∫(F_in + F_out)dt = 0.
        Positivo = energía abandonando el dominio (por r_min: absorción).
        """
        u = self.u if u is None else u
        Pi = self.Pi if Pi is None else Pi

        def _awTrt(i: int, du_i: float) -> float:
            r_i = self.r[i]
            phidot = self.alpha[i] * Pi[i] + self.beta[i] * du_i
            radial = self.alpha[i] * (
                (2.0 * self.M / r_i) * Pi[i] + self.alpha[i] * du_i
            )
            return float(self.alpha[i] * self.w[i] * phidot * radial)

        du_0 = float(self._b0 @ u[:3])
        du_N = float(self._bN @ u[-3:])
        return _awTrt(0, du_0), -_awTrt(-1, du_N)

    def rhs_term_breakdown(
        self, u: Optional[np.ndarray] = None, Pi: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Magnitud puntual de cada término de ∂_tΠ (diagnóstico H2).

        La razón |potential| / (suma de los demás) mide la dominación
        cinética: ≪1 ⇒ el potencial es dinámicamente irrelevante.
        Acepta snapshots (u, Π) externos; por defecto usa el estado actual.
        """
        u = self.u if u is None else u
        Pi = self.Pi if Pi is None else Pi
        dPi_r = self._deriv_r(Pi)
        out = {
            "flux_div": self._flux_divergence(u),
            "transport": self.beta * dPi_r,
            "extrinsic": self.alpha * self.K * Pi,
            "angular": (
                -self.alpha * self.l * (self.l + 1) / self.r**2 * u
                if self.l > 0
                else np.zeros(self.n)
            ),
            "potential": (
                -self.alpha * self.potential.derivative_np(u)
                if not self._V_is_zero
                else np.zeros(self.n)
            ),
        }
        return out

    def energy_density(
        self, u: Optional[np.ndarray] = None, Pi: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """ρ = ½Π² + ½γ^rr(∂_r u)² + ½l(l+1)u²/r² + V(u) (sin pesar por w)."""
        u = self.u if u is None else u
        Pi = self.Pi if Pi is None else Pi
        du_r = self._deriv_r(u)
        rho = 0.5 * Pi**2 + 0.5 * self.gamma_rr_inv * du_r**2
        if self.l > 0:
            rho += 0.5 * self.l * (self.l + 1) * (u - self.u_infinity) ** 2 / self.r**2
        if not self._V_is_zero:
            rho += self.potential.evaluate_np(u)
        return rho

    def log_slope(self) -> np.ndarray:
        """∂u/∂ln r = r·∂_r u (constante ⇔ perfil logarítmico u ~ A ln r)."""
        return self.r * self._deriv_r(self.u)

    # ------------------------------------------------------------------
    # bucle de evolución
    # ------------------------------------------------------------------

    def evolve(
        self,
        t_end: float,
        dt: Optional[float] = None,
        probe_radii: Optional[List[float]] = None,
        output_every: int = 10,
        snapshot_every: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> OracleResult:
        """Evoluciona hasta t_end registrando sondas, energía y snapshots.

        Args:
            probe_radii: radios donde interpolar u(t) cada output_every pasos
            snapshot_every: cada cuántos pasos guardar (u, Π) completos
            callback: f(solver, step) llamada cada output_every pasos
        """
        if dt is None:
            dt = self.compute_dt()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        probe_radii = list(probe_radii or [])
        for rp in probe_radii:
            if not (self.r[0] <= rp <= self.r[-1]):
                raise ValueError(f"probe radius {rp} outside domain")

        ts: List[float] = []
        probes: Dict[float, List[float]] = {rp: [] for rp in probe_radii}
        energies: List[float] = []
        energies_k: List[float] = []
        fk_in: List[float] = []
        fk_out: List[float] = []
        result = OracleResult(
            ts=np.empty(0), probes={}, energies=np.empty(0)
        )

        n_steps = int(np.ceil((t_end - self.current_time) / dt - 1e-12))
        for step_i in range(n_steps):
            step_dt = min(dt, t_end - self.current_time)
            self.step(step_dt)
            if step_i % output_every == 0 or step_i == n_steps - 1:
                if not np.all(np.isfinite(self.u)):
                    raise FloatingPointError(
                        f"u no finito en t={self.current_time:.4f} "
                        f"(paso {step_i}); reduzca cfl o aumente ko_eps"
                    )
                ts.append(self.current_time)
                for rp in probe_radii:
                    probes[rp].append(float(np.interp(rp, self.r, self.u)))
                energies.append(self.energy())
                energies_k.append(self.energy_killing())
                f_in, f_out = self.killing_boundary_flux()
                fk_in.append(f_in)
                fk_out.append(f_out)
                if callback is not None:
                    callback(self, step_i)
            if snapshot_every and step_i % snapshot_every == 0:
                result.snapshot_ts.append(self.current_time)
                result.snapshots_u.append(self.u.copy())
                result.snapshots_Pi.append(self.Pi.copy())

        result.ts = np.asarray(ts)
        result.probes = {rp: np.asarray(v) for rp, v in probes.items()}
        result.energies = np.asarray(energies)
        result.energies_killing = np.asarray(energies_k)
        result.flux_inner_killing = np.asarray(fk_in)
        result.flux_outer_killing = np.asarray(fk_out)
        return result


def trace_K_from_divergence(oracle: SphericalOracle1D) -> np.ndarray:
    """K = (1/(α w)) ∂_r(w β) por diferencias finitas (cross-check del valor
    analítico precomputado; misma definición que usa el solver 3D vía UFL)."""
    wbeta = oracle.w * oracle.beta
    return oracle._deriv_r(wbeta) / (oracle.alpha * oracle.w)
