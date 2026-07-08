# 3+1 Klein-Gordon conventions used by RSD

We evolve the covariant scalar equation
\[
\Box \phi - V'(\phi)=0
\]
on a fixed 3+1 background \((\alpha,\beta^i,\gamma_{ij})\).

## Momentum convention

\[
\Pi := \frac{1}{\alpha}\left(\partial_t\phi-\beta^i\partial_i\phi\right)
\]

So the first-order system is
\[
\partial_t\phi = \alpha\Pi + \beta^i\partial_i\phi
\]
\[
\partial_t\Pi = \alpha D_iD^i\phi + D^i\alpha\,D_i\phi + \beta^i\partial_i\Pi + \alpha K\Pi - \alpha V'(\phi)
\]

In the weak (FEM) formulation the terms \(\alpha D_iD^i\phi + D^i\alpha D_i\phi\)
are produced **together** by integrating by parts the single bilinear form
\(-\int \sqrt{\gamma}\,\alpha\,\gamma^{ij}\partial_i\phi\,\partial_j v\,dx\)
(the lapse-gradient term must NOT be added separately, or it is double-counted).

## Extrinsic curvature convention

We use \(K_{ij}=-\tfrac12 \mathcal{L}_n \gamma_{ij}\) (Baumgarte–Shapiro), which is
the convention for which the evolution equation above carries \(+\alpha K \Pi\).
For a **stationary** background this trace reduces to
\[
K = \frac{1}{\alpha} D_i\beta^i = \frac{1}{\alpha\sqrt{\gamma}}\,\partial_i\!\left(\sqrt{\gamma}\,\beta^i\right),
\]
which RSD evaluates symbolically (UFL `div`). For Schwarzschild in
Kerr-Schild coordinates this reproduces the known value
\[
K = \frac{2M\,(r+3M)}{\left[r\,(r+2M)\right]^{3/2}}
  = \frac{2M\alpha^3}{r^2}\left(1+\frac{3M}{r}\right) > 0 .
\]
Static slicings (flat space, Schwarzschild in isotropic coordinates) have \(K=0\).

## Boundary conditions

### Outer boundary (radiative)

Integrating the diffusion term by parts produces the natural boundary term
\(+\int \alpha\sqrt{\gamma}\,(\gamma^{ij}\partial_i\phi\, n_j)\,v\,ds\).
The outgoing characteristic condition \(\partial_t\phi + \lambda\,\partial_n\phi = 0\)
with \(\lambda = \alpha\sqrt{\gamma^{nn}} - \beta\cdot n\) and
\(\gamma^{nn} = n_i\gamma^{ij}n_j\) reduces (neglecting tangential derivatives) to
\[
\Pi = -\sqrt{\gamma^{nn}}\,\partial_n\phi ,
\]
so the conormal flux is replaced consistently in the weak form:
\[
\gamma^{ij}\partial_i\phi\,n_j \;\longrightarrow\; -\sqrt{\gamma^{nn}}\,\Pi .
\]
In flat space this is the textbook absorbing term \(-\int \Pi\,v\,ds\).

Optional spherical Sommerfeld variant (better for \(1/r\) tails):
\[
\Pi = -\sqrt{\gamma^{nn}}\,\partial_n\phi - \frac{\gamma^{nn}}{\sqrt{\gamma^{nn}}}\frac{\phi}{r}
\quad\Longrightarrow\quad
\gamma^{ij}\partial_i\phi\,n_j \to -\left(\sqrt{\gamma^{nn}}\,\Pi + \gamma^{nn}\frac{\phi}{r}\right).
\]

### Inner boundary (excision)

Black hole backgrounds require an excised inner sphere (`mesh.r_inner > 0`).
Inside the horizon of a horizon-penetrating slicing (Kerr-Schild) all
characteristics leave the computational domain through the inner boundary, so
**no condition** may be imposed there. In the weak form this corresponds to
keeping the natural ("do-nothing") boundary term evaluated with the interior
solution:
\[
+\int_{\text{inner}} \alpha\sqrt{\gamma}\,(\gamma^{ij}\partial_i\phi\,n_j)\,v\,ds .
\]
For Schwarzschild in isotropic coordinates the lapse vanishes at \(r = M/2\),
so excising at (or slightly inside) that radius is appropriate.

## Energy and flux diagnostics

Using the normal observer \(n^\mu\):
\[
\rho = T_{\mu\nu}n^\mu n^\nu,\qquad S_i=-T_{\mu\nu}n^\mu\gamma^\nu{}_i
\]
For the scalar field in first-order variables:
\[
\rho = \tfrac12\Pi^2 + \tfrac12 D_i\phi D^i\phi + V(\phi),
\]
\[
F_{\text{out}} = -\oint_{\partial\Omega} \alpha\sqrt{\gamma}\;\Pi\,
\left(\gamma^{ij}\partial_i\phi\, n_j\right) ds ,
\]
with the sign convention that **outgoing radiation gives \(F_{\text{out}} > 0\)**
(for an outgoing plane wave \(\Pi = -\partial_n\phi\) and \(F_{\text{out}} = \oint \Pi^2\,ds\)).
The solver reports volume energy and outgoing boundary flux using MPI global reductions.
In stationary backgrounds, without radiative losses, \(E(t)\) should be approximately conserved;
with outgoing BC, \(E(t)\) should decay consistently with positive outgoing flux,
so that \(E(t) + \int_0^t F_{\text{out}}\,dt' \approx E(0)\).
