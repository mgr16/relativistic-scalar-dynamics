# 3+1 Klein-Gordon conventions used by PSYOP

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

## Boundary condition

Primary outgoing condition:
\[
\Pi + c_{\text{out}}\partial_n\phi = 0,\qquad c_{\text{out}}=\alpha-\beta\cdot n
\]

Optional spherical Sommerfeld variant:
\[
\Pi + c_{\text{out}}\left(\partial_n\phi + \frac{\phi}{r}\right)=0
\]

## Energy and flux diagnostics

Using the normal observer \(n^\mu\):
\[
\rho = T_{\mu\nu}n^\mu n^\nu,\qquad S_i=-T_{\mu\nu}n^\mu\gamma^\nu{}_i
\]
For the scalar field in first-order variables:
\[
\rho = \tfrac12\Pi^2 + \tfrac12 D_i\phi D^i\phi + V(\phi),\qquad
F_{\text{out}} = -\int_{\partial\Omega} S_i n^i\,dS
= \int_{\partial\Omega}\Pi D_n\phi\,dS.
\]
The solver reports volume energy and outgoing boundary flux using MPI global reductions.
In stationary backgrounds, without radiative losses, \(E(t)\) should be approximately conserved;
with outgoing BC, \(E(t)\) should decay consistently with positive outgoing flux.
