# Price-tail slow-test diagnostic (R2.4)

Date: 2026-07-16. Status: **[REVIEW] expected failure documented; tolerance
unchanged**.

## Reproduction

The single pre-authorized 3D evolution exactly reproduced
`tests/test_price_tails_slow.py::test_massless_l1_price_tail`: Schwarzschild,
$l=1$, $R=60M$, `lc=3M`, `lc_inner=0.5M`, extraction at $6M$, and
$t_{end}=110M$. The versioned outputs are:

- `docs/research/phase3/price_tail_diagnostic.json`;
- `docs/research/phase3/data/price_tail_diagnostic.npz`;
- generator/check: `scripts/price_tail_diagnostic.py`.

The original $55\leq t/M\leq100$ fit gives $p=-1.983424$ rather than the
Price prediction $-5$, with $R^2=0.09547$. The poor $R^2$ is decisive: the
series is not a clean power law in that window.

## Measured floor

No shifted window repairs the measurement. Fits on windows from $45$--$65M$
through $80$--$100M$ span exponents from $-7.61$ to $+7.11$, while every
$R^2$ remains below $0.357$. In particular, the superficially compatible
$45$--$65M$ exponent $-5.32$ has $R^2=0.357$ and is not a valid Price-tail
detection.

The binned RMS falls from $2.93\times10^{-5}$ at $55$--$65M$ to a minimum
$6.27\times10^{-6}$ at $75$--$85M$, then rises to
$1.63\times10^{-5}$ at $95$--$100M$. The last/first RMS-bin ratio is $0.557$;
a $t^{-5}$ envelope over those bin centers predicts $0.0883$. A two-line fit
to the late floor lowers the RMS only from $1.52\times10^{-5}$ to
$9.68\times10^{-6}$ and returns unresolved low frequencies with half-widths
$0.05/M$, so the contamination is not a clean, citable cavity line either.

## Diagnosis and disposition

The coarse 3D signal reaches a non-power-law numerical/domain-junk floor
before the nominal fit ends. The direct outer-boundary echo estimate
($t\simeq106M$) therefore does not guarantee a clean tail: the absorbing
layer starts inside the boundary, and mesh/extraction junk is already larger
than the predicted late signal. This run cannot distinguish the relative
mesh, extraction, and sponge contributions without additional controlled 3D
ablations; none is authorized in round R.

The test is retained with its original exponent and $R^2$ tolerances and is
marked `xfail(strict=False)` with this pointer. A real future repair requires
a redesigned validation (for example a converged radial benchmark followed by
larger/finer 3D and sponge-onset ablations), not a wider acceptance band.
