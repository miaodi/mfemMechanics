# Infinitesimal J2 plasticity

## Scope

`J2PlasticityMaterial` implements associative, rate-independent von Mises
plasticity at infinitesimal strain with optional linear isotropic hardening. It
uses a three-dimensional constitutive state in both two- and three-dimensional
finite-element analyses. A two-dimensional analysis is therefore plane strain:
$\varepsilon_{zz}=\varepsilon_{xz}=\varepsilon_{yz}=0$, while
$\sigma_{zz}$ and $\varepsilon^p_{zz}$ generally remain nonzero.

The model assumes isotropic elasticity, isothermal response, no rate effects,
and a strain history divided into accepted load or time increments. It does not
implement finite-strain plasticity, plane stress, kinematic or nonlinear
hardening, viscosity, damage, plastic heating, contact, AMR state transfer, or
restart serialization. Infinitesimal kinematics cannot represent geometric
necking.

All stress-like inputs must use one consistent unit. Young's modulus $E$,
initial yield stress $\sigma_{y0}$, and hardening modulus $H$ have that unit;
Poisson's ratio $\nu$, strain, plastic strain, and accumulated equivalent
plastic strain $\alpha$ are dimensionless. Admissible parameters are

$$
E>0,\qquad -1<\nu<\tfrac12,\qquad \sigma_{y0}\geq0,\qquad H\geq0.
$$

$H=0$ selects perfect plasticity. The incompressible elastic limit
$\nu\rightarrow1/2$ is excluded because the bulk modulus becomes singular and
the displacement-only discretization becomes poorly conditioned before that
limit.

## Constitutive equations

The additive strain decomposition and isotropic elastic law are

$$
\boldsymbol\varepsilon_m
  = \boldsymbol\varepsilon_e+\boldsymbol\varepsilon^p,
\qquad
\boldsymbol\sigma
  = K\operatorname{tr}(\boldsymbol\varepsilon_e)\mathbf I
    +2G\operatorname{dev}(\boldsymbol\varepsilon_e),
$$

with

$$
G=\frac{E}{2(1+\nu)},
\qquad
K=\frac{E}{3(1-2\nu)}.
$$

If stress-free strains are registered with the integrator, its mechanical
strain input is

$$
\boldsymbol\varepsilon_m
  = \operatorname{sym}(\nabla\mathbf u)-\boldsymbol\varepsilon_0.
$$

For the deviatoric stress $\mathbf s=\operatorname{dev}(\boldsymbol\sigma)$,
the equivalent stress, yield function, and linear hardening law are

$$
q=\sqrt{\tfrac32\,\mathbf s:\mathbf s},
\qquad
f=q-(\sigma_{y0}+H\alpha),
\qquad
Y(\alpha)=\sigma_{y0}+H\alpha.
$$

Associative flow and accumulated plastic strain satisfy

$$
\dot{\boldsymbol\varepsilon}^p
  = \dot\gamma\frac{3}{2}\frac{\mathbf s}{q},
\qquad
\dot\alpha=\dot\gamma,
\qquad
\dot\gamma\geq0,\quad f\leq0,\quad \dot\gamma f=0.
$$

The flow is deviatoric, so plastic incompressibility is preserved. The stored
plastic-strain tensor is required to be symmetric and trace-free.

## Radial return

At an evaluation in increment $n+1$, trial quantities are computed only from
the committed state $(\boldsymbol\varepsilon^p_n,\alpha_n)$ and the current
mechanical strain:

$$
\mathbf e_e^{\mathrm{tr}}
  = \operatorname{dev}(\boldsymbol\varepsilon_m
      -\boldsymbol\varepsilon^p_n),
\qquad
\mathbf s^{\mathrm{tr}}=2G\mathbf e_e^{\mathrm{tr}},
\qquad
q^{\mathrm{tr}}=\sqrt{\tfrac32\,\mathbf s^{\mathrm{tr}}:
                                      \mathbf s^{\mathrm{tr}}}.
$$

If $f^{\mathrm{tr}}=q^{\mathrm{tr}}-Y(\alpha_n)$ is nonpositive within a
precision-scaled tolerance, the step is elastic. Otherwise the closed-form
consistency increment for linear hardening is

$$
\Delta\gamma=\frac{f^{\mathrm{tr}}}{3G+H}.
$$

Defining

$$
\mathbf m=\frac{\mathbf s^{\mathrm{tr}}}
                 {\|\mathbf s^{\mathrm{tr}}\|},
\qquad
a=1-\frac{3G\Delta\gamma}{q^{\mathrm{tr}}},
$$

the radial update is

$$
\mathbf s_{n+1}=a\mathbf s^{\mathrm{tr}},
\qquad
\boldsymbol\varepsilon^p_{n+1}
  =\boldsymbol\varepsilon^p_n
   +\Delta\gamma\frac32
      \frac{\mathbf s^{\mathrm{tr}}}{q^{\mathrm{tr}}},
\qquad
\alpha_{n+1}=\alpha_n+\Delta\gamma.
$$

The pressure remains elastic:

$$
p_{n+1}=K\operatorname{tr}(\boldsymbol\varepsilon_m
                            -\boldsymbol\varepsilon^p_n),
\qquad
\boldsymbol\sigma_{n+1}=p_{n+1}\mathbf I+\mathbf s_{n+1}.
$$

These formulas are specialized to linear hardening. A general hardening law
would require solving a scalar nonlinear consistency equation rather than
silently reusing this denominator.

## Consistent tangent

The implementation differentiates the discrete radial-return map, not the
continuous rate equations. On the elastic branch, the tangent is the isotropic
elastic tensor

$$
\mathbb C_e=K\mathbf I\otimes\mathbf I+2G\mathbb P_{\mathrm{dev}}.
$$

On the plastic branch, the repository-specific derivation gives

$$
\mathbb C_{\mathrm{ep}}
  =K\mathbf I\otimes\mathbf I
   +2Ga\mathbb P_{\mathrm{dev}}
   -6G^2\left(\frac{1}{3G+H}
        -\frac{\Delta\gamma}{q^{\mathrm{tr}}}\right)
       \mathbf m\otimes\mathbf m.
$$

This tangent is symmetric for associative J2 flow with isotropic hardening.
The code stores it as a $6\times6$ matrix mapping engineering strain Voigt
vectors to stress Voigt vectors. The ordering is
`[xx, yy, zz, xy, yz, xz]`; strain shear entries are
$[2\varepsilon_{xy},2\varepsilon_{yz},2\varepsilon_{xz}]$, while stress shear
entries are unscaled. This convention accounts for the one-half entries in the
Voigt representation of $\mathbb P_{\mathrm{dev}}$.

## Discrete residual

For test function $\mathbf v$ and applied traction $\bar{\mathbf t}$, the
quasistatic weak residual at load factor $\lambda$ is

$$
R(\mathbf u;\mathbf v)
  =\int_{\Omega}\boldsymbol\varepsilon(\mathbf v):
          \boldsymbol\sigma(\mathbf u)\,dV
   -\lambda\int_{\Gamma_t}\mathbf v\cdot\bar{\mathbf t}\,dA.
$$

At an integration point with strain-displacement matrix $\mathbf B$, the
element residual and material Jacobian are

$$
\mathbf r_e=\sum_q w_qJ_q\mathbf B_q^T\boldsymbol\sigma_q,
\qquad
\mathbf K_e=\sum_q w_qJ_q\mathbf B_q^T
             \mathbb C_{\mathrm{ep},q}\mathbf B_q.
$$

Residual and Jacobian use the same integration rule, kinematics, material
parameters, committed state, and quadrature measure. The default rule has order
$2p+1$ for an element of order $p$; assigning the integrator's `IntRule`
overrides it for both paths.

MFEM passes local vector element DOFs to nonlinear integrators in component
blocks even when the global finite-element space uses `Ordering::byNODES`.
The integrator's `dofs`-by-`dimension` Eigen map therefore supports both global
MFEM orderings; a regression test compares their assembled residuals DOF by DOF.

## State lifecycle

Each integration point stores committed and trial
`J2PlasticityState` values. Constitutive evaluation is a deterministic function
of the committed state and current trial strain. Residual or Jacobian assembly
replaces trial state but never commits it.

`BeginStep`, `CommitStep`, and `RollbackStep` are driven by the nonlinear solver.
A converged outermost transaction commits the latest trial state. A failed
Newton solve or rejected nested transaction restores both the last accepted
unknown vector and every material history. No general AMR transfer or restart
format exists; changing the mesh and calling `IntegrationPointStorage::Reset`
would discard plastic history, so AMR must remain disabled for this model.

The postprocessing function `ProjectCommittedEquivalentPlasticStrain` writes a physical-volume-weighted
cell average to a discontinuous piecewise-constant scalar field. The material,
all MFEM parameter coefficients, stress-free deformation objects, custom
integration rule, and point storage are borrowed and must outlive the
integrator. Use separate point-storage objects for independent instances of
`J2PlasticityMaterial`. The mutable scratch and storage cursor are host-only and
not thread-safe.

Incremental energy is deliberately unavailable. `GetElementEnergy()` aborts
rather than returning a value that could incorrectly be used for globalization.

## Tensile example

`j2_tensile` reads the two-dimensional rectangular bar in
`data/simple_bar.msh`. Both displacement components are fixed at the bottom,
the top remains horizontally free, and its vertical displacement is prescribed.
A second stage reduces the prescribed displacement to `--unload-fraction`
times its maximum; the default fraction is three quarters. For each accepted
increment, the program prints a labeled status line with the stage, stage
factor, global pseudo-time, prescribed top displacement, and maximum
cell-averaged equivalent plastic strain. It verifies the average top
displacement against the prescribed value without printing the redundant field.
It optionally writes displacement and equivalent plastic strain for ParaView.

The continuation coordinate is dimensionless pseudo-time, not physical time.
For local stage coordinate $s\in[0,1]$, loading uses global pseudo-time
$\tau=s$ and prescribed displacement $\bar u_y=s\,u_{\max}$. Unloading uses
$\tau=1+s$ and
$\bar u_y=(1-s)u_{\max}+s f_{\mathrm{unload}}u_{\max}$. The `--load-steps`
value sets the maximum pseudo-time increment to $1/N$ in each stage. A
converged increment commits J2 history; a failed increment restores the last
accepted displacement and history, halves the increment, and retries. Because
the constitutive law is rate-independent, pseudo-time only orders accepted
states and carries no duration or rate information.

`MultiNewtonAdaptive<NewtonLineSearch>` advances the two pseudo-time intervals.
Its trial-state callback evaluates the prescribed displacement at the proposed
global pseudo-time before each Newton solve. A cutback invokes the callback
again at the reduced pseudo-time. The solver's legacy `lambda` accessors refer
to this pseudo-time continuation coordinate; `eta` is instead the line-search
scale applied to a Newton correction.

```bash
(cd build/debug/bin && ./j2_tensile -r 1 -lr 1 -steps 20 -disp 0.01 -no-output)
```

The `-r` option applies uniform refinement. The `-lr` option then applies local
refinement to elements touching the horizontal centerline of the bar.

The bar first responds elastically, then yields and hardens, and finally unloads
while retaining accumulated plastic strain. Sufficient reverse displacement
can produce reverse yielding even though the hardening law is isotropic. The
example remains a compact constitutive demonstration rather than a localization
or necking benchmark.

The example's absolute nonlinear tolerance scales with an elastic force scale
and `mfem::real_t` precision. This avoids requiring residuals below
the assembly noise floor in single precision while retaining a relative
tolerance and a `1e-8` absolute floor in double precision.

For each Newton tangent system, the serial example uses MFEM's UMFPACK sparse
direct solver when SuiteSparse is available. MFEM configurations without
SuiteSparse fall back to CG with diagonal smoothing; the associative J2
consistent tangent is symmetric.

## Verification

`tests/j2_plasticity_test.cpp` checks:

- zero and hydrostatic elastic states;
- engineering-shear scaling and the three-dimensional J2 invariant;
- perfect-plastic and linear-hardening consistency;
- invalid parameter and history rejection;
- proportional monotonic loading with different increment subdivisions;
- plane-strain out-of-plane stress and plastic flow;
- the analytic tangent against centered differences and an independent
  `autodiff` differentiation of the plastic update;
- deterministic trial evaluation, commit, and rollback behavior;
- two- and three-dimensional element residual/Jacobian directional derivatives;
- committed-history projection and incompatible-mesh rejection;
- equivalent residuals for global `byNODES` and `byVDIM` orderings;
- transaction-wide rejection and failed-Newton solution restoration.

The `J2Tensile.Smoke` CTest runs the plane-strain example on a two-element mesh
through plastic loading and unloading, requires reported active plasticity with
an elastic unload, and does not create output files.

## Provenance

The constitutive equations and radial-return algorithm follow the standard
associative J2 framework described by the references below. The displayed
specialization and consistent-tangent formula were derived for this
repository's definitions of $\Delta\gamma$, $\alpha$, and engineering Voigt
strain, then checked numerically; they are not represented as verbatim copied
equations.

1. Simo, J. C., and Taylor, R. L. (1985), “Consistent tangent operators for
   rate-independent elastoplasticity,” *Computer Methods in Applied Mechanics
   and Engineering*, 48(1), 101–118.
   [doi:10.1016/0045-7825(85)90070-2](https://doi.org/10.1016/0045-7825(85)90070-2),
   [open repository copy](https://escholarship.org/uc/item/9cp19009). Section 2,
   Eqs. (2.4)–(2.6), p. 103 defines associative J2 plasticity and hardening;
   Section 3, Eqs. (3.2)–(3.8), pp. 105–106 gives the endpoint radial return and
   scalar consistency equation; Section 4, Eqs. (4.5) and (4.12)–(4.13),
   pp. 108–110 defines and derives the consistent tangent of the discrete
   update. Specializing those equations to zero backstress and linear isotropic
   hardening, with the paper's multiplier
   $\lambda=\sqrt{3/2}\,\Delta\gamma$, gives this repository's $3G+H$
   denominator and tensor tangent. The engineering-Voigt conversion is a
   repository-specific derivation checked by the tests.
2. Simo, J. C., and Hughes, T. J. R. (1998), *Computational Inelasticity*,
   Interdisciplinary Applied Mathematics 7, Springer-Verlag, New York.
   [doi:10.1007/b98904](https://doi.org/10.1007/b98904). This is background for
   the classical small-strain associative-plasticity and return-mapping
   framework; no book equation is transcribed verbatim here.
3. de Souza Neto, E. A., Perić, D., and Owen, D. R. J. (2008),
   *Computational Methods for Plasticity: Theory and Applications*, Wiley.
   [doi:10.1002/9780470694626](https://doi.org/10.1002/9780470694626). This is
   background for the standard radial-return implementation structure; no book
   equation is transcribed verbatim here.
