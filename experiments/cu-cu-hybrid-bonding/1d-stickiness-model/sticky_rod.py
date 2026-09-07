"""Experimental elastic rod, rigid unilateral contact, and reversible bonds.

The notebook is the interface; importing or simulating writes no files and does
not import Matplotlib. Defaults use arbitrary, consistent force/length/time
units, not calibrated material data. This is a small-strain, quasistatic, scalar
model, not a copper bonding prediction. E and pressure have stress units, A has
area units, L and U have length units, K_n has stress/length units, and k_f/k_d
have inverse-time units. Small strain remains a separate modeling assumption;
the inversion check alone does not establish its validity.

The prescribed equations are condensed directly (no contact penalty):
    k_r = EA/L, k_b = A K_n, r = k_b/k_r,
    g = max(-U, 0)/(1 + r beta), u = -g,
    P = k_r max(U, 0), F_b = beta k_b g, N = F_b - P.
Thus k_r(u-U) = N, P >= 0, g >= 0, and P g = 0. Compression
forms bonds at k_f(P/A)/p_ref. The surviving-bond traction K_n*g activates
the cubic loss switch; nominal traction F_b/A remains the measured output.
This assumes uniform parallel bonds and is a stress-scaled gap law for this
linear spring. There is no power law, latch, cutoff, or extra damage state.
An activated fixed-actuator hold continues losing bonds as nominal force falls.

The scalar residual and derivative below are direct differentiations of these
prescribed equations, not formulas attributed to an external material model.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


_MAX_OUTPUT_POINTS = 200_000
_MAX_TRIALS = 2_000_000
_MAX_SWEEP_CASES = 128
_PHASES = ("approach", "press", "hold", "unload", "pull", "open_hold")
_NUMERIC_ARRAYS = (
    "t", "U", "u", "gap", "P", "F_b", "N", "pressure", "traction",
    "bond_traction", "beta", "k_on", "k_off",
)


def _finite(value: Real, name: str) -> float:
    try:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real scalar, not a boolean.")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite.")
        return result
    except (OverflowError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc


@dataclass(frozen=True)
class Parameters:
    E: float = 100.0
    A: float = 1.0
    L: float = 1.0
    K_n: float = 10.0
    k_f: float = 1.0
    p_ref: float = 0.1
    k_d: float = 1.0
    bond_sigma0: float = 0.04
    bond_delta_sigma: float = 0.02
    beta0: float = 0.0
    p_peak: float = 0.1
    t_approach: float = 1.0
    t_press: float = 1.0
    t_hold: float = 3.0
    t_unload: float = 1.0
    pull_distance: float = 0.12
    v_pull: float = 0.04
    t_open_hold: float = 3.0
    cycles: int = 1
    dt: float = 0.03
    rtol: float = 1.0e-5
    atol: float = 1.0e-9

    def __post_init__(self) -> None:
        self.validate()

    @property
    def k_r(self) -> float:
        return self.E * self.A / self.L

    @property
    def k_b(self) -> float:
        return self.A * self.K_n

    @property
    def U_press(self) -> float:
        return (self.p_peak / self.E) * self.L

    @property
    def t_pull(self) -> float:
        return self.pull_distance / self.v_pull

    def validate(self) -> None:
        for field in fields(self):
            _finite(getattr(self, field.name), field.name)
        if isinstance(self.cycles, (bool, np.bool_)) or not isinstance(self.cycles, Integral):
            raise ValueError("cycles must be an actual positive integer (not bool or float).")
        if not 1 <= self.cycles <= (_MAX_OUTPUT_POINTS - 1) // 4:
            raise ValueError("cycles must be positive and fit the output resource limit.")
        positive = (
            "E", "A", "L", "p_ref", "bond_delta_sigma", "t_approach", "t_press",
            "t_unload", "pull_distance", "v_pull", "dt", "rtol", "atol",
        )
        nonnegative = ("K_n", "k_f", "k_d", "bond_sigma0", "p_peak", "t_hold", "t_open_hold")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive.")
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if not 0 <= self.beta0 <= 1:
            raise ValueError("beta0 must lie in [0, 1].")
        if self.p_peak >= self.E:
            raise ValueError("p_peak must be less than E to avoid rod inversion.")
        try:
            if not math.isfinite(self.k_r) or self.k_r <= 0:
                raise ValueError("Derived k_r must be finite and positive.")
            derived = {
                "k_b": self.k_b,
                "k_b/k_r": self.k_b / self.k_r,
                "U_press": self.U_press,
                "t_pull": self.t_pull,
                "switch upper threshold": self.bond_sigma0 + self.bond_delta_sigma,
                "switch slope": 1.5 / self.bond_delta_sigma,
                "off-rate slope": self.k_d * (1.5 / self.bond_delta_sigma),
                "traction scale": self.K_n * self.pull_distance,
                "adhesive force scale": self.k_b * self.pull_distance,
                "rod force scale": self.k_r * max(self.U_press, self.pull_distance),
                "pressure scale": self.k_r * self.U_press / self.A,
                "formation rate": self.k_f * (self.p_peak / self.p_ref),
                "pressure dose": self.p_peak * (0.5 * self.t_press + self.t_hold + 0.5 * self.t_unload),
                "rod extent": self.L + self.pull_distance,
                "approach speed": self.pull_distance / self.t_approach,
                "press speed": self.U_press / self.t_press,
                "unload speed": self.U_press / self.t_unload,
                "error scale": self.atol + self.rtol,
            }
            for name, value in derived.items():
                _finite(value, f"Derived {name}")
            if self.t_pull <= 0 or (self.p_peak > 0 and self.U_press <= 0):
                raise ValueError("Derived duration/displacement underflows to zero.")
            if self.K_n > 0 and self.k_b <= 0:
                raise ValueError("Derived k_b underflows to zero.")
            if self.U_press >= self.L:
                raise ValueError("Compressed rod length is not representably positive.")
            if self.bond_sigma0 + self.bond_delta_sigma <= self.bond_sigma0:
                raise ValueError("Switch interval is not representable at bond_sigma0.")
            load_segments(self)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Derived parameters overflow or are not representable.") from exc


def mechanics(U: float, beta: float, p: Parameters) -> tuple[float, float, float, float, float]:
    """Return (u, gap, contact force P, adhesive force F_b, tensile rod force N)."""
    U, beta = _finite(U, "U"), _finite(beta, "beta")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1].")
    if U >= p.L:
        raise ValueError("U must be less than L to avoid rod inversion.")
    gap = max(-U, 0.0) / (1.0 + beta * (p.k_b / p.k_r))
    P = p.k_r * max(U, 0.0)
    Fb = (beta * p.k_b) * gap
    result = (-gap, gap, P, Fb, Fb - P)
    if not all(math.isfinite(x) for x in (*result, P / p.A, Fb / p.A, p.K_n * gap)):
        raise ValueError("Mechanical forces or stresses overflow at this U and beta.")
    return result


def switch(stress: float, p: Parameters) -> tuple[float, float]:
    """C1 cubic S and dS/dstress; stress is bonded-area traction K_n*gap."""
    stress = _finite(stress, "stress")
    if stress <= p.bond_sigma0:
        return 0.0, 0.0
    if stress >= p.bond_sigma0 + p.bond_delta_sigma:
        return 1.0, 0.0
    x = (stress - p.bond_sigma0) / p.bond_delta_sigma
    return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x) / p.bond_delta_sigma


def off_rate(stress: float, p: Parameters) -> tuple[float, float]:
    """Return k_d S(bond traction) and its bonded-area stress derivative."""
    S, derivative = switch(stress, p)
    return p.k_d * S, p.k_d * derivative


class _ScalarSolveError(RuntimeError):
    """The uniqueness bound or scalar solve failed; no state was accepted."""


def backward_euler(beta_old: float, U_new: float, h: float, p: Parameters) -> float:
    """Bound-preserving BE with a sufficient opening-step uniqueness bound.

    Opening: tau=K_n*(-U)/(1+r*beta), R=beta-beta_old+h*b(tau)*beta.
    R'=1+h*b-h*beta*b_tau*r*tau/(1+r*beta) can become negative.
    In the switch interval b_tau <= 1.5*k_d/bond_delta_sigma and
    tau <= min(K_n*(-U), bond_sigma0+bond_delta_sigma). Also
    r*beta/(1+r*beta) <= r*beta_old/(1+r*beta_old). Bounding their
    product by B and requiring h*B <= 0.8 ensures R' >= 0.2.
    Solve q=beta/beta_old on [0,1] without clipping; stationary and
    fully active branches have closed forms. See model.pdf for the derivation.
    """
    beta_old, h = _finite(beta_old, "beta_old"), _finite(h, "h")
    if h <= 0:
        raise ValueError("h must be strictly positive.")
    mechanics(U_new, beta_old, p)
    if U_new >= 0:
        # Cancel area analytically in the stress-driven kinetics.
        a = p.k_f * ((p.E * (U_new / p.L)) / p.p_ref)
        c = _finite(h * a, "h * formation rate")
        return (beta_old + c) / (1.0 + c)
    if beta_old == 0 or p.k_d == 0:
        return beta_old
    r = p.k_b / p.k_r
    stress_scale = _finite(p.K_n * (-U_new), "opening traction scale")
    old_traction = stress_scale / (1.0 + r * beta_old)
    b_old = off_rate(old_traction, p)[0]
    if b_old == 0:
        return beta_old
    _finite(h * p.k_d, "h * k_d")
    if old_traction >= p.bond_sigma0 + p.bond_delta_sigma:
        # Traction increases as beta decreases, so the whole decay bracket is active.
        return beta_old / (1.0 + h * p.k_d)
    ratio = r * beta_old / (1.0 + r * beta_old)
    upper_traction = min(stress_scale, p.bond_sigma0 + p.bond_delta_sigma)
    bound = ratio * upper_traction * (p.k_d * (1.5 / p.bond_delta_sigma))
    if not math.isfinite(bound) or h * bound > 0.8:
        raise _ScalarSolveError("Refine the opening step to ensure a unique scalar root.")
    lo, hi = 0.0, 1.0
    q = 1.0 / (1.0 + h * b_old)
    eps = np.finfo(float).eps
    for _ in range(200):
        beta = beta_old * q
        denominator = 1.0 + r * beta
        stress = stress_scale / denominator
        b, bp = off_rate(stress, p)
        residual = q * (1.0 + h * b) - 1.0
        if abs(residual) <= 32.0 * eps:
            return beta
        if residual > 0:
            hi = q
        else:
            lo = q
        if hi - lo <= 8.0 * eps * max(hi, np.finfo(float).tiny):
            # A converged relative root bracket, not a clipped Newton iterate.
            return beta_old * (lo + 0.5 * (hi - lo))
        dstress_dq = -(r * beta_old / denominator) * stress
        derivative = _finite(1.0 + h * b + h * bp * (q * dstress_dq), "BE derivative")
        if derivative <= 0:
            raise _ScalarSolveError("Opening residual lost monotonicity; reduce the time step.")
        trial = q - residual / derivative
        width = hi - lo
        q = trial if lo + 0.1 * width < trial < hi - 0.1 * width else lo + 0.5 * width
    raise _ScalarSolveError("Opening BE solve did not converge; reduce the time step.")


def load_segments(p: Parameters) -> list[dict[str, Any]]:
    """Continuous six-stage cycles, omitting only zero-duration holds.

    Schedules are limited to 200,000 output points. Reject durations/output
    spacings that cannot support representable half steps at absolute time;
    simulation never silently skips a small remainder.
    """
    stages = (
        ("approach", p.t_approach, -p.pull_distance, 0.0),
        ("press", p.t_press, 0.0, p.U_press),
        ("hold", p.t_hold, p.U_press, p.U_press),
        ("unload", p.t_unload, p.U_press, 0.0),
        ("pull", p.t_pull, 0.0, -p.pull_distance),
        ("open_hold", p.t_open_hold, -p.pull_distance, -p.pull_distance),
    )
    result = []
    start, points = 0.0, 1
    for cycle in range(1, p.cycles + 1):
        for phase, duration, U_start, U_end in stages:
            if duration == 0:
                continue
            end = start + duration
            if not math.isfinite(end) or end <= start:
                raise ValueError(f"{phase} duration is not representable at time {start:g}.")
            count_float = duration / p.dt
            if not math.isfinite(count_float) or count_float > _MAX_OUTPUT_POINTS:
                raise ValueError("Schedule exceeds the 200,000 output-point resource limit.")
            count = max(1, math.ceil(count_float))
            points += count
            if points > _MAX_OUTPUT_POINTS:
                raise ValueError("Schedule exceeds the 200,000 output-point resource limit.")
            if duration / count <= 4.0 * math.ulp(end):
                raise ValueError(f"{phase} output half steps are not representable at time {end:g}.")
            result.append(dict(cycle=cycle, phase=phase, start=start, end=end,
                               duration=duration, U_start=U_start, U_end=U_end))
            start = end
    return result


def simulate(p: Parameters) -> dict[str, Any]:
    """Adaptive BE full versus two half steps; accept two halves, never extrapolate.

    dt bounds output spacing and internal steps up to time-roundoff. Every
    segment endpoint is an exact output time. A shared boundary is labeled by
    its preceding segment; consumers must use inclusive time masks, not labels alone.
    Tolerances control local beta error, not global accuracy or peak resolution.
    """
    segments = load_segments(p)
    times, displacements, betas = [0.0], [-p.pull_distance], [p.beta0]
    cycles, phases = [1], ["approach"]
    state = p.beta0
    accepted = rejected = trials = 0
    for segment in segments:
        start, end = segment["start"], segment["end"]
        # Integrate in phase-local time so shifting a cycle in time does not
        # change adaptive step decisions through cancellation in end-start.
        duration = segment["duration"]
        count = max(1, math.ceil(duration / p.dt))
        grid = np.linspace(0.0, duration, count + 1)
        h_next = p.dt
        now = 0.0
        for output_time in grid[1:]:
            output_time = float(output_time)
            while now < output_time:
                if trials >= _MAX_TRIALS:
                    raise RuntimeError("Adaptive BE exhausted the 2,000,000-trial resource limit.")
                target = min(output_time, now + h_next)
                if target > now and output_time - target <= 4.0 * math.ulp(output_time):
                    # Integrate the endpoint sliver with this trial instead of
                    # leaving an unsplittable remainder or skipping its kinetics.
                    target = output_time
                h = target - now
                middle = now + 0.5 * h
                if not now < middle < target:
                    raise RuntimeError("Required BE half step is below the representable time floor.")
                Umid, Uend = np.interp(
                    [middle, target], [0.0, duration], [segment["U_start"], segment["U_end"]]
                )
                trials += 1
                try:
                    full = backward_euler(state, float(Uend), h, p)
                    half = backward_euler(state, float(Umid), middle - now, p)
                    two = backward_euler(half, float(Uend), target - middle, p)
                except _ScalarSolveError:
                    rejected += 1
                    h_next = 0.5 * h
                    continue
                scale = p.atol + p.rtol * max(state, two)
                error = abs(two - full) / scale
                if error > 1.0:
                    rejected += 1
                    h_next = h * max(0.1, 0.85 / math.sqrt(error))
                    continue
                now, state = target, two
                accepted += 1
                factor = 2.0 if error == 0 else min(2.0, max(0.5, 0.9 / math.sqrt(error)))
                h_next = min(p.dt, h * factor)
            times.append(end if output_time == duration else start + output_time)
            displacements.append(float(np.interp(
                output_time, [0.0, duration], [segment["U_start"], segment["U_end"]]
            )))
            betas.append(state)
            cycles.append(segment["cycle"])
            phases.append(segment["phase"])
    t, U, beta = (np.asarray(x, dtype=float) for x in (times, displacements, betas))
    u, gap, P, Fb, N = np.asarray([mechanics(v, b, p) for v, b in zip(U, beta)]).T
    pressure, traction = P / p.A, Fb / p.A
    bond_traction = p.K_n * gap
    k_on = p.k_f * (pressure / p.p_ref)
    k_off = np.asarray([off_rate(stress, p)[0] for stress in bond_traction])
    return dict(t=t, U=U, u=u, gap=gap, P=P, F_b=Fb, N=N, pressure=pressure,
                traction=traction, bond_traction=bond_traction, beta=beta, k_on=k_on, k_off=k_off,
                cycle=np.asarray(cycles, dtype=int), phase=np.asarray(phases),
                stats=dict(accepted_intervals=accepted, rejected_trials=rejected, trials=trials),
                segments=segments)


def _validate_solution(sol: dict[str, Any], p: Parameters) -> None:
    """Validate notebook-provided histories before indexing or interpolation."""
    t = sol.get("t")
    if not isinstance(t, np.ndarray) or t.ndim != 1 or len(t) < 2:
        raise ValueError("Solution t must be a one-dimensional array with at least two samples.")
    for name in (*_NUMERIC_ARRAYS, "cycle", "phase"):
        array = sol.get(name)
        if not isinstance(array, np.ndarray) or array.shape != t.shape:
            raise ValueError(f"Solution {name} must be an array with the same shape as t.")
        if name != "phase":
            if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
                raise ValueError(f"Solution {name} must contain finite real numbers.")
    if t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Solution times must start at zero and strictly increase.")
    if np.any((sol["beta"] < 0) | (sol["beta"] > 1)):
        raise ValueError("Solution beta must lie in [0, 1].")
    for name in ("gap", "P", "F_b", "pressure", "traction", "bond_traction", "k_on", "k_off"):
        if np.any(sol[name] < 0):
            raise ValueError(f"Solution {name} must be nonnegative.")
    if sol["cycle"].dtype.kind not in "iu" or np.any((sol["cycle"] < 1) | (sol["cycle"] > p.cycles)):
        raise ValueError("Solution cycle labels must be integer cycle numbers.")
    if not np.all(np.isin(sol["phase"], _PHASES)):
        raise ValueError("Solution contains an unknown phase label.")
    segments = load_segments(p)
    if sol.get("segments") != segments or t[-1] != segments[-1]["end"]:
        raise ValueError("Solution schedule does not match Parameters.")
    for segment in segments:
        index = np.searchsorted(t, segment["end"])
        if index == len(t) or t[index] != segment["end"]:
            raise ValueError("Solution must include every exact segment endpoint.")


def summarize_cycles(sol: dict[str, Any], p: Parameters) -> list[dict[str, Any]]:
    """Flat per-cycle rows; stress peaks exclude the next tensile approach.

    pressure_dose is integral(pressure dt), in stress*time units. Balance error
    has force units, complementarity error force*length units. A positive peak
    attained at the pull endpoint is flagged as potentially range-limited.
    """
    _validate_solution(sol, p)
    t, beta, stress = sol["t"], sol["beta"], sol["traction"]
    rows = []
    for cycle in range(1, p.cycles + 1):
        segments = [s for s in sol["segments"] if s["cycle"] == cycle]
        pull = next(s for s in segments if s["phase"] == "pull")
        unload = next(s for s in segments if s["phase"] == "unload")
        start, end = segments[0]["start"], segments[-1]["end"]
        mask = (t >= start) & (t <= end)
        peak_indices = np.flatnonzero((t >= pull["start"]) & (t <= end))
        peak = int(peak_indices[np.argmax(stress[peak_indices])])
        pull_end = int(np.searchsorted(t, pull["end"]))
        end_index = int(np.searchsorted(t, end))
        balance = p.k_r * (sol["u"][mask] - sol["U"][mask]) - sol["N"][mask]
        pressure = sol["pressure"][mask]
        dose = np.dot(np.diff(t[mask]), 0.5 * pressure[:-1] + 0.5 * pressure[1:])
        rows.append(dict(
            cycle=cycle, p_peak=p.p_peak, t_hold=p.t_hold,
            beta_start=float(beta[np.searchsorted(t, start)]),
            beta_hold_end=float(beta[np.searchsorted(t, unload["start"])]),
            beta_pull_start=float(beta[np.searchsorted(t, pull["start"])]),
            peak_stress=float(stress[peak]), peak_time=float(t[peak]),
            beta_end=float(beta[end_index]), end_stress=float(stress[end_index]),
            peak_at_pull_end=bool(stress[peak] > 0 and stress[pull_end] == stress[peak]),
            pressure_dose=_finite(dose, "Integrated pressure dose"),
            max_force_balance_error=float(np.max(np.abs(balance))),
            max_complementarity_error=float(np.max(np.abs(sol["P"][mask] * sol["gap"][mask]))),
        ))
    return rows


def sweep(p: Parameters, field: str, values: Any) -> list[dict[str, Any]]:
    """Independent pressure/hold cases; beta carries within, not between, cases."""
    if field not in ("p_peak", "t_hold"):
        raise ValueError("Sweep field must be p_peak or t_hold.")
    try:
        array = np.asarray(values)
        if array.dtype.kind not in "fiu":
            raise ValueError("Sweep values must be real numbers, not strings, booleans, or complex values.")
        array = array.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Sweep values must be a finite one-dimensional numeric array.") from exc
    if array.ndim != 1 or not 1 <= array.size <= _MAX_SWEEP_CASES or not np.all(np.isfinite(array)):
        raise ValueError("Sweep values must contain 1 to 128 finite scalar values.")
    # Validate every case before doing any simulations.
    cases = [replace(p, **{field: float(value)}) for value in array]
    return [row for case in cases for row in summarize_cycles(simulate(case), case)]


def plot_history(sol: dict[str, Any], p: Parameters) -> Any:
    """Return one 2x2 notebook Figure; no global style changes or file output."""
    rows = summarize_cycles(sol, p)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    motion, bonds, loads, opening = axes.flat
    t = sol["t"]
    motion.plot(t, sol["U"], label="Prescribed U")
    motion.plot(t, sol["u"], label="Tip u = -gap")
    motion.set(title="Motion", xlabel="Time", ylabel="Displacement [length]")
    bonds.plot(t, sol["beta"], color="tab:purple", label="Active bond fraction")
    bonds.set(title="Bond kinetics", xlabel="Time", ylabel=r"$\beta$", ylim=(-0.02, 1.02))
    loads.plot(t, sol["pressure"], label="Compressive pressure")
    loads.plot(t, sol["traction"], label="Nominal tensile traction")
    loads.plot(t, sol["beta"] * p.bond_sigma0, color="0.4", linestyle=":",
               label="Nominal onset = beta * bond onset")
    loads.set(title="Contact and adhesion", xlabel="Time", ylabel="Stress")
    for row in rows:
        segments = [s for s in sol["segments"] if s["cycle"] == row["cycle"]]
        mask = (t >= segments[0]["start"]) & (t <= segments[-1]["end"])
        line, = opening.plot(sol["gap"][mask], sol["traction"][mask], label=f"Cycle {row['cycle']}")
        index = np.searchsorted(t, row["peak_time"])
        opening.plot(sol["gap"][index], row["peak_stress"],
                     "s" if row["peak_at_pull_end"] else "o", color=line.get_color())
    opening.set(title="Cycle traces (squares: endpoint peaks)",
                 xlabel="Gap [length]", ylabel="Nominal tensile traction [stress]")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    fig.suptitle("Experimental rigid-contact rod | consistent arbitrary units")
    return fig


def plot_sweep(rows: list[dict[str, Any]], field: str) -> Any:
    """Plot per-cycle peak traction, flagging pull-end (possibly limited) peaks."""
    if field not in ("p_peak", "t_hold") or not rows:
        raise ValueError("Provide nonempty sweep rows and field p_peak or t_hold.")
    for row in rows:
        if not all(key in row for key in (field, "cycle", "peak_stress", "peak_at_pull_end")):
            raise ValueError("Sweep rows are missing required fields.")
        if _finite(row[field], field) < 0 or _finite(row["peak_stress"], "peak_stress") < 0:
            raise ValueError("Sweep values and peak stresses must be nonnegative.")
        if isinstance(row["cycle"], bool) or not isinstance(row["cycle"], Integral) or row["cycle"] < 1:
            raise ValueError("Sweep cycles must be positive integers.")
        if not isinstance(row["peak_at_pull_end"], (bool, np.bool_)):
            raise ValueError("peak_at_pull_end must be boolean.")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    marked = False
    for cycle in sorted({row["cycle"] for row in rows}):
        selected = sorted((row for row in rows if row["cycle"] == cycle), key=lambda row: row[field])
        x = np.asarray([row[field] for row in selected])
        y = np.asarray([row["peak_stress"] for row in selected])
        mask = np.asarray([row["peak_at_pull_end"] for row in selected], dtype=bool)
        line, = ax.plot(x, y, "o-", label=f"Cycle {cycle}")
        if np.any(mask):
            ax.plot(x[mask], y[mask], "s", markersize=9, markerfacecolor="none",
                    color=line.get_color(), label=None if marked else "Pull-end peak (possibly range-limited)")
            marked = True
    ax.set(xlabel="Peak pressure [stress]" if field == "p_peak" else "Compressed hold [time]",
           ylabel="Peak nominal tensile traction [stress]", title="Independent cases; continuous bonds within each case")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    return fig


def animate(sol: dict[str, Any], p: Parameters, frames: int = 480, fps: float = 20,
            *, displacement_scale: float = 1.0) -> Any:
    """Return FuncAnimation for to_jshtml(default_mode='once'); never save files.

    Frames use uniform physical times. Interpolate U/beta and re-solve mechanics
    at each frame, rather than interpolating inconsistent forces. Only displayed
    displacements are magnified: endpoints are (-L+s*U, s*u), s=displacement_scale.
    The dashed length-L guide is not another physical rod. Histories and numeric
    labels remain physical; adhesive opacity is beta, without a debond cutoff.
    Rod color represents uniform signed axial stress N/A, not bond traction.
    One symmetric color range covers the entire history and all rendered frames.
    """
    _validate_solution(sol, p)
    if isinstance(frames, (bool, np.bool_)) or not isinstance(frames, Integral) or not 2 <= frames <= 600:
        raise ValueError("frames must be an integer from 2 to 600.")
    fps = _finite(fps, "fps")
    if not 1 <= fps <= 60:
        raise ValueError("fps must lie in [1, 60].")
    displacement_scale = _finite(displacement_scale, "displacement_scale")
    if displacement_scale < 1:
        raise ValueError("displacement_scale must be at least 1 (actual geometry).")
    if displacement_scale * p.U_press >= p.L:
        raise ValueError("Magnification collapses the displayed compressed rod; reduce displacement_scale.")
    span = _finite(p.L + displacement_scale * p.pull_distance, "display extent")
    _finite(1.08 * span, "display axis limit")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch, Rectangle

    frame_times = np.linspace(float(sol["t"][0]), float(sol["t"][-1]), frames)
    frame_U = np.interp(frame_times, sol["t"], sol["U"])
    frame_beta = np.interp(frame_times, sol["t"], sol["beta"])
    frame_states = np.asarray([mechanics(U, beta, p) for U, beta in zip(frame_U, frame_beta)])
    stress_limit = _finite(max(float(np.max(np.abs(sol["N"] / p.A))),
                              float(np.max(np.abs(frame_states[:, 4] / p.A)))), "stress color range")
    # Zero-load runs need a nondegenerate display range, not a fabricated stress.
    stress_limit = stress_limit if stress_limit > 0 else p.p_ref
    _finite(2 * stress_limit, "stress color span")
    stress_colors = ScalarMappable(norm=Normalize(-stress_limit, stress_limit), cmap="coolwarm")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})
    scene, bonds, loads = axes
    scene.set(xlim=(-1.08 * span, 0.15 * span), ylim=(-0.75, 1.1),
              xlabel=f"Display coordinate [length]; displacements {displacement_scale:g}x",
              yticks=[], title=f"Rod and rigid wall | displacement magnification {displacement_scale:g}x (visual only)")
    scene.axvline(0, color="0.2", linewidth=3)
    scene.axvspan(0, 0.15 * span, color="0.9", hatch="//")
    rod = Rectangle((-span, -0.14), p.L, 0.28, facecolor=stress_colors.to_rgba(0),
                    edgecolor="0.2", linewidth=1)
    scene.add_patch(rod)
    colorbar = fig.colorbar(stress_colors, ax=scene, pad=0.02, fraction=0.045,
                           ticks=[-stress_limit, 0, stress_limit], format="%.3g")
    colorbar.set_label("Rod axial stress [stress]\n(tension +)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    reference = Rectangle((-span, -0.18), p.L, 0.36, fill=False, edgecolor="0.4",
                          linestyle="--", linewidth=1.2, zorder=4, label="Unstretched length L (guide)")
    scene.add_patch(reference)
    adhesive, = scene.plot([], [], color="tab:orange", linewidth=10, solid_capstyle="butt",
                            marker="o", markersize=7, label="Adhesion (opacity = beta)")
    scene.legend(handles=[reference, adhesive], loc="lower left", fontsize=7, ncol=2)
    actuator, = scene.plot([], [], "|", color="black", markersize=35, markeredgewidth=2)
    arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=14, color="tab:red")
    scene.add_patch(arrow)
    status = scene.text(0.02, 0.98, "", transform=scene.transAxes, va="top", family="monospace", fontsize=9)
    bond_line, = bonds.plot(sol["t"], sol["beta"], color="tab:purple", label="Active bonds")
    bonds.set(xlabel="Time", ylabel=r"$\beta$", ylim=(-0.02, 1.02))
    strain_axis = bonds.twinx()
    strain_line, = strain_axis.plot(sol["t"], 100 * (sol["u"] - sol["U"]) / p.L,
                                   color="tab:green", linestyle="--", label="Actual rod strain")
    strain_axis.set_ylabel("Actual rod strain [%]", color="tab:green")
    strain_axis.tick_params(axis="y", colors="tab:green")
    loads.plot(sol["t"], sol["pressure"], label="Pressure")
    loads.plot(sol["t"], sol["traction"], label="Nominal traction")
    loads.plot(sol["t"], sol["beta"] * p.bond_sigma0, color="0.4", linestyle=":",
               label="Nominal onset = beta * bond onset")
    loads.set(xlabel="Time", ylabel="Stress")
    cursors = [ax.axvline(0, color="black", linewidth=1) for ax in (bonds, loads)]
    for ax in (bonds, loads):
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize="small")
    bonds.legend(handles=[bond_line, strain_line], loc="upper right", fontsize="small")
    ends = np.asarray([s["end"] for s in sol["segments"]])

    def update(index: int) -> tuple[Any, ...]:
        time = float(frame_times[index])
        U, beta = float(frame_U[index]), float(frame_beta[index])
        u, gap, P, Fb, N = frame_states[index]
        left = -p.L + displacement_scale * U
        tip = displacement_scale * u
        rod.set_x(left)
        rod.set_width(tip - left)
        rod.set_facecolor(stress_colors.to_rgba(N / p.A))
        reference.set_x(left)
        actuator.set_data([left], [0])
        adhesive.set_data([tip, 0], [0, 0])
        adhesive.set_alpha(beta)
        # Arrow direction shows the signed force at the right tip, not a scale.
        arrow.set_visible(N != 0)
        arrow.set_positions((tip, -0.32), (tip + math.copysign(0.08 * span, N), -0.32))
        segment = sol["segments"][min(int(np.searchsorted(ends, time, side="left")), len(ends) - 1)]
        bond_traction = p.K_n * gap
        loss = "bond loss active" if beta > 0 and off_rate(bond_traction, p)[0] > 0 else "no bond loss"
        status.set_text(f"Cycle {segment['cycle']} | {segment['phase']} | t={time:.5g}\n"
                         f"gap={gap:.5g}   beta={beta:.5g}   pressure={P / p.A:.5g}\n"
                         f"nominal_traction={Fb / p.A:.5g}   bond={bond_traction:.5g}   bond onset={p.bond_sigma0:g}\n"
                         f"actual strain={100 * (u - U) / p.L:.3f}% | {loss}")
        for cursor in cursors:
            cursor.set_xdata([time, time])
        return rod, adhesive, actuator, arrow, status, reference, *cursors

    update(0)
    return FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps,
                         blit=False, repeat=False)
