"""Deterministic regression checks for the experimental notebook model.

Run from the repository root:
    experiments/.venv/bin/python -m unittest discover \
        -s experiments/cu-cu-hybrid-bonding/1d-stickiness-model -p test_sticky_rod.py
"""
from dataclasses import FrozenInstanceError, fields, replace
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import sticky_rod as rod
from sticky_rod import (
    Parameters, animate, backward_euler, load_segments, mechanics, off_rate,
    plot_history, plot_sweep, simulate, summarize_cycles, sweep, switch,
)


class ScalarModelTests(unittest.TestCase):
    def test_parameter_contract(self):
        p = Parameters()
        self.assertEqual(len(fields(p)), 22)
        self.assertEqual((p.k_r, p.k_b, p.U_press, p.t_pull), (100.0, 10.0, 0.001, 3.0))
        self.assertEqual((p.beta0, p.cycles, p.dt, p.rtol, p.atol), (0.0, 1, 0.03, 1e-5, 1e-9))
        with self.assertRaises(FrozenInstanceError):
            p.beta0 = 0.5

    def test_switch_endpoints_and_centered_derivative(self):
        p = Parameters(k_d=2.3)
        for stress in (-1.0, 0.0, p.bond_sigma0):
            self.assertEqual(switch(stress, p), (0.0, 0.0))
            self.assertEqual(off_rate(stress, p), (0.0, 0.0))
        for stress in (p.bond_sigma0 + p.bond_delta_sigma, 10.0):
            self.assertEqual(switch(stress, p), (1.0, 0.0))
            self.assertEqual(off_rate(stress, p), (p.k_d, 0.0))
        for fraction in (0.1, 0.3, 0.5, 0.8, 0.95):
            stress = p.bond_sigma0 + fraction * p.bond_delta_sigma
            for function in (switch, off_rate):
                h = 1e-6 * p.bond_delta_sigma
                numerical = (function(stress + h, p)[0] - function(stress - h, p)[0]) / (2 * h)
                self.assertAlmostEqual(numerical / function(stress, p)[1], 1.0, delta=2e-9)
        h = 1e-8 * p.bond_delta_sigma
        self.assertLess(switch(p.bond_sigma0 + h, p)[1], 1e-4)
        self.assertLess(switch(p.bond_sigma0 + p.bond_delta_sigma - h, p)[1], 1e-4)

    def test_mechanics_equilibrium_and_complementarity(self):
        for area in (0.2, 1.0, 7.0):
            p = Parameters(A=area)
            for U in (-0.8, -0.12, -0.003, 0.0, 0.001, 0.2):
                for beta in (0.0, 1e-20, 0.3, 1.0):
                    u, gap, P, Fb, N = mechanics(U, beta, p)
                    self.assertEqual(u, -gap)
                    self.assertGreaterEqual(gap, 0)
                    self.assertGreaterEqual(P, 0)
                    self.assertEqual(P * gap, 0)
                    self.assertAlmostEqual(p.k_r * (u - U), N, delta=1e-12)
                    self.assertEqual(Fb - P, N)
                    self.assertEqual(Fb / area, max(N / area, 0))
                    self.assertAlmostEqual(Fb / area, beta * p.K_n * gap, delta=2e-15)

    def test_compression_closed_form(self):
        p = Parameters(A=3.2)
        for old in (0.0, 0.1, 0.9, 1.0):
            for h in (1e-9, 0.02, 5.0, 1e8):
                rate = p.k_f * p.p_peak / p.p_ref
                expected = (old + h * rate) / (1 + h * rate)
                self.assertAlmostEqual(backward_euler(old, p.U_press, h, p), expected, delta=2e-16)
            self.assertEqual(backward_euler(old, 0, 1.0, p), old)

    def test_opening_residual_and_centered_derivative(self):
        p = Parameters()
        U, old = -0.0053, 0.8
        r = p.k_b / p.k_r
        for h in (1e-6, 0.03, 0.5):
            def residual(beta):
                stress = p.K_n * mechanics(U, beta, p)[1]
                return beta - old + h * off_rate(stress, p)[0] * beta

            new = backward_euler(old, U, h, p)
            self.assertGreater(new, 0)
            self.assertLessEqual(new, old)
            self.assertAlmostEqual(residual(new), 0, delta=2e-13)
            for beta in (0.05, 0.25, 0.29, 0.7):
                b, bp = off_rate(p.K_n * mechanics(U, beta, p)[1], p)
                analytical = 1 + h * b - h * beta * bp * r * p.K_n * (-U) / (1 + r * beta)**2
                eps = 1e-7
                numerical = (residual(beta + eps) - residual(beta - eps)) / (2 * eps)
                self.assertGreaterEqual(analytical, 0.2)
                self.assertAlmostEqual(numerical / analytical, 1, delta=1e-8)

    def test_tiny_beta_uses_relative_root(self):
        p = Parameters()
        old, U, h = 1e-180, -0.12, 4.0
        new = backward_euler(old, U, h, p)
        b = off_rate(p.K_n * mechanics(U, new, p)[1], p)[0]
        self.assertGreater(new, 0)
        self.assertLess(new, old)
        self.assertAlmostEqual((new / old) * (1 + h * b), 1, delta=2e-13)
        self.assertEqual(backward_euler(0, U, h, p), 0)

    def test_falling_nominal_traction_does_not_stop_activated_loss(self):
        p = Parameters()
        U, old = -0.12, 0.8
        previous = old
        for h in (100.0, 1e6, 1e10):
            new = backward_euler(old, U, h, p)
            self.assertGreater(new, 0)
            self.assertLess(new, previous)
            _, gap, _, Fb, _ = mechanics(U, new, p)
            self.assertLess(Fb / p.A, p.bond_sigma0)
            self.assertEqual(off_rate(p.K_n * gap, p)[0], p.k_d)
            self.assertAlmostEqual(new / old * (1 + h * p.k_d), 1, delta=2e-15)
            previous = new
        self.assertLess(previous, 1e-9)
        # Closing unloads surviving bonds. There is no permanent activation flag.
        closed_U = -p.bond_sigma0 / (2 * p.K_n)
        self.assertEqual(backward_euler(previous, closed_U, 100, p), previous)
        self.assertGreater(backward_euler(previous, p.U_press, 1, p), previous)

    def test_opening_uniqueness_guard_and_stationary_branch(self):
        p = Parameters(E=1, K_n=100, bond_sigma0=0.01, bond_delta_sigma=0.001)
        old = 0.8
        r = p.k_b / p.k_r
        U = -0.0104 * (1 + r * old) / p.K_n
        b, bp = off_rate(0.0104, p)
        derivative_at_old = 1 + b - old * bp * r * 0.0104 / (1 + r * old)
        self.assertLess(derivative_at_old, 0)
        with self.assertRaisesRegex(rod._ScalarSolveError, "unique scalar root"):
            backward_euler(old, U, 1, p)
        h = 0.01
        new = backward_euler(old, U, h, p)
        b = off_rate(p.K_n * mechanics(U, new, p)[1], p)[0]
        self.assertAlmostEqual(new - old + h * b * new, 0, delta=2e-13)
        # Large discrete steps must not nucleate loss from a stationary ODE state.
        U = -0.009 * (1 + r * old) / p.K_n
        self.assertEqual(backward_euler(old, U, 1e6, p), old)

    def test_invalid_parameters(self):
        p = Parameters()
        for field in fields(p):
            for value in (math.nan, math.inf, -math.inf, True, "1"):
                with self.subTest(field=field.name, value=value), self.assertRaises(ValueError):
                    replace(p, **{field.name: value})
        positive = (
            "E", "A", "L", "p_ref", "bond_delta_sigma", "t_approach", "t_press",
            "t_unload", "pull_distance", "v_pull", "dt", "rtol", "atol",
        )
        for name in positive:
            for value in (0, -1):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    replace(p, **{name: value})
        for name in ("K_n", "k_f", "k_d", "bond_sigma0", "p_peak", "t_hold", "t_open_hold"):
            with self.subTest(field=name), self.assertRaises(ValueError):
                replace(p, **{name: -1})
        for value in (0, -1, 1.0, 1.5, False, np.bool_(True), 10**1000):
            with self.subTest(cycles=value), self.assertRaises(ValueError):
                replace(p, cycles=value)
        for value in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                replace(p, beta0=value)
        for value in (p.E, 2 * p.E):
            with self.assertRaises(ValueError):
                replace(p, p_peak=value)
        self.assertEqual(Parameters(cycles=np.int64(2)).cycles, 2)

    def test_derived_scales_and_schedule_resource_checks(self):
        invalid = (
            dict(E=1e308, A=10), dict(A=1e308, K_n=100),
            dict(v_pull=1e-320), dict(dt=1e-12), dict(cycles=1_000_000),
            dict(t_hold=1e20, dt=1e20), dict(t_unload=1e-20),
            dict(bond_sigma0=1e20, bond_delta_sigma=1e-3), dict(bond_delta_sigma=1e-320),
            dict(K_n=1e308, pull_distance=3), dict(k_f=1e308, p_ref=1e-100),
            dict(rtol=1e308, atol=1e308), dict(p_peak=1e-320, E=1e100),
            dict(E=1e301, A=1e-200, p_peak=1e300, t_hold=1e10, dt=1e10),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Parameters(**values)

    def test_invalid_scalar_inputs(self):
        p = Parameters()
        for invalid in (math.nan, math.inf, -math.inf, True, [0.1]):
            for function in (switch, off_rate):
                with self.assertRaises(ValueError):
                    function(invalid, p)
            with self.assertRaises(ValueError):
                mechanics(invalid, 0.5, p)
        for beta in (-0.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                backward_euler(beta, -0.1, 1.0, p)
        for h in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                backward_euler(0.5, -0.1, h, p)
        with self.assertRaises(ValueError):
            mechanics(p.L, 0, p)
        with self.assertRaises(ValueError):
            mechanics(-1e308, 1, p)


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Parameters()
        cls.base = simulate(cls.p)

    def test_output_contract_and_exact_endpoints(self):
        sol, p = self.base, self.p
        expected = {"t", "U", "u", "gap", "P", "F_b", "N", "pressure", "traction", "bond_traction",
                    "beta", "k_on", "k_off", "cycle", "phase", "stats", "segments"}
        self.assertEqual(set(sol), expected)
        self.assertTrue(np.all(np.diff(sol["t"]) > 0))
        self.assertLessEqual(np.max(np.diff(sol["t"])), p.dt + 1e-14)
        for name in expected - {"stats", "segments", "phase"}:
            self.assertEqual(sol[name].shape, sol["t"].shape)
            self.assertTrue(np.all(np.isfinite(sol[name])))
        self.assertTrue(np.all((sol["beta"] >= 0) & (sol["beta"] <= 1)))
        np.testing.assert_array_equal(sol["bond_traction"], p.K_n * sol["gap"])
        np.testing.assert_allclose(sol["traction"], sol["beta"] * sol["bond_traction"], atol=1e-15)
        self.assertEqual(sol["U"][0], -p.pull_distance)
        self.assertEqual(sol["U"][-1], -p.pull_distance)
        for segment in sol["segments"]:
            for time, value in ((segment["start"], segment["U_start"]), (segment["end"], segment["U_end"])):
                index = np.searchsorted(sol["t"], time)
                self.assertEqual(sol["t"][index], time)
                self.assertEqual(sol["U"][index], value)
        self.assertEqual(sol["stats"]["trials"], sol["stats"]["accepted_intervals"] + sol["stats"]["rejected_trials"])

    def test_no_bonds_frozen_and_zero_interface_stiffness(self):
        no_bonds = simulate(replace(self.p, k_f=0))
        self.assertTrue(np.all(no_bonds["beta"] == 0))
        self.assertTrue(np.all(no_bonds["F_b"] == 0))
        frozen = simulate(replace(self.p, k_f=0, k_d=0, beta0=0.6))
        self.assertTrue(np.all(frozen["beta"] == 0.6))
        free = simulate(replace(self.p, K_n=0))
        self.assertTrue(np.all(free["F_b"] == 0))
        self.assertTrue(np.all(free["k_off"] == 0))
        np.testing.assert_array_equal(free["gap"], np.maximum(-free["U"], 0))
        self.assertGreater(free["beta"][-1], 0.9)

    def test_step_doubling_accepts_two_halves_without_extrapolation(self):
        p = replace(self.p, dt=1, rtol=1, atol=1, beta0=0.2, k_d=0)
        sol = simulate(p)
        expected = full_state = p.beta0
        for index in range(1, len(sol["t"])):
            start, end = sol["t"][index - 1:index + 1]
            h = end - start
            U_mid = float(np.interp(start + h / 2, sol["t"], sol["U"]))
            expected = backward_euler(expected, U_mid, h / 2, p)
            expected = backward_euler(expected, sol["U"][index], h / 2, p)
            full_state = backward_euler(full_state, sol["U"][index], h, p)
            self.assertAlmostEqual(sol["beta"][index], expected, delta=2e-15)
        self.assertGreater(abs(sol["beta"][-1] - full_state), 1e-4)
        self.assertEqual(sol["stats"]["rejected_trials"], 0)
        self.assertEqual(sol["stats"]["accepted_intervals"], len(sol["t"]) - 1)

    def test_no_subthreshold_loss_even_at_large_gap(self):
        p = replace(self.p, bond_sigma0=2, k_f=0, beta0=0.7, t_open_hold=20)
        sol = simulate(p)
        self.assertTrue(np.all(sol["beta"] == p.beta0))
        self.assertTrue(np.all(sol["k_off"] == 0))

    def test_exact_compression_dose_includes_both_ramps(self):
        p = replace(self.p, k_d=0, beta0=0.13, k_f=0.7, rtol=1e-7, atol=1e-11, dt=0.023)
        sol = simulate(p)
        press_end = p.t_approach + p.t_press
        hold_end = press_end + p.t_hold
        unload_end = hold_end + p.t_unload
        doses = (
            (press_end, p.p_peak * p.t_press / 2),
            (hold_end, p.p_peak * (p.t_press / 2 + p.t_hold)),
            (unload_end, p.p_peak * ((p.t_press + p.t_unload) / 2 + p.t_hold)),
        )
        for time, dose in doses:
            exact = 1 - (1 - p.beta0) * math.exp(-p.k_f * dose / p.p_ref)
            actual = sol["beta"][np.searchsorted(sol["t"], time)]
            self.assertAlmostEqual(actual, exact, delta=1.5e-4)
        row = summarize_cycles(sol, p)[0]
        self.assertAlmostEqual(row["pressure_dose"], doses[-1][1], delta=2e-15)
        self.assertGreater(row["beta_pull_start"], row["beta_hold_end"])
        self.assertTrue(row["peak_at_pull_end"])

    def test_time_history_and_peak_convergence(self):
        coarse_p = replace(self.p, dt=0.12, rtol=1e-3, atol=1e-7)
        fine_p = replace(self.p, dt=0.0075, rtol=1e-7, atol=1e-11)
        coarse, fine = simulate(coarse_p), simulate(fine_p)
        medium = self.base
        times = medium["t"]
        reference = np.interp(times, fine["t"], fine["beta"])
        coarse_error = np.max(np.abs(np.interp(times, coarse["t"], coarse["beta"]) - reference))
        medium_error = np.max(np.abs(medium["beta"] - reference))
        self.assertLess(medium_error, coarse_error)
        self.assertLess(medium_error, 0.002)
        coarse_peak = summarize_cycles(coarse, coarse_p)[0]
        medium_peak = summarize_cycles(medium, self.p)[0]
        fine_peak = summarize_cycles(fine, fine_p)[0]
        self.assertLess(abs(medium_peak["peak_stress"] - fine_peak["peak_stress"]),
                        abs(coarse_peak["peak_stress"] - fine_peak["peak_stress"]))
        self.assertAlmostEqual(medium_peak["peak_stress"] / fine_peak["peak_stress"], 1, delta=0.01)
        self.assertAlmostEqual(medium_peak["peak_time"], fine_peak["peak_time"], delta=0.06)

    def test_visible_stickiness_preset_stretches_then_relaxes(self):
        p = Parameters(E=5, K_n=250, p_peak=0.05, p_ref=0.05, bond_sigma0=0.02,
                       bond_delta_sigma=0.01, k_d=2, v_pull=0.015, pull_distance=0.08,
                       t_hold=4, t_open_hold=8, dt=0.01)
        sol = simulate(p)
        row = summarize_cycles(sol, p)[0]
        self.assertEqual(p.k_b / p.k_r, 50)
        self.assertLess(1 / (1 + row["beta_pull_start"] * p.k_b / p.k_r), 0.025)
        self.assertGreater(row["peak_stress"] / p.E, 0.015)
        self.assertLess(row["peak_stress"] / p.E, 0.025)
        self.assertLess(row["end_stress"], 0.25 * row["peak_stress"])
        self.assertLess(row["end_stress"] / p.E, 1e-8)
        self.assertLess(row["beta_end"], 1e-9)
        self.assertFalse(row["peak_at_pull_end"])
        peak_index = np.searchsorted(sol["t"], row["peak_time"])
        strain = (sol["u"] - sol["U"]) / p.L
        self.assertGreater(strain[peak_index], strain[-1])
        fine_p = replace(p, dt=p.dt / 2, rtol=p.rtol / 4, atol=p.atol / 4)
        fine_row = summarize_cycles(simulate(fine_p), fine_p)[0]
        self.assertAlmostEqual(row["peak_stress"] / fine_row["peak_stress"], 1, delta=0.005)

    def test_activated_hold_exponential_loss_and_two_traction_measures(self):
        p = replace(self.p, rtol=1e-7, atol=1e-12, dt=0.01)
        sol = simulate(p)
        hold = next(s for s in sol["segments"] if s["phase"] == "open_hold")
        mask = sol["t"] >= hold["start"]
        beta, nominal, bond = (sol[name][mask] for name in ("beta", "traction", "bond_traction"))
        self.assertTrue(np.all(np.diff(beta) < 0))
        self.assertTrue(np.all(np.diff(nominal) < 0))
        self.assertTrue(np.all(np.diff(bond) > 0))
        self.assertTrue(np.all(sol["k_off"][mask] == p.k_d))
        self.assertLess(nominal[-1], p.bond_sigma0)
        expected = np.exp(-p.k_d * (sol["t"][mask] - hold["start"]))
        np.testing.assert_allclose(beta / beta[0], expected, rtol=0.002, atol=1e-10)

    def test_large_finite_stress_dose_does_not_overflow_intermediate_sum(self):
        p = Parameters(E=1.7e308, A=1e-100, p_peak=1e308, p_ref=1e308,
                       t_press=0.1, t_hold=0.1, t_unload=0.1)
        with np.errstate(over="raise", invalid="raise"):
            row = summarize_cycles(simulate(p), p)[0]
        self.assertTrue(math.isfinite(row["pressure_dose"]))
        self.assertAlmostEqual(row["pressure_dose"] / 2e307, 1, delta=2e-14)

    def test_area_changes_only_forces(self):
        p = replace(self.p, A=7.3)
        sol = simulate(p)
        for name in ("t", "U", "gap", "u", "beta", "pressure", "traction", "bond_traction", "k_on", "k_off"):
            np.testing.assert_allclose(sol[name], self.base[name], rtol=2e-10, atol=2e-12)
        for name in ("P", "F_b", "N"):
            np.testing.assert_allclose(sol[name], p.A * self.base[name], rtol=2e-10, atol=2e-12)
        row = summarize_cycles(sol, p)[0]
        self.assertLess(row["max_force_balance_error"], 1e-12)
        self.assertEqual(row["max_complementarity_error"], 0)

    def test_cycle_continuity_and_repeated_single_cycle_equivalence(self):
        p = replace(self.p, cycles=3, dt=0.037, t_open_hold=0.4)
        sol = simulate(p)
        rows = summarize_cycles(sol, p)
        old = p.beta0
        for index, row in enumerate(rows):
            self.assertAlmostEqual(row["beta_start"], old, delta=2e-9)
            one_p = replace(p, cycles=1, beta0=old)
            one = simulate(one_p)
            self.assertAlmostEqual(row["beta_end"], one["beta"][-1], delta=2e-9)
            if index:
                approach = next(s for s in sol["segments"] if s["cycle"] == index + 1 and s["phase"] == "approach")
                boundary = np.searchsorted(sol["t"], approach["start"])
                self.assertEqual(sol["cycle"][boundary], index)
                self.assertEqual(sol["beta"][boundary], rows[index - 1]["beta_end"])
                self.assertLess(sol["beta"][np.searchsorted(sol["t"], approach["end"])], row["beta_start"])
            old = float(one["beta"][-1])
        # An approach peak, even in the next cycle, must not contaminate pull peaks.
        changed = {**sol, "traction": sol["traction"].copy()}
        changed["traction"][sol["phase"] == "approach"] = 999
        changed_rows = summarize_cycles(changed, p)
        self.assertEqual([r["peak_stress"] for r in changed_rows], [r["peak_stress"] for r in rows])

    def test_independent_sweeps_and_pressure_independent_pull_speed(self):
        p = replace(self.p, cycles=2, beta0=0.2, dt=0.07, rtol=1e-4)
        for field, values in (("p_peak", [0.2, 0.0, 0.2]), ("t_hold", [0.0, 1.0])):
            rows = sweep(p, field, values)
            for case, value in enumerate(values):
                q = replace(p, **{field: value})
                expected = summarize_cycles(simulate(q), q)
                self.assertEqual(rows[case * p.cycles:(case + 1) * p.cycles], expected)
                self.assertEqual(rows[case * p.cycles]["beta_start"], p.beta0)
                for segment in load_segments(q):
                    if segment["phase"] == "pull":
                        speed = -(segment["U_end"] - segment["U_start"]) / (segment["end"] - segment["start"])
                        self.assertAlmostEqual(speed, p.v_pull, delta=1e-15)

    def test_zero_pressure_and_zero_holds(self):
        p = replace(self.p, p_peak=0, t_hold=0, t_open_hold=0, cycles=2)
        sol = simulate(p)
        self.assertEqual([s["phase"] for s in sol["segments"]], ["approach", "press", "unload", "pull"] * 2)
        self.assertTrue(np.all(sol["pressure"] == 0))
        self.assertTrue(np.all(sol["beta"] == 0))
        for row in summarize_cycles(sol, p):
            self.assertEqual(row["pressure_dose"], 0)
            self.assertEqual(row["peak_stress"], 0)
            self.assertFalse(row["peak_at_pull_end"])

    def test_invalid_sweeps_and_solution_arrays(self):
        for values in ([], 1, [[1]], [math.nan], [math.inf], [-1], [self.p.E], list(range(129)),
                       [True], ["1"], np.asarray([1 + 2j])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                sweep(self.p, "p_peak", values)
        with self.assertRaises(ValueError):
            sweep(self.p, "v_pull", [0.01])
        for key, value in (("beta", np.array([0.])), ("gap", self.base["gap"][:, None]),
                           ("t", self.base["t"][::-1]), ("pressure", self.base["pressure"] * math.nan)):
            with self.assertRaises(ValueError):
                summarize_cycles({**self.base, key: value}, self.p)
        bad = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.base.items()}
        endpoint = np.searchsorted(bad["t"], self.p.t_approach)
        bad["t"][endpoint] = np.nextafter(bad["t"][endpoint], math.inf)
        with self.assertRaises(ValueError):
            summarize_cycles(bad, self.p)

    def test_small_physical_times_are_not_skipped(self):
        p = replace(self.p, t_approach=1e-18, t_press=1e-18, t_hold=0,
                    t_unload=1e-18, v_pull=1.2e17, t_open_hold=0,
                    dt=3e-20, k_f=1e18, k_d=0)
        sol = simulate(p)
        self.assertLess(sol["t"][-1], 1e-17)
        self.assertGreater(sol["beta"][-1], 0.6)
        self.assertEqual(sol["t"][-1], sol["segments"][-1]["end"])

    def test_trials_are_bounded_including_rejections(self):
        with (
            patch.object(rod, "_MAX_TRIALS", 3),
            patch.object(rod, "backward_euler", side_effect=rod._ScalarSolveError),
        ):
            with self.assertRaisesRegex(RuntimeError, "trial resource limit"):
                simulate(self.p)
        with patch.object(rod, "backward_euler", side_effect=rod._ScalarSolveError):
            with self.assertRaisesRegex(RuntimeError, "time floor"):
                simulate(self.p)

    def test_simulation_imports_no_plotting_and_writes_nothing(self):
        # A fresh interpreter proves lazy imports even if plotting tests ran first.
        code = (
            "import sys; from unittest.mock import patch; import sticky_rod; "
            "assert 'matplotlib' not in sys.modules; "
            "exec(\"with patch('builtins.open', side_effect=AssertionError('file IO')):\\n"
            "    sticky_rod.simulate(sticky_rod.Parameters(k_f=0))\"); "
            "assert 'matplotlib' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).parent,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class NotebookFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cls.plt = plt
        cls.p = Parameters(beta0=0.25, dt=0.7, rtol=1e-4)
        cls.sol = simulate(cls.p)

    def tearDown(self):
        self.plt.close("all")

    def test_figures_and_inline_animation(self):
        from matplotlib.animation import FuncAnimation
        from matplotlib.figure import Figure
        figure = plot_history(self.sol, self.p)
        self.assertIsInstance(figure, Figure)
        self.assertEqual(len(figure.axes), 4)
        rows = summarize_cycles(self.sol, self.p)
        self.assertIsInstance(plot_sweep(rows, "p_peak"), Figure)
        animation = animate(self.sol, self.p, frames=4, fps=5)
        self.assertIsInstance(animation, FuncAnimation)
        self.assertIsInstance(animation._fig, Figure)
        for index in (0, 1, 2, 3):
            artists = animation._func(index)
            time = self.sol["t"][-1] * index / 3
            U = float(np.interp(time, self.sol["t"], self.sol["U"]))
            beta = float(np.interp(time, self.sol["t"], self.sol["beta"]))
            u, gap, P, Fb, _ = mechanics(U, beta, self.p)
            rod_patch, adhesive, _, _, status, reference, *cursors = artists
            self.assertAlmostEqual(rod_patch.get_x(), -self.p.L + U)
            self.assertAlmostEqual(rod_patch.get_x() + rod_patch.get_width(), u)
            np.testing.assert_allclose(adhesive.get_xdata(), [-gap, 0])
            self.assertEqual(adhesive.get_alpha(), beta)
            self.assertIn(f"pressure={P / self.p.A:.5g}", status.get_text())
            self.assertIn(f"traction={Fb / self.p.A:.5g}", status.get_text())
            self.assertIn(f"bond={self.p.K_n * gap:.5g}", status.get_text())
            self.assertAlmostEqual(reference.get_x(), rod_patch.get_x())
            self.assertEqual(reference.get_width(), self.p.L)
            for cursor in cursors:
                np.testing.assert_allclose(cursor.get_xdata(), [time, time])
        # HTML is held in memory; no save(), ffmpeg, or generated simulation files.
        html = animation.to_jshtml(default_mode="once")
        self.assertIn("<script", html)
        self.assertIn("image/png", html)
        self.plt.close(animation._fig)

    def test_magnification_changes_only_displayed_displacements(self):
        scale = 10
        animation = animate(self.sol, self.p, frames=4, fps=5, displacement_scale=scale)
        for index in range(4):
            time = self.sol["t"][-1] * index / 3
            U = float(np.interp(time, self.sol["t"], self.sol["U"]))
            beta = float(np.interp(time, self.sol["t"], self.sol["beta"]))
            u, gap, P, Fb, _ = mechanics(U, beta, self.p)
            rod_patch, adhesive, actuator, _, status, reference, *cursors = animation._func(index)
            self.assertAlmostEqual(rod_patch.get_x(), -self.p.L + scale * U)
            self.assertAlmostEqual(rod_patch.get_width(), self.p.L + scale * (u - U))
            self.assertAlmostEqual(rod_patch.get_x() + rod_patch.get_width(), scale * u)
            np.testing.assert_allclose(actuator.get_xdata(), [-self.p.L + scale * U])
            np.testing.assert_allclose(adhesive.get_xdata(), [-scale * gap, 0])
            self.assertEqual(adhesive.get_alpha(), beta)
            self.assertEqual(reference.get_width(), self.p.L)
            self.assertEqual(reference.get_x(), rod_patch.get_x())
            self.assertIn(f"pressure={P / self.p.A:.5g}", status.get_text())
            self.assertIn(f"gap={gap:.5g}", status.get_text())
            self.assertIn(f"actual strain={100 * (u - U) / self.p.L:.3f}%", status.get_text())
            for cursor in cursors:
                np.testing.assert_allclose(cursor.get_xdata(), [time, time])
        self.assertIn("10x (visual only)", animation._fig.axes[0].get_title())
        self.assertIn("image/png", animation.to_jshtml(default_mode="once"))

    def test_invalid_magnification_is_rejected_not_clamped(self):
        for scale in (0, -1, 0.5, True, math.nan, math.inf):
            with self.assertRaises(ValueError):
                animate(self.sol, self.p, displacement_scale=scale)
        with self.assertRaisesRegex(ValueError, "reduce displacement_scale"):
            animate(self.sol, self.p, displacement_scale=self.p.L / self.p.U_press)

    def test_visualization_validation(self):
        for frames in (0, 1, 601, 3.5, True):
            with self.assertRaises(ValueError):
                animate(self.sol, self.p, frames=frames)
        for fps in (0, 61, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                animate(self.sol, self.p, fps=fps)
        with self.assertRaises(ValueError):
            plot_sweep([], "p_peak")
        with self.assertRaises(ValueError):
            plot_sweep([{}], "t_hold")


if __name__ == "__main__":
    unittest.main()
