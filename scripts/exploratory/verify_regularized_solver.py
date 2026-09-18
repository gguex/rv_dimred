"""Integration checks for the production RV solver (CPU, float64).

Run: .venv/bin/python scripts/exploratory/verify_regularized_solver.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rv_kernels import (  # noqa: E402
    compute_student_t_kernel_torch as student,
)
from src.rv_kernels import (  # noqa: E402
    kernel_trace_penalty,
    rv_coefficient,
    rv_dimred,
)


class RegularizedSolverChecks(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(4)
        self.f = torch.arange(1.0, 8.0)
        self.f /= self.f.sum()
        self.y = torch.randn(7, 2)
        self.kx = student(torch.randn(7, 3), weights=self.f)

    def test_projected_penalty_and_force(self) -> None:
        for f in (self.f, torch.ones(7) / 7):
            for target in (0.1, 0.8):
                y = self.y.clone().requires_grad_()
                k = student(y, weights=f)
                k0 = torch.eye(7) - torch.outer(f.sqrt(), f.sqrt())
                projection = (k * k0).sum() / 6 * k0
                reference = target / 6 * k0
                penalty = kernel_trace_penalty(k, target)
                torch.testing.assert_close(
                    penalty, (projection - reference).square().sum() / 2
                )
                force = torch.autograd.grad(-2.5 * penalty, y)[0]
                delta = y[:, None] - y[None, :]
                g = 1 / (1 + delta.square().sum(-1))
                trace_force = (
                    4
                    * f[:, None]
                    * (f[None, :, None] * g[:, :, None].square() * delta).sum(1)
                )
                expected = 2.5 * (target - k.trace()) / 6 * trace_force
                torch.testing.assert_close(force, expected, atol=1e-12, rtol=1e-10)

    def test_zero_strength_preserves_legacy_trajectory(self) -> None:
        for hollow in (False, True):
            old = self.y.clone().requires_grad_()
            optimizer = torch.optim.Adam([old], lr=0.03)
            for _ in range(12):
                optimizer.zero_grad()
                (
                    -rv_coefficient(
                        self.kx, student(old, weights=self.f), hollow=hollow
                    )
                ).backward()
                optimizer.step()
            for kwargs in ({}, {"trace_strength": 0.0, "trace_target": 0.5}):
                result, rv = rv_dimred(
                    self.kx,
                    weights=self.f,
                    init=self.y,
                    n_iter=12,
                    lr=0.03,
                    hollow=hollow,
                    **kwargs,
                )
                torch.testing.assert_close(result, old, atol=0, rtol=0)
                expected = rv_coefficient(
                    self.kx, student(result, weights=self.f), hollow=hollow
                )
                self.assertEqual(rv, expected.item())

    def test_active_solver_matches_explicit_frobenius_objective(self) -> None:
        y = self.y.clone().requires_grad_()
        optimizer = torch.optim.Adam([y], lr=0.03)
        k0 = torch.eye(7) - torch.outer(self.f.sqrt(), self.f.sqrt())
        for _ in range(12):
            optimizer.zero_grad()
            k = student(y, weights=self.f)
            penalty = 0.5 * (((k * k0).sum() - 0.5) / 6 * k0).square().sum()
            (-rv_coefficient(self.kx, k) + 3 * penalty).backward()
            optimizer.step()
        result, rv = rv_dimred(
            self.kx,
            weights=self.f,
            init=self.y,
            n_iter=12,
            lr=0.03,
            trace_target=0.5,
            trace_strength=3.0,
        )
        torch.testing.assert_close(result, y, atol=1e-12, rtol=1e-10)
        self.assertEqual(
            rv, rv_coefficient(self.kx, student(result, weights=self.f)).item()
        )

    def test_callback_isolation_and_zero_steps(self) -> None:
        steps = []

        def callback(step: int, y: torch.Tensor) -> None:
            steps.append(step)
            self.assertFalse(y.requires_grad)
            y.zero_()  # a callback must not alter the live parameters

        args = dict(weights=self.f, init=self.y, n_iter=3)
        plain, _ = rv_dimred(self.kx, **args)
        observed, _ = rv_dimred(self.kx, callback=callback, **args)
        self.assertEqual(steps, [0, 1, 2, 3])
        torch.testing.assert_close(plain, observed, atol=0, rtol=0)
        result, rv = rv_dimred(self.kx, weights=self.f, init=self.y, n_iter=0)
        torch.testing.assert_close(result, self.y, atol=0, rtol=0)
        self.assertEqual(
            rv, rv_coefficient(self.kx, student(result, weights=self.f)).item()
        )

    def test_invalid_parameters(self) -> None:
        for strength in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rv_dimred(self.kx, trace_strength=strength)
        with self.assertRaises(ValueError):
            rv_dimred(self.kx, trace_strength=1)
        for target in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rv_dimred(self.kx, trace_target=target, trace_strength=1)
        with self.assertRaises(ValueError):
            kernel_trace_penalty(torch.ones(1, 1), 0.5)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RegularizedSolverChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    out = ROOT / "results/06_regularization/solver_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "passed": result.wasSuccessful(),
                "torch": torch.__version__,
                "dtype": "float64",
                "device": "cpu",
            },
            indent=2,
        )
        + "\n"
    )
    sys.exit(0 if result.wasSuccessful() else 1)
