"""
L2-regularised logistic regression, fitted by IRLS.

Neither scikit-learn nor scipy is installed on this machine, and the model is
small enough (tens of features) that a direct implementation is cheaper than
arguing with the package manager -- and auditable, which matters more here: a
paper about evaluation defects should not have a black box in its own baseline.

IRLS is Newton's method on the logistic likelihood. With D in the tens the
D x D solve is trivial and it converges in a handful of iterations. The ridge
term keeps the Hessian well conditioned when a feature is near-constant, which
happens on clips that barely move.

  python3 logreg.py      # self-tests
"""
from __future__ import annotations

import numpy as np


class LogReg:
    def __init__(self, l2: float = 1.0, iters: int = 50, tol: float = 1e-7):
        self.l2, self.iters, self.tol = l2, iters, tol
        self.w = self.mu = self.sd = None

    def _design(self, X):
        Z = (X - self.mu) / self.sd
        return np.hstack([np.ones((len(Z), 1)), Z])

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-8] = 1.0              # constant feature
        A = self._design(X)
        n, d = A.shape
        self.w = np.zeros(d)
        pen = self.l2 * np.eye(d)
        pen[0, 0] = 0.0                            # never penalise the bias
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ self.w, -35, 35)))
            W = np.maximum(p * (1 - p), 1e-9)
            g = A.T @ (p - y) + pen @ self.w
            H = (A * W[:, None]).T @ A + pen
            step = np.linalg.solve(H, g)
            self.w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        return self

    def predict_proba(self, X):
        A = self._design(np.asarray(X, np.float64))
        return 1.0 / (1.0 + np.exp(-np.clip(A @ self.w, -35, 35)))


def _tests():
    rng = np.random.default_rng(0)

    # Recover a known boundary from well-separated data.
    n = 4000
    X = rng.normal(size=(n, 3))
    true = np.array([0.5, 2.0, -1.0, 0.0])
    lin = true[0] + X @ true[1:]
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    m = LogReg(l2=1e-6).fit(X, y)
    assert np.corrcoef(m.w[1:], true[1:])[0, 1] > 0.99, m.w

    # Predictions are probabilities, and ORDER with the linear predictor.
    # Pearson is the wrong check: the sigmoid saturates, so a perfectly
    # monotone map scores well below 1. Rank correlation is the claim.
    p = m.predict_proba(X)
    assert p.min() >= 0 and p.max() <= 1
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    lin_hat = ((X - m.mu) / m.sd) @ m.w[1:]      # the model's own predictor
    assert np.corrcoef(rank(p), rank(lin_hat))[0, 1] > 0.999999
    assert np.corrcoef(rank(p), rank(lin))[0, 1] > 0.98

    # A constant feature must not produce a NaN.
    Xc = np.hstack([X, np.ones((n, 1))])
    m2 = LogReg(l2=1.0).fit(Xc, y)
    assert np.isfinite(m2.predict_proba(Xc)).all()

    # All-one-class data must still fit without blowing up.
    m3 = LogReg(l2=1.0).fit(X[:100], np.ones(100))
    assert np.isfinite(m3.predict_proba(X[:100])).all()
    assert m3.predict_proba(X[:100]).mean() > 0.5

    # Standardisation uses training statistics only: refitting on a shifted
    # copy must give the same probabilities.
    m4 = LogReg(l2=1.0).fit(X + 100.0, y)
    assert np.allclose(m4.predict_proba(X + 100.0),
                       LogReg(l2=1.0).fit(X, y).predict_proba(X), atol=1e-6)
    print("logreg.py: all self-tests pass")


if __name__ == "__main__":
    _tests()
