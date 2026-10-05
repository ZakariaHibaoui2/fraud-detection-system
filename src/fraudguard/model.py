"""Models, time-based evaluation, cost-optimal thresholds and reason codes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, FeatureBuilder
from .simulate import Txn, simulate

REVIEW_COST = 5.0          # analyst time to review one alert
REASONS = [
    ("country_switch_fast", lambda v: v > 0, "country changed within 3 h of the previous purchase"),
    ("new_device", lambda v: v > 0, "first time this device is used"),
    ("foreign_country", lambda v: v > 0, "outside the customer's home country"),
    ("txns_last_1h", lambda v: v >= 4, "burst of transactions in the last hour"),
    ("amount_z", lambda v: v >= 4, "amount far above the customer's usual spend"),
    ("high_risk_category", lambda v: v > 0, "high-risk category (gift card / crypto / gaming)"),
    ("night", lambda v: v > 0, "unusual hour (01:00-06:00)"),
    ("new_merchant", lambda v: v > 0, "new merchant for this customer"),
]


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    amount_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    amount_test: np.ndarray
    test_txns: list


def time_split(txns: list[Txn], train_frac: float = 0.7) -> Split:
    """Train on the past, test on the future - never shuffle transactions across time."""
    cut = txns[int(len(txns) * train_frac)].ts
    x = FeatureBuilder().transform(txns)                  # features computed once, in time order
    y = np.array([t.fraud for t in txns])
    amt = np.array([t.amount for t in txns])
    tr = np.array([t.ts < cut for t in txns])
    return Split(x[tr], y[tr], amt[tr], x[~tr], y[~tr], amt[~tr], [t for t, m in zip(txns, ~tr) if m])


def models(seed: int = 0) -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight="balanced",
                                                            random_state=seed),
    }


def recall_at_alert_rate(y, p, rate: float = 0.01) -> float:
    k = max(1, int(len(p) * rate))
    top = np.argsort(-p)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


@dataclass
class CostResult:
    threshold: float
    cost: float
    caught_amount: float
    missed_amount: float
    alerts: int


def cost_at(threshold: float, p, y, amounts) -> CostResult:
    alert = p >= threshold
    missed = float(amounts[(y == 1) & ~alert].sum())
    caught = float(amounts[(y == 1) & alert].sum())
    return CostResult(threshold, missed + REVIEW_COST * int(alert.sum()), caught, missed, int(alert.sum()))


def best_threshold(p, y, amounts) -> CostResult:
    return min((cost_at(t, p, y, amounts) for t in np.unique(np.round(p, 3))), key=lambda c: c.cost)


def reasons(row: np.ndarray) -> list[str]:
    f = dict(zip(FEATURES, row))
    return [msg for name, rule, msg in REASONS if rule(f[name])]


def evaluate(seed: int = 0, customers: int = 1500, days: int = 90) -> dict:
    txns = simulate(customers, days, seed)
    sp = time_split(txns)
    out = {"transactions": len(txns), "fraud_rate": float(np.mean([t.fraud for t in txns])),
           "test_transactions": int(len(sp.y_test)), "test_frauds": int(sp.y_test.sum()),
           "no_model_cost": float(sp.amount_test[sp.y_test == 1].sum()), "models": {}}
    fitted = {}
    # the most recent 20% of the training period is a validation window for choosing the alert threshold:
    # scores on data the model was fitted on are over-confident and would give a badly placed threshold
    v = int(len(sp.y_train) * 0.8)
    for name, m in models(seed).items():
        m.fit(sp.x_train[:v], sp.y_train[:v])
        thr = best_threshold(m.predict_proba(sp.x_train[v:])[:, 1], sp.y_train[v:], sp.amount_train[v:]).threshold
        m.fit(sp.x_train, sp.y_train)                       # refit on the full past before scoring the future
        p = m.predict_proba(sp.x_test)[:, 1]
        res = cost_at(thr, p, sp.y_test, sp.amount_test)
        out["models"][name] = {
            "pr_auc": float(average_precision_score(sp.y_test, p)), "roc_auc": float(roc_auc_score(sp.y_test, p)),
            "recall_at_1pct_alerts": recall_at_alert_rate(sp.y_test, p, 0.01), "threshold": float(thr),
            "alerts": res.alerts, "fraud_amount_caught": res.caught_amount, "fraud_amount_missed": res.missed_amount,
            "total_cost": res.cost,
        }
        fitted[name] = (m, p, thr)
    _, p, thr = fitted["gradient_boosting"]
    out["detection_by_pattern"] = {
        pat: float(np.mean([p[i] >= thr for i, t in enumerate(sp.test_txns) if t.pattern == pat]))
        for pat in sorted({t.pattern for t in sp.test_txns if t.fraud})
    }
    return out
