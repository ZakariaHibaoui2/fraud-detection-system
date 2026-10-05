import numpy as np
import pytest

from fraudguard.cli import main
from fraudguard.features import FEATURES, FeatureBuilder
from fraudguard.model import best_threshold, cost_at, evaluate, reasons, recall_at_alert_rate, time_split
from fraudguard.simulate import PATTERNS, Txn, simulate


@pytest.fixture(scope="module")
def txns():
    return simulate(customers=500, days=60, seed=3)


def test_simulation_shape(txns):
    rate = np.mean([t.fraud for t in txns])
    assert 0.001 < rate < 0.01                                  # realistic class imbalance
    assert {t.pattern for t in txns if t.fraud} == set(PATTERNS)
    assert all(a.ts <= b.ts for a, b in zip(txns, txns[1:]))    # time ordered


def test_features_use_only_the_past():
    a = Txn(0, 1.0, "C1", 10.0, "grocery", "MA", "D1", "M1")
    b = Txn(1, 2.0, "C1", 500.0, "electronics", "FR", "D2", "M2")
    fb = FeatureBuilder()
    fa = dict(zip(FEATURES, fb.transform_one(a)))
    f2 = dict(zip(FEATURES, fb.transform_one(b)))
    assert fa["history_len"] == 0 and fa["new_device"] == 0     # first transaction: nothing to compare to
    assert f2["history_len"] == 1 and f2["new_device"] == 1 and f2["new_country"] == 1
    assert f2["country_switch_fast"] == 1 and f2["amount_vs_max"] == 50


def test_time_split_has_no_overlap(txns):
    sp = time_split(txns)
    assert len(sp.y_train) + len(sp.y_test) == len(txns)
    assert min(t.ts for t in sp.test_txns) >= max(t.ts for t in txns[: len(sp.y_train)])


def test_cost_threshold_logic():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    amt = np.array([100.0, 50.0, 3.0, 10.0])
    assert cost_at(0.5, p, y, amt).cost == 3.0 + 2 * 5.0          # missed 3 + two reviews
    assert best_threshold(p, y, amt).cost <= cost_at(0.0, p, y, amt).cost


def test_recall_at_alert_rate():
    y = np.array([0] * 99 + [1])
    p = np.linspace(0, 1, 100)
    assert recall_at_alert_rate(y, p, 0.01) == 1.0


def test_reason_codes():
    row = np.zeros(len(FEATURES))
    row[FEATURES.index("country_switch_fast")] = 1
    row[FEATURES.index("txns_last_1h")] = 6
    msgs = reasons(row)
    assert any("country changed" in m for m in msgs) and any("burst" in m for m in msgs)


def test_end_to_end_quality():
    r = evaluate(seed=1)
    gb, lr = r["models"]["gradient_boosting"], r["models"]["logistic_regression"]
    assert gb["pr_auc"] > 0.9 and gb["pr_auc"] >= lr["pr_auc"]
    assert gb["total_cost"] < 0.5 * r["no_model_cost"]            # the model pays for its review cost


def test_cli(capsys):
    assert main(["alerts", "--customers", "300", "--days", "40", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 3 and "- " in out
