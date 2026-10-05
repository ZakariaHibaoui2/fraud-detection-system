"""fraudguard command line."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .model import evaluate, models, reasons, time_split
from .simulate import simulate


def cmd_evaluate(a) -> None:
    r = evaluate(a.seed, a.customers, a.days)
    print(f"{r['transactions']:,} transactions, fraud rate {r['fraud_rate']:.2%}; "
          f"test period: {r['test_transactions']:,} txns, {r['test_frauds']} frauds\n")
    print(f"{'model':<22}{'PR-AUC':>8}{'ROC-AUC':>9}{'recall@1%':>11}{'alerts':>8}{'caught':>11}{'missed':>10}{'cost':>10}")
    for name, m in r["models"].items():
        print(f"{name:<22}{m['pr_auc']:>8.3f}{m['roc_auc']:>9.4f}{m['recall_at_1pct_alerts']:>11.1%}{m['alerts']:>8}"
              f"{m['fraud_amount_caught']:>11,.0f}{m['fraud_amount_missed']:>10,.0f}{m['total_cost']:>10,.0f}")
    print(f"{'no model':<22}{'':>36}{0:>8}{0:>11}{r['no_model_cost']:>10,.0f}{r['no_model_cost']:>10,.0f}")
    print("\nDetection by fraud pattern (gradient boosting, cost-optimal threshold):")
    for pat, rate in r["detection_by_pattern"].items():
        print(f"  {pat:<20}{rate:>7.1%}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)


def cmd_alerts(a) -> None:
    """Show the top alerts of the test period with reason codes."""
    txns = simulate(a.customers, a.days, a.seed)
    sp = time_split(txns)
    m = models(a.seed)["gradient_boosting"].fit(sp.x_train, sp.y_train)
    p = m.predict_proba(sp.x_test)[:, 1]
    for i in np.argsort(-p)[: a.top]:
        t = sp.test_txns[i]
        truth = f"FRAUD ({t.pattern})" if t.fraud else "legit"
        print(f"{p[i]:6.1%}  {t.customer}  {t.amount:>9.2f}  {t.category:<16}{t.country}  [{truth}]")
        for why in reasons(sp.x_test[i])[:4]:
            print(f"          - {why}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fraudguard", description="Payment fraud detection lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("evaluate", cmd_evaluate), ("alerts", cmd_alerts)):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--customers", type=int, default=1500)
        s.add_argument("--days", type=int, default=90)
        if name == "evaluate":
            s.add_argument("--json")
        else:
            s.add_argument("--top", type=int, default=10)
        s.set_defaults(func=fn)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
