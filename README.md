# FraudGuard — Payment Fraud Detection Done Realistically

> An end-to-end fraud-detection lab that avoids the classic pitfalls: it simulates customers with real habits and **five fraud patterns** at a 0.3–0.4% fraud rate, builds **leak-free behavioural features** in time order, evaluates on **future** transactions, picks the alert threshold that **minimises money lost**, and explains every alert with reason codes.

[![CI](https://github.com/ZakariaHibaoui2/fraud-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/ZakariaHibaoui2/fraud-detection-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![FinTech](https://img.shields.io/badge/domain-fraud%20%26%20risk-darkgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Results

About 165,000 transactions from 1,500 customers over 90 days. The model is trained on the first 70% of the timeline and tested on the last 30%. Mean ± std over 4 seeds:

| Model | PR-AUC | Recall @ 1% alert budget | Loss reduction vs. no model* |
|---|---|---|---|
| Logistic regression | 0.841 ± 0.017 | — | 80.0% ± 4.9 |
| **Gradient boosting** | **0.948 ± 0.007** | **97.1%** | **82.2% ± 1.1** |

\* Total cost = fraud amount missed + 5 per alert reviewed, at the cost-optimal threshold.

**PR-AUC, not accuracy:** with 0.4% fraud, "flag nothing" is 99.6% accurate and worthless. Gradient boosting ranks fraud much better (PR-AUC +0.11), but once each model uses its own cost-optimal threshold, the money saved is similar. Model choice should look at the business metric, not just the leaderboard.

**Detection by pattern** (gradient boosting, seed 0):

| Pattern | What happens | Detected |
|---|---|---|
| Card testing | tiny online charges in minutes, then a big one | 100% |
| Account takeover | new device, often abroad (40% via a local VPN), high value | 100% |
| Impossible travel | home purchase, then a foreign one an hour later | 100% |
| High-risk burst | gift cards / crypto at night | 98% |
| Big ticket | large purchase on the victim's **own** device, at home | 92% |

## Avoiding the usual mistakes

| Pitfall | What FraudGuard does |
|---|---|
| **Target leakage** through aggregate features | `FeatureBuilder` streams transactions in time order and updates customer state *after* computing features. A test proves the first transaction sees no history |
| **Random train/test split** (the future leaks into training) | Strict **time-based** split: train on the past, score the future |
| Threshold tuned on training predictions (over-confident) | Threshold chosen on a **later validation window**, then the model is refit on the full past. The first version tuned on training scores and made the better model look *more expensive* |
| Too-easy synthetic data | Legitimate customers also travel, buy new phones, make big purchases, shop at night and in bursts; fraudsters sometimes appear local. The first version scored ROC-AUC 0.9999, which was a sign the data was too easy |

## Behavioural features (17)

Amount (log, z-score vs. the customer's history, ratio to their max) · velocity (transactions and amount in the last hour, count in 24 h, time since last) · novelty (new device, merchant, country, category) · geography (outside home country, **country switch within 3 h**) · time (hour, night) · high-risk category · history length.

## Usage

```bash
pip install -e ".[dev]"
fraudguard evaluate                 # metrics, costs, detection by pattern
fraudguard alerts --top 10          # highest-risk transactions with reason codes
```

```text
 99.7%  C00068       2.18  online_gaming   AE  [FRAUD (card_testing)]
          - outside the customer's home country
          - burst of transactions in the last hour
          - high-risk category (gift card / crypto / gaming)
          - new merchant for this customer
```

## Project structure

```
src/fraudguard/
├── simulate.py   # customers, habits, trips, devices + 5 fraud patterns
├── features.py   # streaming, leak-free behavioural features
├── model.py      # time split, models, PR-AUC / recall@k, cost-optimal threshold, reason codes
└── cli.py
tests/            # 8 tests: imbalance, no-leakage, split ordering, cost logic, recall@k, reasons, end-to-end quality, CLI
```

## License

[MIT](LICENSE)
