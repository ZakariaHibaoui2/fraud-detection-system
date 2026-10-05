"""Leak-free behavioural features: each transaction only sees the customer's *past* transactions."""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from .simulate import CATEGORIES, HIGH_RISK, Txn

FEATURES = [
    "log_amount", "amount_z", "amount_vs_max", "hours_since_last", "txns_last_1h", "txns_last_24h",
    "amount_last_1h", "new_device", "new_merchant", "new_country", "foreign_country", "country_switch_fast",
    "hour", "night", "high_risk_category", "category_new", "history_len",
]


class FeatureBuilder:
    """Streams transactions in time order, keeping per-customer state (like an online feature store)."""

    def __init__(self):
        self.amounts: dict[str, list[float]] = defaultdict(list)
        self.times: dict[str, deque] = defaultdict(deque)
        self.devices: dict[str, set] = defaultdict(set)
        self.merchants: dict[str, set] = defaultdict(set)
        self.countries: dict[str, list[str]] = defaultdict(list)
        self.cats: dict[str, set] = defaultdict(set)
        self.last: dict[str, tuple[float, str]] = {}

    def transform_one(self, t: Txn) -> list[float]:
        c = t.customer
        hist = self.amounts[c]
        mean = float(np.mean(hist)) if hist else t.amount
        std = float(np.std(hist)) if len(hist) > 2 else max(mean * 0.5, 1.0)
        recent = self.times[c]
        while recent and t.ts - recent[0][0] > 24:
            recent.popleft()
        last_1h = [a for ts, a in recent if t.ts - ts <= 1]
        last_ts, last_country = self.last.get(c, (t.ts - 1000, t.country))
        home = max(set(self.countries[c]), key=self.countries[c].count) if self.countries[c] else t.country
        hour = t.ts % 24
        row = [
            math.log1p(t.amount),
            (t.amount - mean) / (std + 1e-6),
            t.amount / (max(hist) if hist else t.amount),
            min(t.ts - last_ts, 1000.0),
            len(last_1h),
            len(recent),
            sum(last_1h),
            float(bool(self.devices[c]) and t.device not in self.devices[c]),
            float(bool(self.merchants[c]) and t.merchant not in self.merchants[c]),
            float(bool(self.countries[c]) and t.country not in self.countries[c]),
            float(t.country != home),
            float(t.country != last_country and t.ts - last_ts < 3),
            hour,
            float(1 <= hour < 6),
            float(t.category in HIGH_RISK),
            float(bool(self.cats[c]) and t.category not in self.cats[c]),
            float(len(hist)),
        ]
        # update state *after* computing features (no leakage of the current transaction)
        hist.append(t.amount)
        recent.append((t.ts, t.amount))
        self.devices[c].add(t.device)
        self.merchants[c].add(t.merchant)
        self.countries[c].append(t.country)
        self.cats[c].add(t.category)
        self.last[c] = (t.ts, t.country)
        return row

    def transform(self, txns: list[Txn]) -> np.ndarray:
        return np.asarray([self.transform_one(t) for t in txns], dtype=float)


assert set(CATEGORIES) >= HIGH_RISK
