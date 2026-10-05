"""Transaction simulator: customers with habits, plus five realistic fraud patterns (~0.5% of transactions)."""
from __future__ import annotations

import random
from dataclasses import dataclass

COUNTRIES = ["MA", "FR", "ES", "DE", "US", "GB", "AE", "MY", "NG", "BR"]
CATEGORIES = {   # category -> typical amount multiplier
    "grocery": 0.6, "restaurant": 0.8, "fuel": 0.9, "pharmacy": 0.5, "clothing": 1.5, "electronics": 3.0,
    "travel": 4.0, "subscription": 0.3, "gift_card": 1.2, "crypto_exchange": 2.5, "online_gaming": 0.7,
}
HIGH_RISK = {"gift_card", "crypto_exchange", "online_gaming"}
PATTERNS = ["card_testing", "account_takeover", "impossible_travel", "high_risk_burst", "big_ticket"]


@dataclass
class Txn:
    id: int
    ts: float              # hours since start
    customer: str
    amount: float
    category: str
    country: str
    device: str
    merchant: str
    fraud: int = 0
    pattern: str = ""


def simulate(customers: int = 1500, days: int = 90, seed: int = 0) -> list[Txn]:
    r = random.Random(seed)
    txns: list[Txn] = []
    profiles = []
    for c in range(customers):
        profiles.append({
            "id": f"C{c:05d}", "home": r.choices(COUNTRIES, weights=[30, 12, 8, 6, 6, 5, 4, 4, 3, 2])[0],
            "base": r.lognormvariate(3.3, 0.6), "rate": r.uniform(0.3, 2.5),          # txns per day
            "devices": [f"D{c}-{k}" for k in range(r.choice([1, 1, 2]))],
            "cats": r.sample(sorted(CATEGORIES), 5), "merchants": [f"M{r.randint(0, 3000)}" for _ in range(12)],
            "travels": r.random() < 0.15,
        })
    tid = 0
    for p in profiles:
        t = r.uniform(0, 24)
        trip_end, trip_country = -1.0, p["home"]
        while t < days * 24:
            t += r.expovariate(p["rate"] / 24)
            hour = t % 24
            if 1 <= hour < 6 and r.random() < 0.8:                     # people mostly sleep (but not always)
                continue
            if p["travels"] and t > trip_end and r.random() < 0.01:   # legitimate week-long trips abroad
                trip_end, trip_country = t + r.uniform(48, 240), r.choice([c for c in COUNTRIES if c != p["home"]])
            country = trip_country if t < trip_end else p["home"]
            if r.random() < 0.003:                                     # new phone / laptop: a genuinely new device
                p["devices"].append(f"D{p['id']}-{len(p['devices'])}")
            cat = r.choice(p["cats"])
            amount = round(p["base"] * CATEGORIES[cat] * r.lognormvariate(0, 0.5), 2)
            if r.random() < 0.01:                                      # occasional legitimate big purchase
                cat, amount = r.choice(["electronics", "travel"]), round(p["base"] * r.uniform(5, 14), 2)
            merchant = r.choice(p["merchants"]) if r.random() < 0.85 else f"M{r.randint(0, 4000)}"
            burst = 1 if r.random() > 0.02 else r.randint(2, 5)          # mall / online shopping sprees
            for k in range(burst):
                txns.append(Txn(tid, t + k * 0.1, p["id"], amount if k == 0 else round(amount * r.uniform(0.3, 1.2), 2),
                                cat, country, p["devices"][-1] if r.random() < 0.7 else r.choice(p["devices"]), merchant))
                tid += 1

    # fraud injection: ~0.5% of transactions, spread over time and customers
    n_fraud_events = int(len(txns) * 0.0012)
    for _ in range(n_fraud_events):
        p = r.choice(profiles)
        t0 = r.uniform(24, days * 24 - 2)
        pattern = r.choice(PATTERNS)
        thief_dev = f"X{r.randint(0, 10**6)}"
        # ~40% of fraudsters route through the victim's country (VPN / local mule)
        thief_country = p["home"] if r.random() < 0.4 else r.choice([c for c in COUNTRIES if c != p["home"]])
        events = []
        if pattern == "card_testing":            # many tiny charges in minutes, then a big one
            for k in range(r.randint(4, 9)):
                events.append((t0 + k * r.uniform(0.01, 0.05), round(r.uniform(0.5, 3), 2), "online_gaming", thief_country))
            events.append((t0 + 0.4, round(p["base"] * r.uniform(8, 20), 2), "electronics", thief_country))
        elif pattern == "account_takeover":      # new device + foreign country + high value
            for k in range(r.randint(1, 3)):
                events.append((t0 + k * 0.3, round(p["base"] * r.uniform(5, 15), 2), r.choice(["electronics", "travel"]),
                               thief_country))
        elif pattern == "impossible_travel":     # card cloned: used abroad an hour after a home purchase
            events.append((t0 + r.uniform(0.3, 1.5), round(p["base"] * r.uniform(1, 4), 2), "clothing", thief_country))
        elif pattern == "high_risk_burst":       # gift cards / crypto at night
            night = (t0 // 24) * 24 + r.uniform(1, 5)
            for k in range(r.randint(2, 5)):
                events.append((night + k * 0.2, round(r.uniform(50, 250), 2), r.choice(sorted(HIGH_RISK)), p["home"]))
        else:                                    # big ticket at a home-country merchant on a known device (hard)
            events.append((t0, round(p["base"] * r.uniform(6, 12), 2), "electronics", p["home"]))
        for ts, amount, cat, country in events:
            dev = r.choice(p["devices"]) if pattern in ("big_ticket", "high_risk_burst") else thief_dev
            txns.append(Txn(tid, ts, p["id"], amount, cat, country, dev, f"M{r.randint(3001, 4000)}", 1, pattern))
            tid += 1
        if pattern == "impossible_travel":       # the legitimate home purchase that makes the travel impossible
            txns.append(Txn(tid, t0, p["id"], round(p["base"] * 0.7, 2), "grocery", p["home"], p["devices"][0],
                            p["merchants"][0]))
            tid += 1
    txns.sort(key=lambda x: x.ts)
    return txns
