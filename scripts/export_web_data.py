"""Exports the data files the web demo reads (web/data/*.json).

Run from the repository root after `pe hh simulate` has written data/synthetic:

    python scripts/export_web_data.py

* opponents.json: posterior estimates (mean, 90% interval, sample size), the
  population priors, each bot's true style and true frequencies (the corpus is
  synthetic, so the truth is known), heuristic clusters and the river exploit
  profile the solver uses.
* hands-sample.json: 250 hands of the corpus for the hand-history browser.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from poker_ecosystem import hh, opponent  # noqa: E402
from poker_ecosystem.cluster import cluster_players  # noqa: E402

DISPLAY = [
    ("vpip", "VPIP", "preflop"),
    ("pfr", "PFR", "preflop"),
    ("three_bet", "3-bet", "preflop"),
    ("fold_to_three_bet", "Fold to 3-bet", "preflop"),
    ("steal", "Steal", "preflop"),
    ("fold_to_steal", "Fold to steal", "preflop"),
    ("cbet_flop", "C-bet flop", "flop"),
    ("fold_vs_cbet_flop", "Fold to flop c-bet", "flop"),
    ("check_raise_flop", "Check-raise flop", "flop"),
    ("cbet_turn", "C-bet turn", "turn"),
    ("bet_unopened_river", "Bet river when checked to", "river"),
    ("fold_vs_bet_river", "Fold to river bet", "river"),
    ("raise_vs_bet_river", "Raise river bet", "river"),
    ("wtsd", "Went to showdown", "showdown"),
    ("wsd", "Won at showdown", "showdown"),
    ("river_bluff", "Bluff share of river bets", "showdown"),
]


def prior_dict(p) -> dict:
    a = getattr(p, "alpha", getattr(p, "a", None))
    b = getattr(p, "beta", getattr(p, "b", None))
    return {"a": a, "b": b, "mean": a / (a + b) if a is not None and b is not None and a + b > 0 else None}


def main() -> None:
    corpus = ROOT / "data" / "synthetic"
    hands = hh.load_hands(corpus / "hands.jsonl")
    bots = {b["name"]: b for b in json.loads((corpus / "players.json").read_text(encoding="utf-8"))}
    config = json.loads((corpus / "config.json").read_text(encoding="utf-8"))
    model = opponent.OpponentModel.from_hands(hands)
    clustering = cluster_players(model, seed=1)
    names = sorted({p["name"] for h in hands for p in h["players"]})
    hand_counts = {n: 0 for n in names}
    for h in hands:
        for p in h["players"]:
            hand_counts[p["name"]] += 1
    players = []
    for name in names:
        est = {}
        for key, _, _ in DISPLAY:
            e = model.estimate(name, key)
            truth = model.table.truth(name, key) if hasattr(model, "table") else None
            est[key] = {
                "mean": e.mean, "lo": e.lo, "hi": e.hi, "n": e.n, "k": e.k,
                "prior_mean": e.prior_mean, "truth": truth[0] if truth else None,
            }
        ex = model.exploit_profile(name)
        players.append({
            "name": name,
            "style": bots.get(name, {}).get("style"),
            "cluster": clustering.names[clustering.label_of(name)] if name in clustering.players else None,
            "hands": hand_counts[name],
            "estimates": est,
            "exploit": {k: ex[k] for k in ("bet_freq", "bluff_share", "fold_to_bet", "raise_freq", "check_raise_freq") if k in ex},
        })
    out = {
        "source": f"Synthetic corpus: {len(hands):,} simulated hands, {len(names)} bots, seed {config.get('seed')} (pe hh simulate)",
        "level": 0.9,
        "stats": [{"key": k, "label": lab, "street": st} for k, lab, st in DISPLAY],
        "population": {k: prior_dict(model.population_prior(k)) for k, _, _ in DISPLAY},
        "clusters": clustering.as_dict(),
        "players": players,
    }
    web = ROOT / "web" / "data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "opponents.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    sample = []
    for h in hands[:250]:
        h = dict(h)
        h["streets"] = [{"name": s["name"], "actions": [{k: v for k, v in a.items() if k != "truth"} for a in s["actions"]]} for s in h["streets"]]
        sample.append(h)
    (web / "hands-sample.json").write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
    shutil.copyfile(ROOT / "data" / "samples" / "pokerstars-sample.txt", web / "pokerstars-sample.txt")
    print(f"wrote {web / 'opponents.json'} ({len(players)} players), hands-sample.json ({len(sample)} hands)")


if __name__ == "__main__":
    main()
