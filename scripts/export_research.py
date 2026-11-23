"""Condenses research/results/*.json into web/data/research.json for the demo's
Research tab (and prints the headline numbers). Missing suites are skipped.

    python scripts/export_research.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "research" / "results"


def load(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def find_key(obj, key):
    """First value stored under `key` anywhere in a nested structure."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def curve(run, x="iteration", y="exploitability"):
    return [[p[x], p[y] if p.get(y) is not None else p.get("nash_conv")] for p in run["curve"]]


def main() -> None:
    out: dict = {}
    if (a := load("bench-analytic.json")):
        out["analytic"] = {
            "kuhn": a["kuhn"], "leduc": a["leduc"],
            "clairvoyance": [{"bet": r["bet_size"], "closed": r["closed_form"], "final": r["curve"][-1], "curve": [[c["iteration"], c["bluff_error"], c["call_error"], c["value_error"]] for c in r["curve"]]} for r in a["clairvoyance"]],
        }
    if (c := load("bench-cfr.json")):
        runs = c["runs"]
        out["cfr"] = {
            game: {r["method"]: {"iterations": curve(r), "nodes": [[p["nodes_touched"], p["exploitability"]] for p in r["curve"]], "infosets": r["infosets"]} for r in runs if r["game"] == game}
            for game in sorted({r["game"] for r in runs})
        }
        out["cfr_references"] = c.get("references")
    if (e := load("bench-eval.json")):
        out["eval"] = e["rows"]
    if (q := load("bench-equity.json")):
        out["equity"] = {"exact": q["exact"], "mc": q["monte_carlo"], "rit": [{"case": r["case"], **r["result"]} for r in q["run_it_twice"]]}
    if (r := load("bench-river.json")):
        out["river"] = [{k: row[k] for k in ("spot", "game", "algorithm", "hands", "nodes", "decision_nodes", "memory_bytes", "cpu_seconds", "iterations", "iterations_to", "final_pct_pot")} | {"curve": [[p["iteration"], p["pct_pot"]] for p in row["curve"]]} for row in r["rows"]]
    if (t := load("bench-turn.json")):
        out["turn"] = [{k: v for k, v in row.items() if k != "curve"} | {"curve": [[p["iteration"], p["pct_pot"]] for p in row["curve"]]} for row in t["rows"]]
    if (ab := load("bench-abstraction.json")):
        out["abstraction"] = {
            "spot": ab["spot"], "buckets": ab["card_buckets"], "menus": ab["bet_menus"],
            "leduc": {"full": curve(ab["leduc"]["full"]), "merged": curve(ab["leduc"]["jack_queen_merged"]),
                      "infosets": [ab["leduc"]["full"]["abstract_infosets"], ab["leduc"]["jack_queen_merged"]["abstract_infosets"]]},
        }
    if (m := load("bench-multiway.json")):
        out["multiway"] = [{k: row[k] for k in ("players", "hands", "tuples", "nodes", "memory_bytes", "iterations", "cpu_seconds_per_iteration")} | {"final": row["final"], "curve": [[p["iteration"], p["nash_conv_pct_pot"]] for p in row["curve"]]} for row in m["rows"]]
    if (pl := load("bench-players.json")):
        out["players"] = [{"game": r["game"], "method": r["method"], "players": r["players"], "infosets": r["infosets"], "nodes": r["nodes"], "tree_bytes": r["tree_bytes"], "solver_bytes": r["solver_bytes"],
                           "cpu_per_iteration": (r["curve"][-1]["cpu_seconds"] / r["curve"][-1]["iteration"]) if r["curve"] else None,
                           "curve": [[p["iteration"], p["nash_conv"]] for p in r["curve"]]} for r in pl["runs"]]
    if (x := load("bench-exploit.json")):
        out["exploit"] = x
    if (o := load("opponent_model.json")):
        out["opponent"] = {
            "hands_total": find_key(o.get("corpus", {}), "hands"),
            "sizes": [{"hands": s["hands"], "mae": s["pooled"]["posterior"]["mae"], "raw_mae": s["pooled"]["raw"]["mae"], "coverage": s["pooled"]["coverage"], "width": s["pooled"].get("mean_interval_width"),
                       "kmeans_ari": s.get("clusters", {}).get("kmeans_ari"), "gmm_ari": s.get("clusters", {}).get("gmm_ari"), "gmm_k": s.get("clusters", {}).get("gmm_k")} for s in o["sizes"]],
        }
    dest = ROOT / "web" / "data" / "research.json"
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest} with sections: {', '.join(out)}")


if __name__ == "__main__":
    main()
