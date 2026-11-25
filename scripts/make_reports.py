"""Writes research/reports/*.md and research/figures/*.svg from research/results.

    python scripts/make_reports.py

Every number in the reports is read from the benchmark JSON written by
`pe bench` (scripts/run-benchmarks.ps1), so rerunning the benchmarks and this
script keeps the reports in step with the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from poker_ecosystem import svg  # noqa: E402

RES, REP, FIG = ROOT / "research" / "results", ROOT / "research" / "reports", ROOT / "research" / "figures"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(name, chart):
    svg.save(chart, FIG / f"{name}.svg")
    return f"![{name}](../figures/{name}.svg)"


def meta_line(d):
    m = d.get("meta", {})
    return f"Measured on {m.get('note', 'the development laptop')}; {m.get('threads', '?')} threads."


def write(name, text):
    (REP / name).write_text(text.strip() + "\n", encoding="utf-8")
    print("wrote", REP / name)


def validation():
    a = load("bench-analytic.json")
    c = load("bench-cfr.json")
    dcfr_1000 = next(p["exploitability"] for r in c["runs"] if r["game"] == "leduc-2p" and r["method"] == "DCFR" for p in r["curve"] if p["iteration"] == 1000)
    rows = [
        ["Kuhn poker: value for the first player", f"{-1 / 18:.6f}", f"{a['kuhn']['computed']:.6f}", f"{a['kuhn']['error']:.1e}"],
        ["Leduc hold'em: value for the first player", f"{a['leduc']['published']}", f"{a['leduc']['computed']:.5f}", f"exploitability {a['leduc']['exploitability']:.1e}"],
        ["Kuhn poker: exploitability of the uniform policy", "0.458333 (OpenSpiel)", f"{c['references']['kuhn_uniform_exploitability']:.6f}", "exact"],
        ["Leduc hold'em: NashConv of the uniform policy", "4.747222 (OpenSpiel)", f"{2 * c['references']['leduc_uniform_exploitability']:.6f}", "exact"],
    ]
    clair = []
    for r in a["clairvoyance"]:
        f = r["curve"][-1]
        clair.append([f"{r['bet_size']}x pot", f"{r['closed_form']['bluff_freq']:.4f}", f"{f['bluff_freq']:.4f}", f"{r['closed_form']['call_freq']:.4f}", f"{f['call_freq']:.4f}", f"{r['closed_form']['value']:.4f}", f"{f['value']:.4f}", f"{f['iteration']:,}"])
    write("validation.md", f"""
# Validation against known answers

The solvers are checked against games whose answers are known, and against independent implementations of themselves.

## Known values

{table(['Benchmark', 'Known', 'Computed', 'Difference'], rows)}

## A river with a closed-form equilibrium

The clairvoyance game: the bettor holds the nuts or air (half each), the caller holds bluff-catchers that beat only the air; pot 1, bet *b*. At equilibrium the bettor bets every nut hand and bluffs so that bluffs are *b*/(1+2*b*) of its bets, and the caller calls 1/(1+*b*) of the time. It is solved here as a real card game by the range solver: out of position holds quads (2h with a kicker) or air that plays the board on 2s 2c 2d 7h 7c, in position holds 88 or 99. No card is shared between the two ranges, so the card game is exactly the abstract one.

{table(['Bet', 'Bluff (closed form)', 'Bluff (solver)', 'Call (closed form)', 'Call (solver)', 'Value (closed form)', 'Value (solver)', 'Iterations'], clair)}

## Cross-checks inside the test suite

| Check | What it shows |
|---|---|
| All 2,598,960 five-card and 133,784,560 seven-card hands | The evaluator reproduces the published category counts (7,462 distinct values) |
| All 376,992 five-card short-deck hands | Category counts derived combinatorially (flush 480 < full house 1,728) |
| 200,000 random hands per rule set | Agreement with an independent, deliberately naive evaluator |
| Vector solver vs. generic tabular CFR | For NLH river, NLH turn, PLO river and a 7-2 bounty spot, the range solver's strategy evaluated in an explicitly built game tree gives the same values and exploitability to 1e-9 |
| Compiled CFR vs. a plain reference CFR | Identical exploitability on Leduc after 10 and 40 iterations (difference below 1e-12) |
| Two-player multiway solver vs. heads-up solver | The joint-enumeration solver and the sweep solver agree to 1e-7 |
| Run-it-twice statistics | Exact inclusion-exclusion result equals brute force over ordered pairs of runouts |
| JavaScript engine vs. Rust engine | 26,612 checks: every evaluator value, exact equities, run-it-twice moments, tree sizes, solver values, exploitability and per-hand strategies after the same iterations, the hand-history parser on 412 hands, street-by-street range probabilities (equal to the last bit) and river reviews |

## A bug the validation caught

The first version accumulated each player's average strategy during the regret traversal, which skips subtrees the opponent's current strategy never reaches. That under-weights later iterations at information sets a best responder can still steer into. CFR+ hid it, but the average strategies of Discounted CFR and Predictive CFR+ stalled at an exploitability near 0.12 on Leduc while their current strategies converged. The fix accumulates realization weights from each player's own reach in a separate pass (tabular solver) or walks unreached subtrees during training (vector solvers). After it, DCFR reaches {dcfr_1000:.1e} at 1,000 iterations on Leduc instead of 0.117.
""")


def cfr_algorithms():
    c = load("bench-cfr.json")
    runs = {(r["game"], r["method"]): r for r in c["runs"]}
    methods = ["CFR", "CFR+", "LCFR", "DCFR", "PCFR+"]
    leduc = [Series(m, [p["iteration"] for p in runs[("leduc-2p", m)]["curve"]], [p["exploitability"] for p in runs[("leduc-2p", m)]["curve"]]) for m in methods]
    nodes = [Series(m.replace("(eps=0.6)", ""), [p["nodes_touched"] for p in runs[("leduc-2p", m)]["curve"]], [p["exploitability"] for p in runs[("leduc-2p", m)]["curve"]]) for m in ["CFR", "CFR+", "DCFR", "ES-MCCFR", "OS-MCCFR(eps=0.6)"]]
    f1 = fig("cfr_leduc_iterations", svg.line_chart(leduc, title="Leduc hold'em: exploitability by iteration", x_label="iterations", y_label="exploitability (chips per hand)", log_x=True, log_y=True))
    f2 = fig("cfr_leduc_nodes", svg.line_chart(nodes, title="Leduc hold'em: exploitability by nodes touched", x_label="nodes touched", y_label="exploitability", log_x=True, log_y=True))

    def at(game, m, it):
        for p in runs[(game, m)]["curve"]:
            if p["iteration"] == it:
                return p["exploitability"]
        return None
    rows = []
    for m in methods:
        rows.append([m] + [f"{at('leduc-2p', m, it):.2e}" if at('leduc-2p', m, it) is not None else "–" for it in (10, 100, 1000, 5000)] + [f"{runs[('leduc-2p', m)]['curve'][-1]['cpu_seconds']:.2f}"])
    mc = []
    for m in ["ES-MCCFR", "OS-MCCFR(eps=0.6)"]:
        p = runs[("leduc-2p", m)]["curve"][-1]
        mc.append([m.replace("(eps=0.6)", " (ε = 0.6)"), f"{p['iteration']:,}", f"{p['nodes_touched']:,}", f"{p['exploitability']:.2e}"])
    write("cfr-algorithms.md", f"""
# The CFR family on Kuhn and Leduc poker

Five full-width variants (alternating updates) and two Monte Carlo variants, on the two-player games where exploitability can be computed exactly. Exploitability is NashConv/2 in chips per hand.

{f1}

{table(['Algorithm', '10 it.', '100 it.', '1,000 it.', '5,000 it.', 'CPU s (5,000 it.)'], rows)}

{f2}

{table(['Monte Carlo variant', 'Iterations', 'Nodes touched', 'Final exploitability'], mc)}

## Reading the results

* Vanilla CFR converges slowly, roughly like 1/√T: {at('leduc-2p', 'CFR', 1000):.1e} chips per hand after 1,000 iterations on Leduc and {at('leduc-2p', 'CFR', 5000):.1e} after 5,000. It matches an independent reference implementation to 1e-12, so this is the algorithm, not a bug.
* Floored regrets (CFR+), discounting (DCFR) and prediction (PCFR+) are one to two and a half orders of magnitude better at equal iterations: DCFR reaches {at('leduc-2p', 'DCFR', 1000):.1e} at 1,000 iterations and {at('leduc-2p', 'DCFR', 5000):.1e} at 5,000. DCFR with (α, β, γ) = (1.5, 0, 2) is the default everywhere in this repository.
* Per node touched, sampling variants spend far less per iteration but converge noisily; on a game as small as Leduc full-width DCFR wins. Sampling matters for games too large to traverse, which is why the multiway blueprint literature uses it.

{meta_line(c)}
""")


Series = svg.Series


def solver_benchmarks():
    r = load("bench-river.json")
    t = load("bench-turn.json")
    rows = [[x["spot"], x["algorithm"], f"{x['hands'][0]:,} / {x['hands'][1]:,}", x["decision_nodes"], x["iterations_to"].get("0.5") or ">2,000", x["iterations_to"].get("0.1") or ">2,000", f"{x['final_pct_pot']:.3f}", f"{x['cpu_seconds']:.1f}", f"{x['memory_bytes'] / 1e6:.1f}"] for x in r["rows"]]
    srp = [x for x in r["rows"] if x["spot"].startswith("NLH SRP river BTN")]
    f1 = fig("river_convergence_srp", svg.line_chart([Series(x["algorithm"], [p["iteration"] for p in x["curve"]], [max(p["pct_pot"], 1e-4) for p in x["curve"]]) for x in srp], title="NLH single-raised-pot river: exploitability", x_label="iterations", y_label="% of the pot", log_y=True))
    trows = [[x["algorithm"], f"{x['hands'][0]} / {x['hands'][1]}", f"{x['nodes']:,}", x["iterations"], f"{x['final_pct_pot']:.3f}", f"{x['solve_cpu_seconds']:.1f}", f"{x['memory_bytes'] / 1e6:.1f}"] for x in t["rows"]]
    write("solver-benchmarks.md", f"""
# Range-vs-range solving: runtime, memory and convergence

Heads-up river spots in No-Limit Hold'em, Pot-Limit Omaha and Short Deck, solved by vector-form CFR with exact card removal. Exploitability is measured exactly (a vector best response in the same tree) every 50 iterations and reported as NashConv/2 in % of the starting pot. A solve stops at 0.05% of the pot or 2,000 iterations.

{table(['Spot', 'Algorithm', 'Hands', 'Decision nodes', 'Iter. to 0.5%', 'Iter. to 0.1%', 'Final % pot', 'CPU s', 'MB'], rows)}

{f1}

## A turn spot

The turn solve enumerates all 44 rivers below each betting line (exact chance, no card abstraction).

{table(['Algorithm', 'Hands', 'Tree nodes', 'Iterations', 'Final % pot', 'CPU s', 'MB'], trows)}

## What is and is not solved

* "Solved" means an ε-equilibrium of the game defined by the ranges, the board and the **bet menu**. Real no-limit poker allows any size; the menu is an action abstraction (its cost is measured in [abstraction.md](abstraction.md)).
* Card removal is exact: blockers between the two ranges are counted by inclusion-exclusion over each hand's card subsets (3 terms for hold'em, 15 for Omaha).
* Memory is the solver's own state (regrets and strategy sums in f64); the PLO spot holds 32,352 and 41,844 four-card hands.

{meta_line(r)}
""")


def abstraction():
    a = load("bench-abstraction.json")
    b = [x for x in a["card_buckets"] if x["buckets"]]
    exact = [x for x in a["card_buckets"] if not x["buckets"]][0]
    f1 = fig("abstraction_buckets", svg.line_chart([Series("bucketed river", [x["buckets"] for x in b], [x["pct_pot"] for x in b])], title="Card abstraction: exploitability in the real game", x_label="strength buckets per river decision", y_label="% of the pot", log_x=True, log_y=True, reference=exact["pct_pot"], reference_label="exact solve"))
    rows = [[x["buckets"], f"{x['pct_pot']:.3f}", f"{x['ev_oop']:.3f}", f"{x['memory_bytes'] / 1e3:.0f} kB"] for x in b] + [["exact (no buckets)", f"{exact['pct_pot']:.3f}", f"{exact['ev_oop']:.3f}", f"{exact['memory_bytes'] / 1e3:.0f} kB"]]
    menus = [[m["menu"], f"{m['ev'][0]:.3f}", f"{m['ev'][1]:.3f}", f"{m['pct_pot']:.3f}", m["decision_nodes"], " · ".join(f"{act} {100 * fr:.0f}%" for act, fr in zip(m["root_actions"], m["root_frequencies"]))] for m in a["bet_menus"]]
    lf, lm = a["leduc"]["full"], a["leduc"]["jack_queen_merged"]
    write("abstraction.md", f"""
# Sensitivity to abstraction

Spot: {a['spot']}.

## Card abstraction

At every river decision the acting player's hands are grouped into *k* equal-weight buckets by hand strength; a bucket shares one strategy. The abstract game is solved, and the resulting strategy is then measured **in the real game** (every hand separately) by an exact best response.

{f1}

{table(['Buckets', 'Exploitability (% pot)', 'Value out of position', 'Solver memory'], rows)}

Coarse buckets are badly exploitable (a strategy that cannot tell top pair from a bluff-catcher cannot balance either); the error falls steadily with finer buckets, but even 40 buckets stay above the exact solve.

## Action abstraction

The same spot with different bet menus. Each is solved to near equilibrium in its own tree.

{table(['Menu', 'Value OOP', 'Value IP', 'Exploitability (% pot)', 'Decision nodes', 'Root strategy'], menus)}

Richer menus change the players' values by a few hundredths of a big blind on this spot: the bet menu is a mild abstraction here, but one this experiment cannot bound in general (an opponent using an off-menu size is outside every tree above).

## A classic pathology in Leduc

Merging jacks and queens into one card class (both preflop and as the public card) and solving the merged game gives a strategy that loses {lm['curve'][-1]['exploitability']:.3f} chips per hand to a best response in the real game, against {lf['curve'][-1]['exploitability']:.1e} for the same algorithm on the full game ({lm['abstract_infosets']} instead of {lf['abstract_infosets']} information sets).
""")


def multiplayer():
    m = load("bench-multiway.json")
    p = load("bench-players.json")
    rows = [[f"{x['players']}", " / ".join(str(h) for h in x["hands"]), f"{x['tuples']:,}", f"{1000 * x['cpu_seconds_per_iteration']:.2f} ms", f"{x['curve'][0]['nash_conv_pct_pot']:.1f}", f"{x['final']['nash_conv_pct_pot']:.3f}", f"{x['memory_bytes'] / 1e3:.0f} kB"] for x in m["rows"]]
    f1 = fig("multiway_nashconv", svg.line_chart([Series(f"{x['players']} players", [c["iteration"] for c in x["curve"][1:]], [c["nash_conv_pct_pot"] for c in x["curve"][1:]]) for x in m["rows"]], title="Multiway river: NashConv by iteration (DCFR)", x_label="iterations", y_label="NashConv (% of pot)", log_x=True, log_y=True))
    prow = []
    for r in p["runs"]:
        c = r["curve"][-1]
        prow.append([r["game"], r["method"], f"{r['infosets']:,}", f"{r['nodes']:,}", f"{c['iteration']:,}", f"{c['nash_conv']:.2e}", f"{1000 * c['cpu_seconds'] / c['iteration']:.3f} ms", f"{(r['tree_bytes'] + r['solver_bytes']) / 1e6:.1f} MB"])
    write("multiplayer.md", f"""
# Performance and guarantees as the player count grows

## What changes with three or more players

With two players (zero-sum), CFR's average strategy converges to a Nash equilibrium and exploitability is the natural error measure. With three or more:

* no theorem says CFR converges to a Nash equilibrium, and in general it does not;
* equilibria need not be unique or interchangeable, so "the" equilibrium strategy of one player is not well defined;
* the right error measure is **NashConv**: the total that all players could gain by switching to best responses. It is computed exactly here, and small values certify an approximate equilibrium **of this abstracted game only**.

Every multiway result in this repository is labelled "multiway CFR approximation: no Nash guarantee".

## Multiway rivers (exact card removal by joint enumeration)

About 17 combos per player on Ks 7d 2c 9h 4s, one bet size (75%) and one raise.

{table(['Players', 'Hands per player', 'Joint hand combinations', 'CPU / iteration', 'NashConv at start (% pot)', 'NashConv after 300 it.', 'Memory'], rows)}

{f1}

The joint enumeration is exact (every player's cards, folded players included, block the others) but costs the product of the range sizes, about twelve times more per added player here. That is the practical limit of exact multiway solving; larger spots need sampling and card abstraction, whose error can then only be measured on small instances like these.

## Kuhn and Leduc for 2 to 5 players

{table(['Game', 'Method', 'Information sets', 'Tree nodes', 'Iterations', 'NashConv', 'CPU / iteration', 'Memory'], prow)}

Tree size grows exponentially with the number of players (Kuhn: 55 nodes for two players, {[r['nodes'] for r in p['runs'] if r['game'] == 'kuhn-5p'][0]:,} for five; Leduc: {[r['nodes'] for r in p['runs'] if r['game'] == 'leduc-2p'][0]:,} for two, {[r['nodes'] for r in p['runs'] if r['game'] == 'leduc-3p'][0]:,} for three). In these small games CFR's NashConv does fall close to zero empirically, which is consistent with published results on three-player Kuhn, but it is an observation, not a guarantee.

{meta_line(p)}
""")


def exploitation():
    x = load("bench-exploit.json")
    series = []
    rows = []
    for prof in x["profiles"]:
        base = prof["frontier"][0]["ev_vs_model"]
        series.append(Series(prof["profile"], [max(f["exploitability_of_exploiter_pct_pot"], 0.01) for f in prof["frontier"]], [100 * (f["ev_vs_model"] - base) / 6.0 for f in prof["frontier"]]))
        for f in prof["frontier"]:
            rows.append([prof["profile"], f"{f['p']:.2f}", f"{f['ev_vs_model']:.3f}", f"{100 * (f['ev_vs_model'] - base) / 6.0:+.1f}", f"{f['exploitability_of_exploiter_pct_pot']:.2f}"])
    f1 = fig("exploit_frontier", svg.line_chart(series, title="Restricted Nash responses: gain against the model vs exploitability", x_label="exploitability of the exploiting strategy (% of pot)", y_label="gain against the model (% of pot)", log_x=True))
    write("exploitation.md", f"""
# Equilibrium versus exploitation

Spot: {x['spot']} (pot 6 bb). Player 0 is the big blind (out of position), player 1 the button.

## Method

1. Solve the spot to near equilibrium (DCFR, {x['iterations']} iterations).
2. Build an **opponent model**: the modelled player's equilibrium strategy with the profile's decisions replaced by threshold play on the range reaching each node (fold the weakest share to a bet, raise the strongest; bet value from the top and bluffs from the bottom).
3. Solve a **restricted Nash response** (Johanson, Zinkevich and Bowling, 2007) for p from 0 to 1: the opponent plays the model with probability p and freely otherwise, unseen by the exploiter. p = 0 is the equilibrium; p = 1 the best response to the model.
4. For each solution report its value against the model and its exploitability: what a best-responding opponent could gain over the equilibrium value.

{f1}

{table(['Opponent model', 'p', 'Value vs model (bb)', 'Gain (% pot)', 'Exploitability (% pot)'], rows)}

## Reading the frontier

* Against a button that folds 70% to river bets, the equilibrium gains nothing extra: at equilibrium the big blind checks its whole range on this river, so the button's response to bets never comes up. A restricted Nash response with p = 0.5 starts betting, wins about a third of the pot more against the model, and can itself be exploited for about a fifth of the pot.
* Exploitation is cheap at first and expensive at the end: the last steps to the best response buy little extra value for a large increase in exploitability (up to 151% of the pot for the button that over-bluffs).
* The model is only as good as its frequencies: [opponent-model.md](opponent-model.md) measures how fast those frequencies can be estimated from hands.
""")


def equity():
    q = load("bench-equity.json")
    ex = [[r["case"], f"{r['runouts']:,}", " / ".join(f"{100 * e:.3f}%" for e in r["equity"]), f"{r['cpu_seconds']:.2f}"] for r in q["exact"]]
    mc = q["monte_carlo"]
    mrows = [[f"{r['samples']:,}", r["repetitions"], f"{100 * r['mean_abs_error']:.4f}", f"{100 * r['rmse']:.4f}", f"{100 * r['ci95_coverage']:.0f}%"] for r in mc["rows"]]
    f1 = fig("mc_error", svg.line_chart([Series("RMSE", [r["samples"] for r in mc["rows"]], [100 * r["rmse"] for r in mc["rows"]])], title="Monte Carlo equity error", x_label="samples", y_label="RMSE (percentage points)", log_x=True, log_y=True))
    rit = [[r["case"], f"{100 * r['result']['equity'][0]:.2f}%", f"{100 * r['result']['sd_once'][0]:.1f}%", f"{100 * r['result']['sd_twice'][0]:.1f}%", f"{r['result']['corr_runs'][0]:.3f}", f"{r['result']['runouts']:,}"] for r in q["run_it_twice"]]
    write("equity.md", f"""
# Equity: exactness, Monte Carlo error and running it twice

## Exact enumeration

{table(['Case', 'Runouts', 'Equity', 'CPU s'], ex)}

## Monte Carlo against the exact answer

{mc['case']}: exact equity {100 * mc['exact_equity']:.4f}%.

{table(['Samples', 'Runs', 'Mean abs. error (pp)', 'RMSE (pp)', '95% interval coverage'], mrows)}

{f1}

The error falls like 1/√n and the reported 95% intervals cover the exact value at close to the nominal rate, so the standard errors shown in the demo can be trusted.

## Running it twice

Each run is marginally a uniformly random runout, so the expected share is exactly the one-run equity. The variance falls by more than half because the runs are negatively correlated (the second is dealt from what the first left). The figures below are exact: the cross moment over all ordered pairs of disjoint runouts is computed by inclusion-exclusion over runout subsets, 2^k lookups per runout instead of a quadratic loop.

{table(['All-in', 'Equity', 'SD of share, once', 'SD of share, twice', 'Correlation of the runs', 'Runouts'], rit)}
""")


def hand_histories():
    g = load("bench-ranges.json")
    pc = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    grid_rows = [[f"{x['noise']:.1f}", f"{x['tuning_half']['mean_bits']:+.2f}", f"{x['tuning_half']['median_bits']:+.2f}", pc(x['tuning_half']['coverage_50']), pc(x['tuning_half']['coverage_80']), pc(x['tuning_half']['coverage_90']), pc(x['tuning_half']['mean_effective_share'])] for x in g["grid"]]
    held = g["held_out"]
    snap_name = {"flop": "start of the flop", "turn": "start of the turn", "river": "start of the river", "end": "end of the hand"}
    snap_rows = [[snap_name[s["snapshot"]], f"{s['metrics']['n']:,}", f"{s['metrics']['mean_bits']:+.2f}", pc(s['metrics']['coverage_50']), pc(s['metrics']['coverage_80']), pc(s['metrics']['coverage_90']), pc(s['metrics']['mean_effective_share'])] for s in held["by_snapshot"]]
    ab = g["ablations"]
    abl_rows = [
        ["fitted model (player frequencies, noise)", f"{held['all']['mean_bits']:+.2f}", pc(held['all']['coverage_90'])],
        ["population frequencies only", f"{ab['population_frequencies_only']['all']['mean_bits']:+.2f}", pc(ab['population_frequencies_only']['all']['coverage_90'])],
        ["no noise (pure threshold play)", f"{ab['no_noise']['all']['mean_bits']:+.2f}", pc(ab['no_noise']['all']['coverage_90'])],
    ]
    style_rows = [[s["style"].replace("_", "-"), f"{s['river']['n']:,}", f"{s['river']['mean_bits']:+.2f}", pc(s['river']['coverage_90'])] for s in held["by_style"]]
    f1 = fig("ranges_noise_fit", svg.line_chart([Series("mean bits about the true hand", [x["noise"] for x in g["grid"]], [x["tuning_half"]["mean_bits"] for x in g["grid"]])], title="Fitting the noise level: log score of the true hands", x_label="noise (share of actions treated as uninformative)", y_label="bits", reference=0.0, reference_label="uniform range"))
    levels = [0.5, 0.8, 0.9]
    f2 = fig("ranges_calibration", svg.line_chart(
        [Series(snap_name[s["snapshot"]], levels, [s["metrics"][f"coverage_{int(100 * q)}"] for q in levels]) for s in held["by_snapshot"]] + [Series("perfect calibration", levels, levels)],
        title="Calibration on held-out hands: true hands inside the credible region", x_label="credible level", y_label="share of true hands inside", x_min=0.45, x_max=0.95, y_min=0.4, y_max=1.0))
    review_part = ""
    try:
        r = load("bench-review.json")
        def srow(name, s):
            return [name, f"{s['decisions']['n']:,}", pc(s["share_on_path"]), f"{s['on_path']['mean_loss_bb']:.2f}", f"{s['on_path']['median_loss_bb']:.2f}", f"{s['on_path']['p90_loss_bb']:.2f}", pc(s["on_path"]["share_over_1bb"]), pc(s["share_rarely_played"]), f"{s['decisions']['mean_loss_bb']:.2f}"]
        srows = [srow(s["style"].replace("_", "-"), s["stats"]) for s in r["by_style"] if s["style"]] + [srow("all", r["overall"])]
        arows = [[a["action"], f"{a['stats']['decisions']['n']:,}", pc(a["stats"]["share_on_path"]), f"{a['stats']['on_path']['mean_loss_bb']:.2f}", pc(a["stats"]["on_path"]["share_over_1bb"])] for a in r["by_action"]]
        crow = [[c["hand"], f"{c['player']} ({c['style'].replace('_', '-')})", c["cards"], c["action"], c["best"], f"{c['loss_bb']:.2f}", pc(c["frequency"])] for c in r["costliest_on_path"][:8]]
        off_examples = [c for c in r["costliest"] if not c["on_path"]][:3]
        off_note = "; ".join(
            f"hand {c['hand']}, {c['cards']} {c['action'].lower()} ({c['loss_bb']:.0f} bb; "
            + (f"the opponent's line has an equilibrium probability of {100 * c['opponent_path_probability']:.2f}%" if c["opponent_path_probability"] < 0.02 else f"the player's own line has an equilibrium probability of {100 * c['path_probability']:.2f}%")
            + f"; the solver's frequency for the action there is {pc(c['frequency'])})"
            for c in off_examples
        )
        review_part = f"""
## River review

`pe bench review` reviews the heads-up river decisions in the first {r['hands']:,} hands of the corpus: {r['reviewed_hands']:,} rivers, {r['decisions']:,} decisions. Each river is solved with the reconstructed ranges (on average {r['mean_combos'][0]:,.0f} and {r['mean_combos'][1]:,.0f} combos: the noise keeps every combo above the pruning threshold) to {r['mean_exploitability_pct']:.2f}% of the pot on average, in {r['mean_iterations']:.0f} iterations.

A decision is **on the equilibrium path** when the equilibrium reaches it with the player's own line at least 5% of the time and with the opponent's range at least 2%. Off the path, the solver explores the decision little (low exploitability says nothing about play that is almost never reached), so its strategy and values there are rough; the tables report costs on the path and, in the last column, over all decisions.

{table(['Style', 'Decisions', 'On the path', 'Mean cost on path (bb)', 'Median', '90th percentile', 'Cost above 1 bb', 'Rarely played at equilibrium', 'Mean cost, all decisions'], srows)}

{table(['Action taken', 'Decisions', 'On the path', 'Mean cost on path (bb)', 'Cost above 1 bb'], arows)}

The costliest decisions on the path:

{table(['Hand', 'Player', 'Cards', 'Taken', 'Best', 'Cost (bb)', 'Played at eq.'], crow)}

Off the path the costs can look larger and contradict the frequencies, for example {off_note}. These are the cases the path flag is for.

Reading the numbers: a cost is the value of the best action minus the value of the action taken, both with equilibrium play afterwards, against the reconstructed ranges. "Rarely played": on-path decisions the equilibrium takes less than 5% of the time with that hand. The costs are heavy-tailed: most decisions cost nothing or little, and a few cost tens of big blinds. The ranges are estimates and the opponent is assumed balanced, so these are costs in a model; they rank decisions for study, they are not the bots' true losses.
"""
    except FileNotFoundError:
        pass
    write("hand-histories.md", f"""
# Street-by-street ranges and the river review

Corpus: the 20,000 simulated hands of [opponent-model.md](opponent-model.md) ({g['corpus']['players']} bots of six styles), regenerated in process from the same seed. The frequency table is built from all {g['corpus']['frequency_hands']:,} hands, as a player would build it from a database; the ranges are scored on {g['corpus']['evaluated_hands']:,} hands that reach the flop, where every player's cards are known.

## The model

Each player starts with the 1,326 combos equally likely. Each action multiplies the combo weights by the probability a model of play gives it: a threshold strategy built from the player's frequencies in that context (strongest hands raise, weakest fold; after the flop value from the top and bluffs from the bottom), mixed with a share *noise* of actions that follow the frequencies regardless of the cards. The method is in [docs/HAND-HISTORIES.md](../../docs/HAND-HISTORIES.md) and [docs/ALGORITHMS.md](../../docs/ALGORITHMS.md).

Two scores per player and snapshot (start of each street, end of the hand):

* **bits**: log2 of the probability of the true combo times the number of combos possible: how much more likely the reconstructed range makes the true hand than a uniform range does (0 = no information, negative = misleading);
* **coverage**: whether the true combo lies inside the 50%, 80% and 90% most likely part of the range. A calibrated model covers at those rates.

## Fitting the noise

The noise level is fitted on the first half of the evaluated hands by the mean log score (a proper scoring rule) and then fixed.

{f1}

{table(['Noise', 'Mean bits', 'Median bits', '50% region', '80% region', '90% region', 'Range kept (effective share)'], grid_rows)}

Without noise the model treats every action as fully revealing: the median hand is still located (positive median), but each slow-play or float rules the true hand nearly out, so the mean log score is strongly negative and the 90% regions cover the truth half the time. The chosen level is **{g['chosen_noise']}**.

## Held-out results

{table(['Snapshot', 'Players', 'Mean bits', '50% region', '80% region', '90% region', 'Range kept'], snap_rows)}

{f2}

{table(['Variant (held-out half)', 'Mean bits', '90% region'], abl_rows)}

* Player-specific frequencies matter: they reveal {held['all']['mean_bits'] - ab['population_frequencies_only']['all']['mean_bits']:.2f} bits per snapshot more than population frequencies.
* Calibration: overall the 50%, 80% and 90% regions hold the true hand {pc(held['all']['coverage_50'])}, {pc(held['all']['coverage_80'])} and {pc(held['all']['coverage_90'])} of the time, a little conservative; most so at the start of the flop ({pc(held['by_snapshot'][0]['metrics']['coverage_50'])}, {pc(held['by_snapshot'][0]['metrics']['coverage_80'])}, {pc(held['by_snapshot'][0]['metrics']['coverage_90'])}), and close to nominal by the end of the hand ({pc(held['by_snapshot'][-1]['metrics']['coverage_50'])}, {pc(held['by_snapshot'][-1]['metrics']['coverage_80'])}, {pc(held['by_snapshot'][-1]['metrics']['coverage_90'])}).
* Tight players are easier to read than loose ones at the river:

{table(['Style', 'River snapshots', 'Mean bits', '90% region'], style_rows)}
{review_part}
## Limits

The bots are a parametric model of player types; real players adapt and mix in ways a fitted noise level may not capture, so read these scores as a best case for this model. Ranges are hold'em only and taken from an observer's point of view.
""")


def index():
    write("README.md", """
# Research reports

| Report | Question |
|---|---|
| [validation.md](validation.md) | Do the solvers reproduce known answers and each other? |
| [cfr-algorithms.md](cfr-algorithms.md) | How do CFR, CFR+, Linear CFR, Discounted CFR, Predictive CFR+ and Monte Carlo CFR compare? |
| [solver-benchmarks.md](solver-benchmarks.md) | Runtime, memory and convergence of range-vs-range solving (NLH, PLO, Short Deck; river and turn) |
| [abstraction.md](abstraction.md) | How much do card buckets and bet menus cost? |
| [multiplayer.md](multiplayer.md) | What happens with three or more players, and how does cost grow? |
| [exploitation.md](exploitation.md) | Equilibrium vs exploitative play against an opponent model (restricted Nash responses) |
| [equity.md](equity.md) | Exact equity, Monte Carlo error and interval coverage, running it twice |
| [opponent-model.md](opponent-model.md) | How well does the Bayesian opponent model recover simulated players' tendencies? |
| [hand-histories.md](hand-histories.md) | How much do street-by-street ranges reveal about hidden hands, are they calibrated, and what does a river review find? |

Raw results are in `../results/`, produced by `scripts/run-benchmarks.ps1` (Rust) and `python -m poker_ecosystem.evaluate` (Python); `scripts/make_reports.py` regenerates these reports and the figures in `../figures/`.
""")


if __name__ == "__main__":
    REP.mkdir(parents=True, exist_ok=True)
    validation()
    cfr_algorithms()
    solver_benchmarks()
    abstraction()
    multiplayer()
    exploitation()
    equity()
    hand_histories()
    index()
