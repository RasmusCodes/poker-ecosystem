"""``python -m poker_ecosystem``: overview of the research package."""

from __future__ import annotations

import sys
from typing import Sequence

from . import __version__

HELP = f"""poker_ecosystem {__version__} - research layer of the poker ecosystem (standard library only)

Input: pe-hh/1 hand histories (docs/HAND-HISTORY-SCHEMA.md) as .json arrays or .jsonl files.

Command-line tools
  python -m poker_ecosystem.hhdb import DB FILE [FILE ...]      build / extend a SQLite hand database
  python -m poker_ecosystem.hhdb query DB [filters] [--json]    list hands (player, position, game, pot,
                                                                street, action, showdown, all-in, texture)
  python -m poker_ecosystem.hhdb stats DB [--player NAME] [--by-position] [--json]
  python -m poker_ecosystem.hhdb players DB [--min-hands N] [--json]
  python -m poker_ecosystem.evaluate --hands FILE --players players.json
                                                                opponent-model evaluation on a synthetic
                                                                corpus (results, figures, report under research/)

Library modules
  hh        loading, positions, replay (pot before each action, amount to call, raise counts)
  stats     VPIP, PFR, RFI, 3-bet, c-bets, check-raises, WTSD, sizing bins, ... (definitions in the docstring)
  bayes     incomplete beta, Beta quantiles, Beta-binomial, empirical-Bayes prior fits, Dirichlet posteriors
  opponent  hierarchical opponent model, preflop ranges, exploit_profile() for river solver locks
  cluster   k-means++, diagonal Gaussian mixtures with BIC, adjusted Rand index, style names
  leaks     probability of being outside a reference band, heuristic EV-impact ranking, all-in luck
  svg       line, grouped-bar, scatter and histogram charts for light and dark backgrounds

Run the tests from the repository root:
  python -m unittest discover -s python/tests -t python
"""


def main(argv: Sequence[str] | None = None) -> int:
    print(HELP)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
