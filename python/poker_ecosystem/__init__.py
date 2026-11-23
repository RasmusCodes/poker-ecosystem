"""poker_ecosystem: the Python research layer of the poker ecosystem.

Standard library only. Modules (imported on demand, not eagerly):

* :mod:`~poker_ecosystem.hh`        load ``pe-hh/1`` hands and replay them
* :mod:`~poker_ecosystem.stats`     per-player frequency counters with documented definitions
* :mod:`~poker_ecosystem.bayes`     Beta / Beta-binomial / Dirichlet numerics and empirical-Bayes fits
* :mod:`~poker_ecosystem.opponent`  hierarchical opponent model and solver lock profiles
* :mod:`~poker_ecosystem.ranges`    the 169 hand classes and a preflop strength ordering
* :mod:`~poker_ecosystem.cluster`   k-means++, Gaussian mixtures, adjusted Rand index
* :mod:`~poker_ecosystem.hhdb`      SQLite hand database and its command line
* :mod:`~poker_ecosystem.leaks`     leak finder against population or baseline bands
* :mod:`~poker_ecosystem.svg`       dependency-free SVG charts
* :mod:`~poker_ecosystem.evaluate`  ground-truth evaluation on synthetic corpora

Run ``python -m poker_ecosystem`` for an overview of the command-line tools.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "bayes",
    "cluster",
    "evaluate",
    "hh",
    "hhdb",
    "leaks",
    "opponent",
    "ranges",
    "stats",
    "svg",
]
