"""SQLite hand database for ``pe-hh/1`` corpora.

Tables
------
* ``hands``: one row per hand, unique on (site, hand_id); stakes, pot in big
  blinds, street reached (0 preflop .. 3 river), showdown / all-in /
  synthetic flags, flop texture of the first board, and the raw JSON.
* ``players``: screen names.
* ``hand_players``: seat, position, stack, hole cards, contributed, collected,
  net (milli and big blinds), last street seen and showdown flag.
* ``actions``: every action with the replayed pot before it and the amount to call.
* ``boards``: final board(s).
* ``stats_cache``: one row per statistic opportunity (see
  :mod:`poker_ecosystem.stats`): player, statistic, position, success, truth.
  Aggregates are ``GROUP BY`` queries over it. The cache is tagged with
  :data:`STATS_VERSION` and rebuilt from the raw JSON when that changes.
* ``river_reviews``: one row per reviewed heads-up river decision with known
  cards, imported from ``pe hh review FILE --out reviews.json``: the action
  taken, the best action at the equilibrium of the reconstructed river, the
  cost in big blinds and how often the equilibrium takes the action with that
  hand. :func:`costliest_decisions` lists the most expensive ones.

Importing is idempotent: a hand whose (site, id) already exists is skipped.
"all-in" means some action put a player all-in or ``derived.allin`` is set.

Command line::

    python -m poker_ecosystem.hhdb import DB FILE [FILE ...]
    python -m poker_ecosystem.hhdb query DB [--player NAME --position BTN --game NLH
        --limit-type NL --min-pot-bb 20 --max-pot-bb 200 --street turn
        --action flop:raise --showdown --allin --texture monotone --synthetic
        --limit 50 --json]
    python -m poker_ecosystem.hhdb stats DB [--player NAME] [--by-position] [--stat vpip] [--json]
    python -m poker_ecosystem.hhdb players DB [--min-hands 100] [--json]
    python -m poker_ecosystem.hhdb reviews DB REVIEWS.json
    python -m poker_ecosystem.hhdb costly DB [--player NAME] [--limit 20] [--json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import hh
from .stats import ALL, STAT_KEYS, StatsTable, hand_observations

STATS_VERSION = "1"
STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
TEXTURE_COLUMNS = {
    "paired": "flop_paired",
    "monotone": "flop_monotone",
    "two-tone": "flop_two_tone",
    "two_tone": "flop_two_tone",
    "rainbow": "flop_rainbow",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hands (
    hand_pk INTEGER PRIMARY KEY,
    site TEXT NOT NULL,
    hand_id TEXT NOT NULL,
    game TEXT, limit_type TEXT, currency TEXT, played_at TEXT, table_name TEXT,
    max_seats INTEGER, button INTEGER, sb INTEGER, bb INTEGER NOT NULL, ante INTEGER,
    bomb_pot INTEGER NOT NULL, board_kind TEXT, players_dealt INTEGER NOT NULL,
    total_pot INTEGER, rake INTEGER, pot_bb REAL,
    street_reached INTEGER NOT NULL, showdown INTEGER NOT NULL, allin INTEGER NOT NULL,
    synthetic INTEGER NOT NULL,
    flop_paired INTEGER, flop_monotone INTEGER, flop_two_tone INTEGER, flop_rainbow INTEGER,
    raw_json TEXT NOT NULL,
    UNIQUE (site, hand_id)
);
CREATE TABLE IF NOT EXISTS players (player_pk INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS hand_players (
    hand_pk INTEGER NOT NULL REFERENCES hands(hand_pk) ON DELETE CASCADE,
    seat INTEGER NOT NULL,
    player_pk INTEGER NOT NULL REFERENCES players(player_pk),
    position TEXT, position_group TEXT, stack INTEGER, hole TEXT, hero INTEGER NOT NULL,
    contributed INTEGER NOT NULL, collected INTEGER NOT NULL, net INTEGER NOT NULL, net_bb REAL NOT NULL,
    street_reached INTEGER NOT NULL, showdown INTEGER NOT NULL,
    PRIMARY KEY (hand_pk, seat)
);
CREATE TABLE IF NOT EXISTS actions (
    hand_pk INTEGER NOT NULL REFERENCES hands(hand_pk) ON DELETE CASCADE,
    street TEXT NOT NULL, idx INTEGER NOT NULL, seat INTEGER NOT NULL,
    player_pk INTEGER REFERENCES players(player_pk),
    type TEXT NOT NULL, amount INTEGER NOT NULL, to_amount INTEGER NOT NULL, all_in INTEGER NOT NULL,
    pot_before INTEGER NOT NULL, to_call INTEGER NOT NULL,
    PRIMARY KEY (hand_pk, street, idx)
);
CREATE TABLE IF NOT EXISTS boards (
    hand_pk INTEGER NOT NULL REFERENCES hands(hand_pk) ON DELETE CASCADE,
    board_no INTEGER NOT NULL, cards TEXT NOT NULL,
    PRIMARY KEY (hand_pk, board_no)
);
CREATE TABLE IF NOT EXISTS stats_cache (
    hand_pk INTEGER NOT NULL REFERENCES hands(hand_pk) ON DELETE CASCADE,
    player_pk INTEGER NOT NULL REFERENCES players(player_pk),
    stat TEXT NOT NULL, position TEXT NOT NULL, success INTEGER NOT NULL, truth REAL
);
CREATE TABLE IF NOT EXISTS river_reviews (
    hand_id TEXT NOT NULL, idx INTEGER NOT NULL,
    hand_pk INTEGER REFERENCES hands(hand_pk) ON DELETE CASCADE,
    seat INTEGER NOT NULL, player_pk INTEGER REFERENCES players(player_pk),
    cards TEXT NOT NULL, action TEXT NOT NULL, best TEXT NOT NULL,
    ev_loss_bb REAL NOT NULL, frequency REAL NOT NULL, size_matched INTEGER NOT NULL,
    board TEXT NOT NULL, pot_bb REAL NOT NULL, exploitability_pct REAL NOT NULL,
    PRIMARY KEY (hand_id, idx)
);
CREATE INDEX IF NOT EXISTS ix_reviews_loss ON river_reviews(ev_loss_bb);
CREATE INDEX IF NOT EXISTS ix_hands_game ON hands(game, limit_type);
CREATE INDEX IF NOT EXISTS ix_hands_pot ON hands(pot_bb);
CREATE INDEX IF NOT EXISTS ix_hands_flags ON hands(showdown, allin, synthetic);
CREATE INDEX IF NOT EXISTS ix_hand_players_player ON hand_players(player_pk, position);
CREATE INDEX IF NOT EXISTS ix_actions_street_type ON actions(street, type);
CREATE INDEX IF NOT EXISTS ix_actions_player ON actions(player_pk, street, type);
CREATE INDEX IF NOT EXISTS ix_stats_player ON stats_cache(player_pk, stat, position);
CREATE INDEX IF NOT EXISTS ix_stats_hand ON stats_cache(hand_pk);
"""


# --------------------------------------------------------------------------- connection


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (and initialise) a hand database; ``":memory:"`` works for tests."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    row = conn.execute("SELECT value FROM meta WHERE key = 'stats_version'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta(key, value) VALUES ('stats_version', ?)", (STATS_VERSION,))
        conn.commit()
    elif row["value"] != STATS_VERSION:
        rebuild_stats_cache(conn)


# --------------------------------------------------------------------------- import


@dataclass
class ImportResult:
    inserted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "ImportResult") -> None:
        self.inserted += other.inserted
        self.skipped += other.skipped
        self.failed += other.failed
        self.errors.extend(other.errors)


def _player_pk(conn: sqlite3.Connection, cache: dict[str, int], name: str) -> int:
    pk = cache.get(name)
    if pk is None:
        conn.execute("INSERT OR IGNORE INTO players(name) VALUES (?)", (name,))
        pk = int(conn.execute("SELECT player_pk FROM players WHERE name = ?", (name,)).fetchone()[0])
        cache[name] = pk
    return pk


def _hand_row(hand: hh.Hand, rp: hh.Replay) -> tuple[Any, ...]:
    bb = hh.big_blind(hand)
    texture = hh.flop_texture(hh.board(hand)) or {}
    reached = max((STREET_INDEX[s] for s in rp.streets_dealt), default=0)
    allin = hh.allin_info(hand) is not None or any(d.all_in for d in rp.decisions)
    total_pot = hand.get("total_pot")
    pot = int(total_pot) if total_pot is not None else rp.final_pot

    def flag(key: str) -> int | None:
        return int(texture[key]) if texture else None

    return (
        str(hand.get("site") or "unknown"), str(hand["id"]), hand.get("game"), hand.get("limit"),
        hand.get("currency"), hand.get("time"), hand.get("table"), hand.get("max_seats"), hand.get("button"),
        hand.get("sb"), bb, hand.get("ante"), int(bool(hand.get("bomb_pot"))), hand.get("board_kind"),
        len(rp.seats), total_pot, hand.get("rake"), pot / bb, reached,
        int(len(hh.showdown_seats(hand, rp)) >= 2), int(allin), int(bool(hand.get("synthetic"))),
        flag("paired"), flag("monotone"), flag("two_tone"), flag("rainbow"),
        json.dumps(hand, separators=(",", ":"), sort_keys=True),
    )


_HAND_COLUMNS = (
    "site, hand_id, game, limit_type, currency, played_at, table_name, max_seats, button, sb, bb, ante, "
    "bomb_pot, board_kind, players_dealt, total_pot, rake, pot_bb, street_reached, showdown, allin, synthetic, "
    "flop_paired, flop_monotone, flop_two_tone, flop_rainbow, raw_json"
)


def _player_rows(hand: hh.Hand, rp: hh.Replay) -> list[tuple[Any, ...]]:
    """hand_players values except the leading hand_pk; player name in place of its key."""
    bb = hh.big_blind(hand)
    shown = hh.showdown_seats(hand, rp)
    collected = hh.collected_by_seat(hand)
    net = hh.net_by_seat(hand, rp)
    rows = []
    for p in hh.dealt_players(hand):
        seat = int(p["seat"])
        position = rp.positions.get(seat)
        seen = max((STREET_INDEX[s] for s in rp.streets_dealt if s == "preflop" or rp.saw(seat, s)), default=0)
        hole = p.get("hole")
        rows.append((
            seat, rp.names[seat], position, hh.position_group(position), p.get("stack"),
            " ".join(hole) if isinstance(hole, list) else None, int(bool(p.get("hero"))),
            rp.contributions.get(seat, 0), collected.get(seat, 0), net.get(seat, 0), net.get(seat, 0) / bb,
            seen, int(seat in shown),
        ))
    return rows


def _insert_hand(conn: sqlite3.Connection, hand: hh.Hand, cache: dict[str, int], bluff_threshold: float) -> bool:
    """Insert one hand; everything that can fail is computed before the first write."""
    if hand.get("id") is None:
        raise hh.HandError("hand without an id")
    key = (str(hand.get("site") or "unknown"), str(hand["id"]))
    if conn.execute("SELECT 1 FROM hands WHERE site = ? AND hand_id = ?", key).fetchone():
        return False
    rp = hh.replay(hand)
    row = _hand_row(hand, rp)
    players = _player_rows(hand, rp)
    observations = hand_observations(hand, bluff_threshold=bluff_threshold).observations
    marks = ", ".join("?" * len(row))
    cur = conn.execute(f"INSERT OR IGNORE INTO hands({_HAND_COLUMNS}) VALUES ({marks})", row)
    if cur.rowcount == 0:
        return False
    hand_pk = int(cur.lastrowid)
    for seat, name, *rest in players:
        conn.execute(
            "INSERT INTO hand_players VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (hand_pk, seat, _player_pk(conn, cache, name), *rest),
        )
    conn.executemany(
        "INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (hand_pk, d.street, d.index, d.seat, cache.get(rp.names.get(d.seat, "")), d.type, d.amount, d.to,
             int(d.all_in), d.pot_before, d.to_call)
            for d in rp.decisions
        ],
    )
    conn.executemany(
        "INSERT INTO boards VALUES (?,?,?)",
        [(hand_pk, i, " ".join(b)) for i, b in enumerate(hh.boards(hand), start=1)],
    )
    _insert_observations(conn, hand_pk, observations, cache)
    return True


def _insert_observations(conn: sqlite3.Connection, hand_pk: int, observations: Iterable[Any], cache: dict[str, int]) -> None:
    conn.executemany(
        "INSERT INTO stats_cache VALUES (?,?,?,?,?,?)",
        [
            (hand_pk, _player_pk(conn, cache, o.player), o.stat, o.position, int(o.success), o.truth)
            for o in observations
        ],
    )


def import_hands(conn: sqlite3.Connection, hands: Iterable[hh.Hand], *, bluff_threshold: float = 0.5, source: str = "") -> ImportResult:
    """Insert hands in one transaction; existing (site, id) pairs are skipped."""
    result = ImportResult()
    cache: dict[str, int] = {}
    with conn:
        for i, hand in enumerate(hands):
            try:
                if _insert_hand(conn, hand, cache, bluff_threshold):
                    result.inserted += 1
                else:
                    result.skipped += 1
            except (hh.HandError, KeyError, TypeError, ValueError) as exc:
                result.failed += 1
                where = f"{source}#{i}" if source else f"#{i}"
                result.errors.append(f"{where} ({hand.get('id') if isinstance(hand, dict) else '?'}): {exc}")
    return result


def import_files(conn: sqlite3.Connection, paths: Iterable[str | Path], *, bluff_threshold: float = 0.5) -> ImportResult:
    """Import several ``.json`` / ``.jsonl`` files (one transaction per file)."""
    total = ImportResult()
    for path in paths:
        total.merge(import_hands(conn, hh.iter_hands(path), bluff_threshold=bluff_threshold, source=str(path)))
    return total


def rebuild_stats_cache(conn: sqlite3.Connection, *, bluff_threshold: float = 0.5) -> int:
    """Recompute ``stats_cache`` from the stored raw JSON; returns the hand count."""
    cache = {row["name"]: row["player_pk"] for row in conn.execute("SELECT name, player_pk FROM players")}
    count = 0
    with conn:
        conn.execute("DELETE FROM stats_cache")
        for row in conn.execute("SELECT hand_pk, raw_json FROM hands ORDER BY hand_pk").fetchall():
            obs = hand_observations(json.loads(row["raw_json"]), bluff_threshold=bluff_threshold)
            _insert_observations(conn, int(row["hand_pk"]), obs.observations, cache)
            count += 1
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('stats_version', ?)", (STATS_VERSION,))
    return count


# --------------------------------------------------------------------------- queries


@dataclass
class HandQuery:
    """Filters for :func:`query_hands`. ``None`` means "no filter".

    ``position``, ``street`` and ``action`` refer to ``player`` when one is
    given (the player sat there, saw that street, took that action) and to
    the hand otherwise; ``position`` needs a player. ``street`` is one of
    flop, turn, river (reached) or showdown. ``action`` is ``"street:type"``,
    e.g. ``"flop:raise"``. ``texture`` is paired, monotone, two-tone or rainbow
    (flop of the first board).
    """

    player: str | None = None
    position: str | None = None
    game: str | None = None
    limit_type: str | None = None
    min_pot_bb: float | None = None
    max_pot_bb: float | None = None
    street: str | None = None
    action: str | None = None
    showdown: bool | None = None
    allin: bool | None = None
    texture: str | None = None
    synthetic: bool | None = None
    limit: int | None = 100
    offset: int = 0


def _where(q: HandQuery) -> tuple[str, list[Any]]:
    joins, where, params = [], [], []  # type: ignore[var-annotated]
    if q.player is not None:
        joins.append("JOIN hand_players hp ON hp.hand_pk = h.hand_pk JOIN players p ON p.player_pk = hp.player_pk")
        where.append("p.name = ?")
        params.append(q.player)
    if q.position is not None:
        if q.player is None:
            raise ValueError("a position filter needs a player")
        where.append("hp.position = ?")
        params.append(q.position)
    for column, value in (("h.game", q.game), ("h.limit_type", q.limit_type)):
        if value is not None:
            where.append(f"{column} = ?")
            params.append(value)
    if q.min_pot_bb is not None:
        where.append("h.pot_bb >= ?")
        params.append(q.min_pot_bb)
    if q.max_pot_bb is not None:
        where.append("h.pot_bb <= ?")
        params.append(q.max_pot_bb)
    if q.street is not None:
        owner = "hp" if q.player is not None else "h"
        if q.street == "showdown":
            where.append(f"{owner}.showdown = 1")
        elif q.street in STREET_INDEX:
            where.append(f"{owner}.street_reached >= ?")
            params.append(STREET_INDEX[q.street])
        else:
            raise ValueError(f"unknown street {q.street!r}")
    if q.action is not None:
        street, _, kind = q.action.partition(":")
        if street not in STREET_INDEX or kind not in hh.ACTION_TYPES:
            raise ValueError(f"action filter must be street:type, got {q.action!r}")
        mine = " AND a.player_pk = hp.player_pk" if q.player is not None else ""
        where.append(f"EXISTS (SELECT 1 FROM actions a WHERE a.hand_pk = h.hand_pk AND a.street = ? AND a.type = ?{mine})")
        params.extend([street, kind])
    for column, value in (("h.showdown", q.showdown), ("h.allin", q.allin), ("h.synthetic", q.synthetic)):
        if value is not None:
            where.append(f"{column} = ?")
            params.append(int(value))
    if q.texture is not None:
        column = TEXTURE_COLUMNS.get(q.texture)
        if column is None:
            raise ValueError(f"unknown texture {q.texture!r}")
        where.append(f"h.{column} = 1")
    sql = " ".join(joins)
    if where:
        sql += " WHERE " + " AND ".join(where)
    return sql, params


def _query(q: HandQuery | None, filters: dict[str, Any]) -> HandQuery:
    if q is None:
        q = HandQuery()
    known = {f.name for f in fields(HandQuery)}
    unknown = set(filters) - known
    if unknown:
        raise TypeError(f"unknown filters: {sorted(unknown)}")
    for key, value in filters.items():
        setattr(q, key, value)
    return q


def query_hands(conn: sqlite3.Connection, query: HandQuery | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Hand summaries matching the filters, in import order."""
    q = _query(query, filters)
    tail, params = _where(q)
    extra = ", hp.position AS p_position, hp.net_bb AS p_net_bb, hp.hole AS p_hole" if q.player else ""
    sql = (
        "SELECT h.hand_pk, h.site, h.hand_id, h.game, h.limit_type, h.sb, h.bb, h.players_dealt, h.pot_bb, "
        f"h.street_reached, h.showdown, h.allin, h.synthetic{extra} FROM hands h {tail} ORDER BY h.hand_pk"
    )
    if q.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([int(q.limit), int(q.offset)])
    return [_summary(conn, row, bool(q.player)) for row in conn.execute(sql, params).fetchall()]


def _summary(conn: sqlite3.Connection, row: sqlite3.Row, with_player: bool) -> dict[str, Any]:
    boards = [r["cards"] for r in conn.execute("SELECT cards FROM boards WHERE hand_pk = ? ORDER BY board_no", (row["hand_pk"],))]
    winners = [
        r["name"]
        for r in conn.execute(
            "SELECT p.name FROM hand_players hp JOIN players p ON p.player_pk = hp.player_pk "
            "WHERE hp.hand_pk = ? AND hp.collected > 0 ORDER BY hp.seat",
            (row["hand_pk"],),
        )
    ]
    sb = row["sb"] or 0
    out = {
        "site": row["site"],
        "id": row["hand_id"],
        "game": row["game"],
        "limit": row["limit_type"],
        "stakes": f"{sb / 1000:g}/{row['bb'] / 1000:g}",
        "players": row["players_dealt"],
        "pot_bb": round(row["pot_bb"], 3) if row["pot_bb"] is not None else None,
        "board": " | ".join(boards),
        "street": hh.STREETS[row["street_reached"]],
        "showdown": bool(row["showdown"]),
        "allin": bool(row["allin"]),
        "synthetic": bool(row["synthetic"]),
        "winners": winners,
    }
    if with_player:
        out.update({"position": row["p_position"], "net_bb": round(row["p_net_bb"], 3), "hole": row["p_hole"]})
    return out


def iter_raw_hands(conn: sqlite3.Connection, query: HandQuery | None = None, **filters: Any) -> Iterator[dict[str, Any]]:
    """The stored JSON of every hand matching the filters (no limit by default)."""
    filters.setdefault("limit", None)
    q = _query(query, filters)
    tail, params = _where(q)
    sql = f"SELECT h.raw_json FROM hands h {tail} ORDER BY h.hand_pk"
    if q.limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([int(q.limit), int(q.offset)])
    for row in conn.execute(sql, params):
        yield json.loads(row["raw_json"])


def stats_table(conn: sqlite3.Connection, query: HandQuery | None = None, *, bluff_threshold: float = 0.5, **filters: Any) -> StatsTable:
    """A full :class:`StatsTable` (sizing, showdowns, sequences) for matching hands."""
    return StatsTable.from_hands(iter_raw_hands(conn, query, **filters), bluff_threshold=bluff_threshold)


def player_stats(
    conn: sqlite3.Connection,
    *,
    player: str | None = None,
    by_position: bool = False,
    stats: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregated (n, k, frequency) per player and statistic from ``stats_cache``."""
    position = "s.position" if by_position else f"'{ALL}'"
    where, params = [], []  # type: ignore[var-annotated]
    if player is not None:
        where.append("p.name = ?")
        params.append(player)
    if stats:
        where.append(f"s.stat IN ({', '.join('?' * len(stats))})")
        params.extend(stats)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    group = "p.name, s.stat" + (", s.position" if by_position else "")
    sql = (
        f"SELECT p.name AS player, s.stat AS stat, {position} AS position, COUNT(*) AS n, SUM(s.success) AS k "
        f"FROM stats_cache s JOIN players p ON p.player_pk = s.player_pk {clause} GROUP BY {group}"
    )
    order = {k: i for i, k in enumerate(STAT_KEYS)}
    pos_order = {p: i for i, p in enumerate(hh.POSITIONS)}
    rows = [
        {"player": r["player"], "stat": r["stat"], "position": r["position"], "n": r["n"], "k": r["k"], "freq": r["k"] / r["n"]}
        for r in conn.execute(sql, params)
    ]
    rows.sort(key=lambda r: (r["player"], order.get(r["stat"], len(order)), pos_order.get(r["position"], -1)))
    return rows


def list_players(conn: sqlite3.Connection, *, min_hands: int = 0) -> list[dict[str, Any]]:
    """Players with hand count, net result in big blinds, VPIP and PFR."""
    rows = conn.execute(
        "SELECT p.player_pk AS pk, p.name AS name, COUNT(*) AS hands, SUM(hp.net_bb) AS net_bb "
        "FROM hand_players hp JOIN players p ON p.player_pk = hp.player_pk "
        "GROUP BY p.player_pk HAVING COUNT(*) >= ? ORDER BY hands DESC, p.name",
        (min_hands,),
    ).fetchall()
    freq: dict[tuple[int, str], tuple[int, int]] = {}
    for r in conn.execute(
        "SELECT player_pk, stat, COUNT(*) AS n, SUM(success) AS k FROM stats_cache "
        "WHERE stat IN ('vpip', 'pfr') GROUP BY player_pk, stat"
    ):
        freq[(r["player_pk"], r["stat"])] = (r["n"], r["k"])
    out = []
    for r in rows:
        entry = {"player": r["name"], "hands": r["hands"], "net_bb": round(r["net_bb"] or 0.0, 3)}
        for stat in ("vpip", "pfr"):
            n, k = freq.get((r["pk"], stat), (0, 0))
            entry[stat] = round(k / n, 4) if n else None
        out.append(entry)
    return out


# --------------------------------------------------------------------------- command line


# --------------------------------------------------------------------------- river reviews


def import_reviews(conn: sqlite3.Connection, reviews: Iterable[dict[str, Any]]) -> int:
    """Stores the decisions with known cards of ``pe hh review`` results (the
    ``reviews`` list of its JSON). Hands already in the database are linked by
    id. Re-importing replaces earlier rows. Returns the number of decisions."""
    cache: dict[str, int] = {}
    count = 0
    for r in reviews:
        row = conn.execute("SELECT hand_pk FROM hands WHERE hand_id = ? ORDER BY hand_pk LIMIT 1", (str(r["id"]),)).fetchone()
        hand_pk = row["hand_pk"] if row is not None else None
        for d in r.get("decisions", []):
            if d.get("ev_loss_bb") is None or d.get("hand") is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO river_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(r["id"]), int(d["index"]), hand_pk, int(d["seat"]), _player_pk(conn, cache, d["name"]),
                    d["hand"], d["actions"][d["taken"]], d["actions"][d["best"]], float(d["ev_loss_bb"]),
                    float(d["frequency"]), int(bool(d["size_matched"])), r["board"], float(r["pot_bb"]), float(r["exploitability_pct"]),
                ),
            )
            count += 1
    conn.commit()
    return count


def costliest_decisions(conn: sqlite3.Connection, *, player: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Reviewed river decisions, most expensive first (optionally for one player)."""
    sql = (
        "SELECT r.hand_id AS id, p.name AS player, r.cards, r.action, r.best, r.ev_loss_bb, r.frequency, r.board, r.pot_bb "
        "FROM river_reviews r LEFT JOIN players p ON p.player_pk = r.player_pk"
    )
    args: list[Any] = []
    if player:
        sql += " WHERE p.name = ?"
        args.append(player)
    sql += " ORDER BY r.ev_loss_bb DESC, r.hand_id, r.idx LIMIT ?"
    args.append(int(limit))
    return [
        {**dict(row), "ev_loss_bb": round(row["ev_loss_bb"], 3), "frequency": round(row["frequency"], 3), "pot_bb": round(row["pot_bb"], 2)}
        for row in conn.execute(sql, args)
    ]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def format_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Fixed-width text table."""
    cells = [[_cell(r.get(c)) for c in columns] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths)), "  ".join("-" * w for w in widths)]
    lines.extend("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m poker_ecosystem.hhdb", description="SQLite database of pe-hh/1 hands.")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="import .json / .jsonl files (idempotent)")
    imp.add_argument("db")
    imp.add_argument("files", nargs="+")
    q = sub.add_parser("query", help="list hands matching filters")
    q.add_argument("db")
    q.add_argument("--player")
    q.add_argument("--position")
    q.add_argument("--game")
    q.add_argument("--limit-type")
    q.add_argument("--min-pot-bb", type=float)
    q.add_argument("--max-pot-bb", type=float)
    q.add_argument("--street", choices=["flop", "turn", "river", "showdown"])
    q.add_argument("--action", help="street:type, e.g. flop:raise")
    q.add_argument("--showdown", action=argparse.BooleanOptionalAction, default=None)
    q.add_argument("--allin", action=argparse.BooleanOptionalAction, default=None)
    q.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=None)
    q.add_argument("--texture", choices=["paired", "monotone", "two-tone", "rainbow"])
    q.add_argument("--limit", type=int, default=50)
    q.add_argument("--json", action="store_true")
    st = sub.add_parser("stats", help="aggregated statistics from the cache")
    st.add_argument("db")
    st.add_argument("--player")
    st.add_argument("--by-position", action="store_true")
    st.add_argument("--stat", action="append", help="restrict to a statistic (repeatable)")
    st.add_argument("--json", action="store_true")
    pl = sub.add_parser("players", help="players with hand counts, net, VPIP and PFR")
    pl.add_argument("db")
    pl.add_argument("--min-hands", type=int, default=0)
    pl.add_argument("--json", action="store_true")
    rv = sub.add_parser("reviews", help="import river reviews written by `pe hh review FILE --out REVIEWS.json`")
    rv.add_argument("db")
    rv.add_argument("file")
    co = sub.add_parser("costly", help="the most expensive reviewed river decisions")
    co.add_argument("db")
    co.add_argument("--player")
    co.add_argument("--limit", type=int, default=20)
    co.add_argument("--json", action="store_true")
    return parser


def _emit(rows: list[dict[str, Any]], columns: Sequence[str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_table(rows, columns))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        conn = connect(args.db)
    except (OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return _run(conn, args)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()


def _run(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    if args.command == "import":
        result = import_files(conn, args.files)
        print(f"inserted {result.inserted}, skipped {result.skipped} (already present), failed {result.failed}")
        for err in result.errors[:20]:
            print(f"  {err}", file=sys.stderr)
        return 1 if result.failed and not result.inserted else 0
    if args.command == "query":
        rows = query_hands(
            conn, player=args.player, position=args.position, game=args.game, limit_type=args.limit_type,
            min_pot_bb=args.min_pot_bb, max_pot_bb=args.max_pot_bb, street=args.street, action=args.action,
            showdown=args.showdown, allin=args.allin, synthetic=args.synthetic, texture=args.texture,
            limit=args.limit,
        )
        cols = ["site", "id", "game", "stakes", "players", "pot_bb", "street", "board", "showdown", "allin", "winners"]
        if args.player:
            cols[4:4] = ["position", "net_bb"]
        _emit(rows, cols, args.json)
        return 0
    if args.command == "stats":
        rows = player_stats(conn, player=args.player, by_position=args.by_position, stats=args.stat)
        _emit(rows, ["player", "stat", "position", "n", "k", "freq"], args.json)
        return 0
    if args.command == "reviews":
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        n = import_reviews(conn, data["reviews"] if isinstance(data, dict) else data)
        print(f"imported {n} reviewed river decisions")
        return 0
    if args.command == "costly":
        rows = costliest_decisions(conn, player=args.player, limit=args.limit)
        _emit(rows, ["id", "player", "cards", "action", "best", "ev_loss_bb", "frequency", "board", "pot_bb"], args.json)
        return 0
    rows = list_players(conn, min_hands=args.min_hands)
    _emit(rows, ["player", "hands", "net_bb", "vpip", "pfr"], args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
