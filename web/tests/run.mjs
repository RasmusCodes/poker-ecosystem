// Parity tests: the browser engine against vectors written by the Rust engine
// (`pe vectors`). Run from the repository root: node web/tests/run.mjs
import { readFileSync } from 'node:fs';
import { parseCards, gameOf, PREFLOP_ORDER } from '../js/engine/cards.js';
import { evalCards, evalOmaha, evalOmahaLow } from '../js/engine/eval.js';
import { equityExact, runItTwice } from '../js/engine/equity.js';
import { prepareSpot } from '../js/engine/tree.js';
import { HuSolver } from '../js/engine/solver.js';
import { parsePokerStars, analyseHand } from '../js/engine/hh.js';

const dir = new URL('../../research/vectors/', import.meta.url);
const load = (f) => JSON.parse(readFileSync(new URL(f, dir), 'utf8'));
let failures = 0, checks = 0;
const check = (ok, msg) => { checks++; if (!ok) { failures++; if (failures < 25) console.error('FAIL', msg); } };
const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

// evaluator
const ev = load('eval.json');
for (const h of ev.hands) check(evalCards(parseCards(h.cards), h.rules) === h.value, `eval ${h.rules} ${h.cards}`);
for (const o of ev.omaha) {
  check(evalOmaha(parseCards(o.hole), parseCards(o.board)) === o.high, `omaha high ${o.hole} ${o.board}`);
  check(evalOmahaLow(parseCards(o.hole), parseCards(o.board)) === o.low, `omaha low ${o.hole} ${o.board}`);
}
console.log(`evaluator: ${ev.hands.length + 2 * ev.omaha.length} values`);

// equity
const eq = load('equity.json');
for (const c of eq.cases) {
  const r = equityExact(gameOf(c.game), c.hands.map(parseCards), c.boards.map(parseCards));
  check(r.runouts === c.runouts, `runouts ${c.game} ${c.hands}`);
  r.equity.forEach((e, i) => check(close(e, c.equity[i], 1e-12), `equity ${c.game} ${c.hands} player ${i}: ${e} vs ${c.equity[i]}`));
}
const rit = runItTwice(gameOf('NLH'), eq.rit.hands.map(parseCards), parseCards(eq.rit.board));
for (const k of [['equity', 'equity'], ['sdOnce', 'sd_once'], ['sdTwice', 'sd_twice'], ['corrRuns', 'corr_runs']]) {
  rit[k[0]].forEach((x, i) => check(close(x, eq.rit.result[k[1]][i], 1e-9), `run it twice ${k[0]}[${i}] ${x} vs ${eq.rit.result[k[1]][i]}`));
}
console.log(`equity: ${eq.cases.length} exact cases + run it twice`);

// preflop order
const order = load('preflop_ranking.json');
check(order.join() === PREFLOP_ORDER.join(), 'preflop order');

// solver
const sv = load('solver.json');
for (const c of sv) {
  const t0 = Date.now();
  const s = new HuSolver(prepareSpot(c.config), { algo: 'DCFR' });
  check(s.tree.nodes.length === c.nodes, `${c.name}: nodes ${s.tree.nodes.length} vs ${c.nodes}`);
  check(s.hands[0].length === c.hands[0] && s.hands[1].length === c.hands[1], `${c.name}: hands`);
  for (let i = 0; i < c.iterations; i++) s.iterate();
  const e = s.exploitability();
  check(close(e.ev[0], c.ev[0], 1e-7) && close(e.ev[1], c.ev[1], 1e-7), `${c.name}: ev ${e.ev} vs ${c.ev}`);
  check(close(e.pctPot, c.pct_pot, 1e-6), `${c.name}: exploitability ${e.pctPot} vs ${c.pct_pot}`);
  const rep = s.nodeReport(0);
  rep.frequencies.forEach((f, a) => check(close(f, c.root_frequencies[a], 1e-7), `${c.name}: root frequency ${a} ${f} vs ${c.root_frequencies[a]}`));
  for (const row of c.root_sample) {
    const r = rep.rows.find((x) => x.hand.map((cc) => '23456789TJQKA'[cc >> 2] + 'cdhs'[cc & 3]).join('') === row.hand);
    check(!!r, `${c.name}: hand ${row.hand} present`);
    if (r) row.probs.forEach((p, a) => check(close(r.probs[a], p, 1e-7), `${c.name}: ${row.hand} prob ${a}`));
    if (r) row.evs.forEach((x, a) => check(close(r.evs[a], x, 1e-6), `${c.name}: ${row.hand} ev ${a} ${r.evs[a]} vs ${x}`));
  }
  console.log(`solver ${c.name}: ${c.iterations} iterations, ${(Date.now() - t0) / 1000}s, ${e.pctPot.toFixed(4)}% pot`);
}

// hand histories: the JS parser against the Rust parser's JSON for the same text
const root = new URL('../../', import.meta.url);
const compareHands = (label, text, jsonl, limit) => {
  const { hands, warnings } = parsePokerStars(text);
  check(warnings.length === 0, `${label}: warnings ${warnings.slice(0, 3)}`);
  const rust = jsonl.split('\n').filter(Boolean).slice(0, limit).map((l) => JSON.parse(l));
  check(hands.length >= rust.length, `${label}: ${hands.length} hands vs ${rust.length}`);
  rust.forEach((r, i) => {
    const h = analyseHand(hands[i]);
    const key = (x) => JSON.stringify([x.id, x.posts.map((p) => [p.seat, p.kind, p.amount]), x.streets.map((s) => [s.name, s.actions.map((a) => [a.seat, a.type, a.amount, a.to, a.all_in])]), x.boards, x.uncalled.map((u) => [u.seat, u.amount]), x.collected.map((c) => [c.seat, c.amount, c.pot, c.board ?? null]).sort(), x.total_pot, x.rake, x.players.map((p) => [p.seat, p.position, p.stack])]);
    check(key(h) === key(r), `${label}: hand ${r.id} differs`);
    check(h.derived.ok, `${label}: hand ${r.id} reconstructs: ${h.derived.issues}`);
    if (r.derived && r.derived.allin && h.derived.allin) {
      for (const [s, e] of Object.entries(r.derived.allin.equity)) check(Math.abs(h.derived.allin.equity[s] - e) < 0.02, `${label}: hand ${r.id} all-in equity ${h.derived.allin.equity[s]} vs ${e}`);
    }
  });
  console.log(`hand histories ${label}: ${rust.length} hands compared`);
};
compareHands('samples', readFileSync(new URL('data/samples/pokerstars-sample.txt', root), 'utf8'), readFileSync(new URL('research/vectors/sample-hands.jsonl', root), 'utf8'), 1000);
try {
  const text = readFileSync(new URL('data/synthetic/hands.txt', root), 'utf8');
  const head = text.split(/\n(?=PokerStars Hand #)/).slice(0, 400).join('\n');
  const jsonl = readFileSync(new URL('data/synthetic/hands.jsonl', root), 'utf8').split('\n').slice(0, 400).join('\n');
  compareHands('synthetic', head, jsonl, 400);
} catch (e) {
  console.log('synthetic corpus not present (run `pe hh simulate`); skipped');
}

console.log(`${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
