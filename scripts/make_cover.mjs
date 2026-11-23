// Renders the project cover: the big blind's river strategy at equilibrium next to
// a restricted Nash response against a button that folds 70% to bets.
// node scripts/make_cover.mjs OUT.svg
import { writeFileSync } from 'node:fs';
import { prepareSpot } from '../web/js/engine/tree.js';
import { HuSolver } from '../web/js/engine/solver.js';
import { buildModel } from '../web/js/engine/profile.js';
import { classOf, className } from '../web/js/engine/cards.js';
import { PRESETS } from '../web/js/ui/presets.js';

const cfg = PRESETS[0].config;
const solve = (model) => {
  const s = new HuSolver(prepareSpot(cfg), { algo: 'DCFR' });
  for (let i = 0; i < 400; i++) s.iterate();
  if (model) { s.setModel(1, model, 0.5); for (let i = 0; i < 400; i++) s.iterate(); }
  return s;
};
const gto = solve(null);
const model = buildModel(gto, 1, { fold_to_bet: 0.7, raise_vs_bet: 0.03, bluff_share: 0.33 });
const rnr = solve(model);
const gtoE = gto.exploitability();

function grid(s) {
  const r = s.nodeReport(0);
  const k = r.actions.length;
  const cls = Array.from({ length: 169 }, () => ({ w: 0, p: new Array(k).fill(0) }));
  for (const row of r.rows) {
    const c = cls[classOf(row.hand[0], row.hand[1])];
    c.w += row.weight;
    row.probs.forEach((p, a) => (c.p[a] += row.weight * p));
  }
  return { cls, actions: r.actions, freq: r.frequencies };
}
const colors = ['#2f8f6b', '#d9a441', '#c2452d'];
function drawGrid(g, x0, y0, size, title, sub) {
  const cell = size / 13;
  let out = `<text x="${x0}" y="${y0 - 34}" font-size="26" font-weight="600" fill="#1a1a18">${title}</text><text x="${x0}" y="${y0 - 10}" font-size="17" fill="#6b6963">${sub}</text>`;
  g.cls.forEach((c, i) => {
    const x = x0 + (i % 13) * cell, y = y0 + Math.floor(i / 13) * cell;
    if (!c.w) { out += `<rect x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="#ece9e1"/>`; }
    else {
      let off = 0;
      c.p.forEach((p, a) => { const w = ((cell - 2) * p) / c.w; if (w > 0.2) out += `<rect x="${x + 1 + off}" y="${y + 1}" width="${w}" height="${cell - 2}" fill="${colors[a]}"/>`; off += w; });
    }
    out += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" font-size="${cell * 0.3}" text-anchor="middle" fill="${c.w ? '#ffffff' : '#b5b1a6'}" font-family="IBM Plex Mono, Consolas, monospace" font-weight="600">${className(i)}</text>`;
  });
  return out;
}
const a = grid(gto), b = grid(rnr);
const W = 1200, H = 630, size = 455;
const legend = a.actions.map((act, i) => `<rect x="${60 + i * 170}" y="${H - 44}" width="16" height="16" rx="3" fill="${colors[i]}"/><text x="${84 + i * 170}" y="${H - 31}" font-size="17" fill="#2f2e2a">${act.type === 'check' ? 'Check' : `Bet ${Math.round(act.size * 100)}% pot`}</text>`).join('');
const pct = (f) => `${Math.round(100 * f[1] + 100 * (f[2] || 0))}% bets`;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Inter, Segoe UI, Arial, sans-serif">
<rect width="${W}" height="${H}" fill="#fbfaf6"/>
${drawGrid(a, 60, 110, size, 'Equilibrium', `checks everything · ${gtoE.pctPot.toFixed(2)}% of pot from equilibrium`)}
${drawGrid(b, 60 + size + 120, 110, size, 'Exploiting an over-folder', `restricted Nash response, p = 0.5 · ${pct(b.freq)}`)}
<text x="${60 + size + 60}" y="${110 + size / 2}" font-size="40" text-anchor="middle" fill="#14594b">→</text>
${legend}
<text x="${W - 60}" y="${H - 31}" font-size="17" text-anchor="end" fill="#6b6963">Big blind, river Q♠8♥5♦3♣2♠ · Poker Ecosystem</text>
</svg>`;
writeFileSync(process.argv[2] || 'cover.svg', svg);
console.log('equilibrium', a.freq.map((x) => x.toFixed(3)), 'rnr', b.freq.map((x) => x.toFixed(3)), 'gto expl', gtoE.pctPot.toFixed(3));
