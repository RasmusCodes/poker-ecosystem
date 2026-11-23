use super::*;
use pe_cfr::{GameDef, NodeSpec};

fn cfg(game: &str, board: &str, r0: &str, r1: &str, pot: f64, stack: f64, menu: BetMenu) -> SpotConfig {
    SpotConfig {
        game: game.into(),
        board: board.into(),
        board2: None,
        ranges: vec![r0.into(), r1.into()],
        pot,
        stacks: vec![stack],
        menu,
        rake: None,
        side_rule: None,
        dead: None,
    }
}

fn small_menu() -> BetMenu {
    BetMenu { bets: vec![0.5, 1.0], raises: vec![1.0], max_raises: 2, allin_threshold: 0.8, add_allin: false }
}

/// The same spot as an explicit extensive-form game, for pe-cfr's generic tools.
struct SpotGame<'a> {
    s: &'a HuSolver,
    pairs: Vec<(usize, usize)>,
    probs: Vec<f64>,
}

#[derive(Clone)]
struct St {
    pair: Option<usize>,
    node: usize,
}

impl<'a> SpotGame<'a> {
    fn new(s: &'a HuSolver) -> Self {
        let mut pairs = Vec::new();
        let mut w = Vec::new();
        for (i, &m0) in s.spot.masks[0].iter().enumerate() {
            for (j, &m1) in s.spot.masks[1].iter().enumerate() {
                if m0 & m1 == 0 {
                    pairs.push((i, j));
                    w.push(s.weights[0][i] * s.weights[1][j]);
                }
            }
        }
        let z: f64 = w.iter().sum();
        SpotGame { s, pairs, probs: w.iter().map(|x| x / z).collect() }
    }
}

impl GameDef for SpotGame<'_> {
    type State = St;
    fn name(&self) -> String {
        "spot".into()
    }
    fn num_players(&self) -> usize {
        2
    }
    fn initial(&self) -> St {
        St { pair: None, node: 0 }
    }
    fn spec(&self, st: &St) -> NodeSpec {
        let Some(pi) = st.pair else { return NodeSpec::Chance(self.probs.clone()) };
        let (h0, h1) = self.pairs[pi];
        let hands = [h0, h1];
        let tree = &self.s.tree;
        let nd = &tree.nodes[st.node];
        let p0 = tree.start_pot;
        match &nd.kind {
            NodeKind::Decision { player, actions, .. } => NodeSpec::Decision {
                player: *player,
                infoset: format!("{}:{}", st.node, hands[*player]),
                actions: actions.iter().map(|a| a.label()).collect(),
            },
            NodeKind::Chance { cards, .. } => {
                let used = self.s.spot.masks[0][h0] | self.s.spot.masks[1][h1];
                let ok: Vec<bool> = cards.iter().map(|c| c.mask() & used == 0).collect();
                let k = ok.iter().filter(|&&x| x).count() as f64;
                NodeSpec::Chance(ok.iter().map(|&x| if x { 1.0 / k } else { 0.0 }).collect())
            }
            NodeKind::Terminal { showdown } => {
                let bounty = match &self.s.spot.config.side_rule {
                    Some(SideRule::SevenDeuce { amount }) => *amount,
                    None => 0.0,
                };
                let is72 = [self.s.spot.is_72[0][h0], self.s.spot.is_72[1][h1]];
                let (mut u, winner) = if !showdown {
                    let f = if nd.folded[0] { 0 } else { 1 };
                    let c_f = nd.contrib[f];
                    let mut u = [0.0; 2];
                    u[f] = -c_f;
                    u[1 - f] = p0 + c_f;
                    (u, Some(1 - f))
                } else {
                    let game = self.s.spot.game;
                    let board = pe_cards::cards_of(nd.board);
                    let v0 = pe_eval::eval_hand(&game, &self.s.spot.ranges[0].hands[h0], &board);
                    let v1 = pe_eval::eval_hand(&game, &self.s.spot.ranges[1].hands[h1], &board);
                    let m = nd.contrib[0].min(nd.contrib[1]);
                    let q = p0 + 2.0 * m;
                    if v0 > v1 {
                        ([q - m, -m], Some(0))
                    } else if v1 > v0 {
                        ([-m, q - m], Some(1))
                    } else {
                        ([q / 2.0 - m, q / 2.0 - m], None)
                    }
                };
                // the 7-2 game: the winner of a pot holding seven-deuce offsuit collects the bounty
                if let Some(w) = winner {
                    if is72[w] {
                        u[w] += bounty;
                        u[1 - w] -= bounty;
                    }
                }
                // shift by half the starting pot so the game is zero-sum
                NodeSpec::Terminal(vec![u[0] - p0 / 2.0, u[1] - p0 / 2.0])
            }
        }
    }
    fn next(&self, st: &St, a: usize) -> St {
        match st.pair {
            None => St { pair: Some(a), node: 0 },
            Some(p) => {
                let child = match &self.s.tree.nodes[st.node].kind {
                    NodeKind::Decision { children, .. } | NodeKind::Chance { children, .. } => children[a],
                    _ => unreachable!(),
                };
                St { pair: Some(p), node: child }
            }
        }
    }
}

/// Converts the vector solver's average strategy into pe-cfr's flat format and
/// checks that both libraries compute the same values and exploitability.
fn cross_check(solver: &HuSolver) {
    let game = SpotGame::new(solver);
    let tree = pe_cfr::Tree::compile(&game);
    let mut probs = vec![0.0; tree.abs_actions];
    for iset in &tree.infosets {
        let (node, hand) = iset.key.split_once(':').unwrap();
        let (node, hand): (usize, usize) = (node.parse().unwrap(), hand.parse().unwrap());
        let sigma = solver.average_strategy(node);
        let n = solver.hands(iset.player);
        let abs = &tree.abs_infosets[iset.abs as usize];
        for a in 0..abs.num_actions {
            probs[abs.offset + a] = sigma[a * n + hand];
        }
    }
    let r = pe_cfr::nash_conv(&tree, &pe_cfr::Strategy { probs });
    let e = solver.exploitability();
    let p0 = solver.tree.start_pot;
    assert!((r.values[0] + p0 / 2.0 - e.ev[0]).abs() < 1e-9 * p0, "EV {} vs {}", r.values[0] + p0 / 2.0, e.ev[0]);
    assert!((r.exploitability.unwrap() - e.nash_conv / 2.0).abs() < 1e-9 * p0, "exploitability {} vs {}", r.exploitability.unwrap(), e.nash_conv / 2.0);
}

#[test]
fn river_solver_matches_generic_cfr_tools() {
    let c = cfg("NLH", "Ks7d2c9h4s", "AA,KK,QQ,AKs,KQo,72o,65s", "AK,KJs,JJ,T9s,99,A5s", 10.0, 40.0, small_menu());
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    for _ in 0..60 {
        s.iterate();
    }
    cross_check(&s);
}

#[test]
fn turn_solver_matches_generic_cfr_tools() {
    let menu = BetMenu { bets: vec![0.75], raises: vec![], max_raises: 1, allin_threshold: 0.9, add_allin: false };
    let c = cfg("NLH", "Ks7d2c9h", "AA,KQs,98s", "QQ,K9s,T8s", 10.0, 30.0, menu);
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::CfrPlus, None, 100_000).unwrap();
    for _ in 0..20 {
        s.iterate();
    }
    cross_check(&s);
}

#[test]
fn omaha_river_matches_generic_cfr_tools() {
    let menu = BetMenu { bets: vec![1.0], raises: vec![1.0], max_raises: 2, allin_threshold: 0.9, add_allin: false };
    let c = cfg("PLO", "Ah9h4d2cTs", "AAKx:0.2,JJT9,KQJ9", "QJ87,KKxx:0.05,9876", 10.0, 40.0, menu);
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    for _ in 0..30 {
        s.iterate();
    }
    cross_check(&s);
}

#[test]
fn converges_on_a_typical_river() {
    let c = cfg("NLH", "Qs8h5d3c2s", "random:0.25,22+,A2s+,KTs+,QTs+,JTs,AJo+,KQo", "TT+,AQs+,AKo,QJs,JTs,T9s,A5s:0.5", 20.0, 80.0, BetMenu::default());
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    let curve = solve_hu(&mut s, 400, 0.3, 50);
    let last = curve.last().unwrap();
    assert!(last.pct_pot < 0.5, "{curve:?}");
    assert!(curve.first().unwrap().pct_pot > last.pct_pot);
}

#[test]
fn every_algorithm_converges() {
    for algo in [Algorithm::Cfr, Algorithm::CfrPlus, Algorithm::Linear, Algorithm::DCFR, Algorithm::PredictiveCfrPlus] {
        let c = cfg("NLH", "Jh9h4c3d2s", "JJ,99,44,AJ,KJs,QTs,T8s,AhKh,65s", "QQ,TT,AJ,KJ,QJs,AhQh,A5s", 10.0, 50.0, small_menu());
        let mut s = HuSolver::new(Spot::from_config(c).unwrap(), algo, None, 100_000).unwrap();
        let curve = solve_hu(&mut s, 300, 0.25, 100);
        // vanilla CFR is far slower (as on Leduc); the others must be well under 1%
        let limit = if algo == Algorithm::Cfr { 3.0 } else { 1.0 };
        assert!(curve.last().unwrap().pct_pot < limit, "{}: {curve:?}", algo.name());
    }
}

#[test]
fn multiway_two_players_matches_heads_up() {
    let c = cfg("NLH", "Ks7d2c9h4s", "AA,KQ,72o,JTs", "QQ,K9s,AK", 10.0, 40.0, small_menu());
    let mut hu = HuSolver::new(Spot::from_config(c.clone()).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    let mut mw = MultiwaySolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, 1_000_000, 100_000).unwrap();
    for _ in 0..50 {
        hu.iterate();
        mw.iterate();
    }
    let e = hu.exploitability();
    let r = mw.report();
    assert!((e.ev[0] - r.ev[0]).abs() < 1e-7, "{e:?} {r:?}");
    assert!((e.nash_conv - r.nash_conv).abs() < 1e-7, "{e:?} {r:?}");
}

#[test]
fn three_way_river_nash_conv_falls() {
    let menu = BetMenu { bets: vec![0.75], raises: vec![], max_raises: 1, allin_threshold: 0.9, add_allin: false };
    let c = SpotConfig {
        game: "NLH".into(),
        board: "Ks7d2c9h4s".into(),
        board2: None,
        ranges: vec!["AA,KQ,JTs,99".into(), "KJs,QQ,77,A5s".into(), "AK,TT,98s,T8s".into()],
        pot: 15.0,
        stacks: vec![50.0],
        menu,
        rake: None,
        side_rule: None,
        dead: None,
    };
    let mut mw = MultiwaySolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, 2_000_000, 100_000).unwrap();
    let start = mw.report().nash_conv;
    for _ in 0..100 {
        mw.iterate();
    }
    let r = mw.report();
    assert!(r.nash_conv < start / 5.0, "{start} -> {}", r.nash_conv);
    assert!(r.label.contains("no Nash guarantee"));
}

#[test]
fn restricted_nash_response_trades_exploitation_for_safety() {
    let c = cfg("NLH", "Qs8h5d3c2s", "QQ+,AQs+,AKo,QJs,JTs,T9s,98s,A5s,76s,65s", "JJ-88,AQ,KQ,QJ,QT,J9s,T9s", 20.0, 60.0, BetMenu::default());
    let spot = Spot::from_config(c).unwrap();
    let mut gto = HuSolver::new(spot.clone(), Algorithm::DCFR, None, 100_000).unwrap();
    solve_hu(&mut gto, 300, 0.2, 100);
    // an opponent (player 1) who folds far too much to bets
    let profile = Profile { fold_to_bet: Some(0.75), raise_vs_bet: Some(0.02), ..Default::default() };
    let model = build_model(&gto, 1, &profile);
    gto.set_model(1, model.clone(), 0.0);
    let (gto_vs_model, br_vs_model) = gto.values_vs_model(0).unwrap();
    let gto_expl = gto.exploitability().pct_pot;
    let mut rnr = HuSolver::new(spot, Algorithm::DCFR, None, 100_000).unwrap();
    rnr.set_model(1, model, 0.5);
    for _ in 0..300 {
        rnr.iterate();
    }
    let (rnr_vs_model, _) = rnr.values_vs_model(0).unwrap();
    let rnr_expl = rnr.exploitability().pct_pot;
    assert!(br_vs_model >= rnr_vs_model - 1e-9);
    assert!(rnr_vs_model > gto_vs_model, "{gto_vs_model} {rnr_vs_model}");
    assert!(rnr_expl > gto_expl);
}

#[test]
fn seven_deuce_bounty_matches_generic_cfr_tools() {
    // The bounty's effect on strategy is a measured result (see the research
    // reports: at equilibrium the opponent calls more, and 72o may bluff less).
    // Here the payoff rule itself is checked exactly against the explicit game.
    let mut c = cfg("NLH", "Ts9c4h3d3s", "AA,KK,AQs,72o,64s,53s", "QQ-TT,AJs,KQs,98s", 10.0, 30.0, BetMenu { bets: vec![1.0], raises: vec![1.0], max_raises: 2, allin_threshold: 0.9, add_allin: false });
    c.side_rule = Some(SideRule::SevenDeuce { amount: 10.0 });
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    for _ in 0..40 {
        s.iterate();
    }
    cross_check(&s);
}

#[test]
fn hilo_double_board_and_buckets_run() {
    let menu = BetMenu { bets: vec![0.75], raises: vec![1.0], max_raises: 1, allin_threshold: 0.9, add_allin: false };
    let c = cfg("O8", "Ah4d7c9sKs", "A2xx:0.3,A3xx:0.3,KKxx:0.05", "AKxx:0.05,23xx:0.2,QJT9", 10.0, 30.0, menu.clone());
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    let curve = solve_hu(&mut s, 200, 0.5, 100);
    assert!(curve.last().unwrap().pct_pot < 3.0, "{curve:?}");
    let mut c = cfg("NLH", "Ks7d2c9h4s", "AA,KQ,JTs,99", "QQ,K9s,AK,T8s", 10.0, 30.0, menu.clone());
    c.board2 = Some("As8s3d5c6h".into());
    let mut s = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    let curve = solve_hu(&mut s, 200, 0.5, 100);
    assert!(curve.last().unwrap().pct_pot < 3.0, "{curve:?}");
    // a river solved with 3 strength buckets is more exploitable than the exact solve
    let c = cfg("NLH", "Qs8h5d3c2s", "QQ+,AQs+,AKo,QJs,JTs,T9s,98s,A5s,76s,65s", "JJ-88,AQ,KQ,QJ,QT,J9s,T9s", 20.0, 60.0, menu);
    let mut exact = HuSolver::new(Spot::from_config(c.clone()).unwrap(), Algorithm::DCFR, None, 100_000).unwrap();
    solve_hu(&mut exact, 200, 0.0, 200);
    let mut abs = HuSolver::new(Spot::from_config(c).unwrap(), Algorithm::DCFR, Some(Buckets { count: 3 }), 100_000).unwrap();
    solve_hu(&mut abs, 200, 0.0, 200);
    assert!(abs.exploitability().pct_pot > exact.exploitability().pct_pot + 0.5);
}
