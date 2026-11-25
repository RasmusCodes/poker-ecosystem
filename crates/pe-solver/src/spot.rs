//! Spot definitions: the game, board, ranges, pot, stacks and bet menu.

use pe_cards::{Card, CardMask, Game, HandList, mask_of, parse_cards, parse_hand_list};
use serde::{Deserialize, Serialize};

/// Bet sizes as fractions of the pot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BetMenu {
    /// Opening bet sizes (fraction of the pot), e.g. `[0.33, 0.75]`.
    pub bets: Vec<f64>,
    /// Raise sizes: a raise of `x` calls first, then adds `x` times the pot after the call.
    pub raises: Vec<f64>,
    /// Bets and raises allowed per street.
    pub max_raises: usize,
    /// A bet or raise that would leave less than this share of the stack behind goes all-in.
    pub allin_threshold: f64,
    /// Always offer an all-in.
    pub add_allin: bool,
}

impl Default for BetMenu {
    fn default() -> Self {
        BetMenu { bets: vec![0.33, 0.75], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false }
    }
}

/// Rake taken from the pot at the end of the hand (makes the game non-zero-sum).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rake {
    pub percent: f64,
    pub cap: f64,
}

/// Payoff side rules.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SideRule {
    /// The 7-2 game: whoever wins a pot holding seven-deuce offsuit collects
    /// `amount` from every other player dealt in.
    SevenDeuce { amount: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpotConfig {
    /// `NLH`, `PLO`, `PLO5`, `SD`, `SDC`, `O8`, `BIGO`.
    pub game: String,
    /// Three to five board cards, e.g. `"Ks7d2c"`.
    pub board: String,
    /// A second board for double-board pots (same number of cards).
    #[serde(default)]
    pub board2: Option<String>,
    /// One range per player in acting order (first to act first).
    pub ranges: Vec<String>,
    /// Pot at the start of the spot.
    pub pot: f64,
    /// Remaining stack of each player (one value = the same for everyone).
    pub stacks: Vec<f64>,
    #[serde(default)]
    pub menu: BetMenu,
    #[serde(default)]
    pub rake: Option<Rake>,
    #[serde(default)]
    pub side_rule: Option<SideRule>,
    /// Cards known to be out of play (e.g. folded hands' cards revealed).
    #[serde(default)]
    pub dead: Option<String>,
}

/// A validated, prepared spot.
#[derive(Clone, Debug)]
pub struct Spot {
    pub config: SpotConfig,
    pub game: Game,
    pub board: Vec<Card>,
    pub board2: Option<Vec<Card>>,
    pub dead: CardMask,
    pub ranges: Vec<HandList>,
    pub masks: Vec<Vec<CardMask>>,
    pub pot: f64,
    pub stacks: Vec<f64>,
    /// Which hands are seven-deuce offsuit (for the 7-2 rule).
    pub is_72: Vec<Vec<bool>>,
}

impl Spot {
    pub fn players(&self) -> usize {
        self.ranges.len()
    }

    /// All board cards known at the start (both boards).
    pub fn board_mask(&self) -> CardMask {
        mask_of(&self.board) | self.board2.as_ref().map_or(0, |b| mask_of(b))
    }

    pub fn from_config(config: SpotConfig) -> Result<Spot, String> {
        Spot::build(config, None)
    }

    /// A spot whose ranges are given as weighted hand lists (for example ranges
    /// reconstructed from a hand history) instead of range notation; the strings
    /// in `config.ranges` are kept only as descriptions. Hands must not touch the
    /// board or the dead cards.
    pub fn from_lists(config: SpotConfig, lists: Vec<HandList>) -> Result<Spot, String> {
        Spot::build(config, Some(lists))
    }

    fn build(config: SpotConfig, lists: Option<Vec<HandList>>) -> Result<Spot, String> {
        let game = Game::parse(&config.game)?;
        let board = parse_cards(&config.board)?;
        if !(3..=5).contains(&board.len()) {
            return Err("the board needs 3 to 5 cards".into());
        }
        let board2 = match &config.board2 {
            Some(s) if !s.trim().is_empty() => {
                let b = parse_cards(s)?;
                if b.len() != board.len() {
                    return Err("both boards need the same number of cards".into());
                }
                Some(b)
            }
            _ => None,
        };
        let dead = match &config.dead {
            Some(s) if !s.trim().is_empty() => mask_of(&parse_cards(s)?),
            _ => 0,
        };
        let bm = mask_of(&board) | board2.as_ref().map_or(0, |b| mask_of(b));
        if bm.count_ones() as usize != board.len() + board2.as_ref().map_or(0, |b| b.len()) {
            return Err("the boards share a card".into());
        }
        if bm & !game.deck() != 0 {
            return Err(format!("a board card is not in the {} deck", game.code()));
        }
        if bm & dead != 0 {
            return Err("a dead card is on the board".into());
        }
        let n = lists.as_ref().map_or(config.ranges.len(), |l| l.len());
        if !(2..=6).contains(&n) {
            return Err("2 to 6 players".into());
        }
        let ranges = match lists {
            Some(lists) => {
                for (i, l) in lists.iter().enumerate() {
                    if l.is_empty() {
                        return Err(format!("range {} is empty", i + 1));
                    }
                    if l.hands.iter().any(|h| h.len() != game.hole_cards || mask_of(h) & (bm | dead | !game.deck()) != 0) {
                        return Err(format!("range {} holds a hand that touches the board or is not a {} hand", i + 1, game.code()));
                    }
                    if l.weights.iter().any(|w| !(*w >= 0.0) || !w.is_finite()) {
                        return Err(format!("range {} has an invalid weight", i + 1));
                    }
                }
                lists
            }
            None => {
                let ranking = crate::preflop_ranking();
                let mut ranges = Vec::new();
                for (i, r) in config.ranges.iter().enumerate() {
                    let list = parse_hand_list(&game, r, bm | dead, Some(&ranking)).map_err(|e| format!("range {}: {e}", i + 1))?;
                    if list.is_empty() {
                        return Err(format!("range {} is empty on this board", i + 1));
                    }
                    ranges.push(list);
                }
                ranges
            }
        };
        let stacks = match config.stacks.len() {
            0 => return Err("give the remaining stack".into()),
            1 => vec![config.stacks[0]; n],
            k if k == n => config.stacks.clone(),
            _ => return Err("give one stack or one per player".into()),
        };
        if config.pot <= 0.0 || stacks.iter().any(|s| *s < 0.0) {
            return Err("the pot must be positive and stacks non-negative".into());
        }
        let masks = ranges.iter().map(|r| r.masks()).collect();
        let is_72 = ranges
            .iter()
            .map(|r| {
                r.hands
                    .iter()
                    .map(|h| {
                        h.len() == 2 && {
                            let (a, b) = (h[0], h[1]);
                            let mut ranks = [a.rank(), b.rank()];
                            ranks.sort();
                            ranks == [0, 5] && a.suit() != b.suit()
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Spot { pot: config.pot, stacks, game, board, board2, dead, ranges, masks, is_72, config })
    }
}

/// Weighted fraction of a player's starting range (by combos) for reporting.
pub fn range_size(list: &HandList) -> f64 {
    list.weights.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates() {
        let cfg = SpotConfig {
            game: "NLH".into(),
            board: "Ks7d2c9h4s".into(),
            board2: None,
            ranges: vec!["AA,KK,AK,72o".into(), "QQ-TT,AQs".into()],
            pot: 10.0,
            stacks: vec![50.0],
            menu: BetMenu::default(),
            rake: None,
            side_rule: None,
            dead: None,
        };
        let s = Spot::from_config(cfg.clone()).unwrap();
        assert_eq!(s.players(), 2);
        // AA 6, KK 3 (no Ks), AK 12 (no Ks), 72o 7 (no 7d or 2c)
        assert_eq!(s.ranges[0].len(), 28);
        assert!(s.is_72[0].iter().filter(|&&x| x).count() > 0);
        let mut bad = cfg;
        bad.board = "Ks7d".into();
        assert!(Spot::from_config(bad).is_err());
    }
}
