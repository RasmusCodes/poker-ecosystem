//! Game variants: which deck, how many hole cards, how hands are evaluated.

use serde::{Deserialize, Serialize};

use crate::mask::{CardMask, FULL_DECK, SHORT_DECK};

/// Ranking rules for high hands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HighRules {
    /// Standard poker: straight flush > quads > full house > flush > straight > trips.
    Standard,
    /// Short deck as played under "Triton" rules: flush > full house and three of a
    /// kind > straight. A-6-7-8-9 is the lowest straight.
    ShortDeckTriton,
    /// Short deck with flush > full house but the usual straight > three of a kind.
    ShortDeckClassic,
}

impl HighRules {
    pub fn is_short_deck(self) -> bool {
        !matches!(self, HighRules::Standard)
    }
}

/// How hole cards combine with the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HoleRule {
    /// Any five of hole cards plus board (hold'em).
    AnyFive,
    /// Exactly two hole cards and three board cards (Omaha).
    TwoPlusThree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Game {
    pub hole_cards: usize,
    pub hole_rule: HoleRule,
    pub high: HighRules,
    /// Split pot with an ace-to-five eight-or-better low.
    pub hilo: bool,
}

impl Game {
    pub const NLH: Game = Game { hole_cards: 2, hole_rule: HoleRule::AnyFive, high: HighRules::Standard, hilo: false };
    pub const PLO: Game = Game { hole_cards: 4, hole_rule: HoleRule::TwoPlusThree, high: HighRules::Standard, hilo: false };
    pub const PLO5: Game = Game { hole_cards: 5, hole_rule: HoleRule::TwoPlusThree, high: HighRules::Standard, hilo: false };
    pub const PLO6: Game = Game { hole_cards: 6, hole_rule: HoleRule::TwoPlusThree, high: HighRules::Standard, hilo: false };
    pub const SHORT_DECK: Game =
        Game { hole_cards: 2, hole_rule: HoleRule::AnyFive, high: HighRules::ShortDeckTriton, hilo: false };
    pub const SHORT_DECK_CLASSIC: Game =
        Game { hole_cards: 2, hole_rule: HoleRule::AnyFive, high: HighRules::ShortDeckClassic, hilo: false };
    pub const O8: Game = Game { hole_cards: 4, hole_rule: HoleRule::TwoPlusThree, high: HighRules::Standard, hilo: true };
    pub const BIG_O: Game = Game { hole_cards: 5, hole_rule: HoleRule::TwoPlusThree, high: HighRules::Standard, hilo: true };

    pub fn deck(&self) -> CardMask {
        if self.high.is_short_deck() { SHORT_DECK } else { FULL_DECK }
    }

    /// Short identifier used by the CLI, the JSON formats and the web demo.
    pub fn code(&self) -> &'static str {
        match (self.hole_cards, self.hole_rule, self.high, self.hilo) {
            (2, HoleRule::AnyFive, HighRules::Standard, false) => "NLH",
            (2, HoleRule::AnyFive, HighRules::ShortDeckTriton, false) => "SD",
            (2, HoleRule::AnyFive, HighRules::ShortDeckClassic, false) => "SDC",
            (4, HoleRule::TwoPlusThree, HighRules::Standard, false) => "PLO",
            (5, HoleRule::TwoPlusThree, HighRules::Standard, false) => "PLO5",
            (6, HoleRule::TwoPlusThree, HighRules::Standard, false) => "PLO6",
            (4, HoleRule::TwoPlusThree, HighRules::Standard, true) => "O8",
            (5, HoleRule::TwoPlusThree, HighRules::Standard, true) => "BIGO",
            _ => "CUSTOM",
        }
    }

    pub fn parse(s: &str) -> Result<Game, String> {
        Ok(match s.trim().to_ascii_uppercase().as_str() {
            "NLH" | "HOLDEM" | "NLHE" | "HE" | "LHE" => Game::NLH,
            "PLO" | "PLO4" | "OMAHA" => Game::PLO,
            "PLO5" | "5CARD" | "5-CARD" => Game::PLO5,
            "PLO6" => Game::PLO6,
            "SD" | "6+" | "SHORTDECK" | "SHORT-DECK" | "SD-TRITON" => Game::SHORT_DECK,
            "SDC" | "SD-CLASSIC" => Game::SHORT_DECK_CLASSIC,
            "O8" | "PLO8" | "OMAHA8" | "PLO-HILO" => Game::O8,
            "BIGO" | "BIG-O" | "BIG O" => Game::BIG_O,
            other => return Err(format!("unknown game '{other}'")),
        })
    }

    pub fn name(&self) -> &'static str {
        match self.code() {
            "NLH" => "Texas Hold'em",
            "SD" => "Short Deck (6+, trips beat straights)",
            "SDC" => "Short Deck (6+, straights beat trips)",
            "PLO" => "Pot-Limit Omaha",
            "PLO5" => "Five-Card Omaha",
            "PLO6" => "Six-Card Omaha",
            "O8" => "Omaha Hi-Lo (8 or better)",
            "BIGO" => "Big O (five-card Omaha Hi-Lo)",
            _ => "Custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for g in [Game::NLH, Game::PLO, Game::PLO5, Game::PLO6, Game::SHORT_DECK, Game::SHORT_DECK_CLASSIC, Game::O8, Game::BIG_O] {
            assert_eq!(Game::parse(g.code()).unwrap(), g);
        }
        assert_eq!(Game::SHORT_DECK.deck().count_ones(), 36);
    }
}
