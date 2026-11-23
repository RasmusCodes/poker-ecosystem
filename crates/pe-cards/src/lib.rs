//! Cards, card sets, game variants, ranges and a deterministic RNG.
//!
//! This crate is the foundation of the poker-ecosystem workspace. It has no
//! poker logic beyond notation: evaluation lives in `pe-eval`, equity in
//! `pe-equity`.

pub mod card;
pub mod combos;
pub mod game;
pub mod mask;
pub mod omaha;
pub mod range;
pub mod ranking;
pub mod rng;
pub mod sys;

pub use card::{Card, cards_to_spaced, cards_to_string, parse_cards};
pub use combos::{NUM_CLASSES, NUM_COMBOS, choose, class_name, class_of, combo_cards, combo_index, combo_mask};
pub use game::{Game, HighRules, HoleRule};
pub use mask::{CardMask, FULL_DECK, MaskIter, SHORT_DECK, cards_of, mask_of};
pub use omaha::{HandList, parse_omaha_range};
pub use range::HoldemRange;
pub use rng::Rng;

/// Parses a range for any game into a weighted hand list, removing `dead` cards.
/// Hold'em-style games (two hole cards) use [`HoldemRange`] notation; Omaha games
/// use the pattern language of [`omaha`].
pub fn parse_hand_list(game: &Game, s: &str, dead: CardMask, ranking: Option<&[usize]>) -> Result<HandList, String> {
    let deck = game.deck();
    if game.hole_cards == 2 {
        let mut r = HoldemRange::parse_ranked(s, ranking)?;
        r.remove_dead(dead | !deck);
        let mut out = HandList::default();
        for (i, w) in r.combos() {
            let (a, b) = combo_cards(i);
            out.push(vec![b, a], w);
        }
        Ok(out)
    } else {
        parse_omaha_range(s, game.hole_cards, deck, dead)
    }
}
