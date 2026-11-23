//! Cards, ranks and suits.
//!
//! A card is a `u8` id `rank * 4 + suit`, rank 0..=12 for deuce..ace and suit
//! 0..=3 for clubs, diamonds, hearts, spades. Card sets use the suit-major bit
//! layout of [`crate::CardMask`] (bit `suit * 16 + rank`), which lets the
//! evaluators read the four 13-bit suit masks with shifts.

use std::fmt;

pub const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];
pub const RANK_NAMES: [&str; 13] = [
    "Deuce", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace",
];
pub const RANK_PLURALS: [&str; 13] = [
    "Deuces", "Threes", "Fours", "Fives", "Sixes", "Sevens", "Eights", "Nines", "Tens", "Jacks", "Queens", "Kings",
    "Aces",
];

/// A playing card. Ordering is by id (rank first, then suit).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Card(pub u8);

impl Card {
    #[inline]
    pub const fn new(rank: u8, suit: u8) -> Card {
        Card(rank * 4 + suit)
    }
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 2
    }
    #[inline]
    pub const fn suit(self) -> u8 {
        self.0 & 3
    }
    #[inline]
    pub const fn id(self) -> usize {
        self.0 as usize
    }
    /// Bit index of this card in a [`crate::CardMask`].
    #[inline]
    pub const fn bit(self) -> u32 {
        (self.suit() as u32) * 16 + self.rank() as u32
    }
    #[inline]
    pub const fn mask(self) -> u64 {
        1u64 << self.bit()
    }
    /// Inverse of [`Card::bit`].
    #[inline]
    pub const fn from_bit(bit: u32) -> Card {
        Card::new((bit & 15) as u8, (bit >> 4) as u8)
    }
    pub fn rank_char(self) -> char {
        RANK_CHARS[self.rank() as usize]
    }
    pub fn suit_char(self) -> char {
        SUIT_CHARS[self.suit() as usize]
    }
    /// Parses `"Ah"`, `"tc"`, `"10d"`.
    pub fn parse(s: &str) -> Result<Card, String> {
        let s = s.trim();
        let (r, su) = if let Some(rest) = s.strip_prefix("10") {
            ('T', rest)
        } else {
            let mut it = s.chars();
            let r = it.next().ok_or_else(|| "empty card".to_string())?;
            (r, it.as_str())
        };
        let rank = parse_rank(r).ok_or_else(|| format!("bad rank in card '{s}'"))?;
        let mut sc = su.chars();
        let suit_c = sc.next().ok_or_else(|| format!("missing suit in card '{s}'"))?;
        if sc.next().is_some() {
            return Err(format!("trailing characters in card '{s}'"));
        }
        let suit = parse_suit(suit_c).ok_or_else(|| format!("bad suit in card '{s}'"))?;
        Ok(Card::new(rank, suit))
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank_char(), self.suit_char())
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank_char(), self.suit_char())
    }
}

impl serde::Serialize for Card {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for Card {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(d)?;
        Card::parse(&s).map_err(serde::de::Error::custom)
    }
}

pub fn parse_rank(c: char) -> Option<u8> {
    match c.to_ascii_uppercase() {
        '2'..='9' => Some(c as u8 - b'2'),
        'T' => Some(8),
        'J' => Some(9),
        'Q' => Some(10),
        'K' => Some(11),
        'A' => Some(12),
        _ => None,
    }
}

pub fn parse_suit(c: char) -> Option<u8> {
    match c.to_ascii_lowercase() {
        'c' => Some(0),
        'd' => Some(1),
        'h' => Some(2),
        's' => Some(3),
        _ => None,
    }
}

/// Parses a run of cards with or without separators: `"AhKd"`, `"Ah Kd"`, `"[Ah Kd]"`, `"Ah,Kd"`.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, String> {
    let cleaned: String = s.chars().filter(|c| !matches!(c, '[' | ']' | ',' | ' ' | '\t' | '-')).collect();
    let chars: Vec<char> = cleaned.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() + 1 && chars[i] == '1' && chars.get(i + 1) == Some(&'0') {
            let suit = chars.get(i + 2).copied().ok_or("missing suit")?;
            out.push(Card::parse(&format!("T{suit}"))?);
            i += 3;
            continue;
        }
        if i + 1 >= chars.len() {
            return Err(format!("dangling character '{}' in '{s}'", chars[i]));
        }
        out.push(Card::parse(&format!("{}{}", chars[i], chars[i + 1]))?);
        i += 2;
    }
    let mut seen = 0u64;
    for c in &out {
        if seen & c.mask() != 0 {
            return Err(format!("duplicate card {c} in '{s}'"));
        }
        seen |= c.mask();
    }
    Ok(out)
}

pub fn cards_to_string(cards: &[Card]) -> String {
    cards.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("")
}

pub fn cards_to_spaced(cards: &[Card]) -> String {
    cards.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display_round_trip() {
        for id in 0..52u8 {
            let c = Card(id);
            assert_eq!(Card::parse(&c.to_string()).unwrap(), c);
            assert_eq!(Card::from_bit(c.bit()), c);
        }
        assert_eq!(Card::parse("10d").unwrap().to_string(), "Td");
        assert!(Card::parse("Ax").is_err());
        assert!(Card::parse("1h").is_err());
    }

    #[test]
    fn parse_runs() {
        let v = parse_cards("[Ah Kd] 10c").unwrap();
        assert_eq!(cards_to_string(&v), "AhKdTc");
        assert!(parse_cards("AhAh").is_err());
        assert!(parse_cards("AhK").is_err());
    }
}
