//! A small deterministic RNG (xoshiro256** seeded through splitmix64).
//!
//! Every random choice in the engine flows from an explicit seed, so results do
//! not depend on thread scheduling or library versions.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut st = seed;
        let s = [splitmix64(&mut st), splitmix64(&mut st), splitmix64(&mut st), splitmix64(&mut st)];
        Rng { s }
    }

    /// An independent stream for worker `index` derived from `seed`.
    pub fn stream(seed: u64, index: u64) -> Rng {
        Rng::new(seed ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03).rotate_left(17))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `0..n` (Lemire's nearly-divisionless method).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let t = n.wrapping_neg() % n;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Picks an index with probability proportional to `weights` (non-negative, not all zero).
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut x = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                return i;
            }
            x -= w;
        }
        weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_uniform_enough() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(7);
        let mut counts = [0u32; 10];
        for _ in 0..100_000 {
            counts[r.below(10) as usize] += 1;
        }
        for c in counts {
            assert!((9_500..10_500).contains(&c), "{counts:?}");
        }
        let f: f64 = (0..10_000).map(|_| r.next_f64()).sum::<f64>() / 10_000.0;
        assert!((f - 0.5).abs() < 0.02);
    }
}
