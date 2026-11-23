//! A small argument parser: positional words, `--key value` options and `--flag`s.

use std::collections::HashMap;

pub struct Args {
    pub positional: Vec<String>,
    opts: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    pub fn parse(raw: &[String], flag_names: &[&str]) -> Args {
        let mut positional = Vec::new();
        let mut opts = HashMap::new();
        let mut flags = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(key) = a.strip_prefix("--") {
                if let Some((k, v)) = key.split_once('=') {
                    opts.insert(k.to_string(), v.to_string());
                } else if flag_names.contains(&key) || i + 1 >= raw.len() || raw[i + 1].starts_with("--") {
                    flags.push(key.to_string());
                } else {
                    opts.insert(key.to_string(), raw[i + 1].clone());
                    i += 1;
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Args { positional, opts, flags }
    }

    pub fn get(&self, k: &str) -> Option<&str> {
        self.opts.get(k).map(|s| s.as_str())
    }

    pub fn str_or(&self, k: &str, d: &str) -> String {
        self.get(k).unwrap_or(d).to_string()
    }

    pub fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> Result<T, String> {
        match self.get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: cannot parse '{v}'")),
        }
    }

    pub fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}
