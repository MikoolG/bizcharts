//! Coin ticker extraction from post text

use regex::Regex;
use std::collections::HashSet;
use once_cell::sync::Lazy;

use crate::db::CoinMention;

// Common cryptocurrency tickers
static KNOWN_TICKERS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // Top coins
        "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "LINK",
        "UNI", "ATOM", "LTC", "BCH", "NEAR", "APT", "ARB", "OP", "INJ", "SUI",
        "SEI", "TIA", "PEPE", "SHIB", "WIF", "BONK", "FLOKI", "MEME", "WOJAK",
        // DeFi
        "AAVE", "MKR", "SNX", "COMP", "CRV", "SUSHI", "YFI", "BAL", "LDO",
        // Layer 2
        "IMX", "STRK", "ZK", "MANTA", "BLAST",
        // AI coins
        "FET", "AGIX", "OCEAN", "RNDR", "TAO",
        // Gaming
        "AXS", "SAND", "MANA", "ENJ", "GALA", "IMX",
        // Exchange tokens
        "BNB", "FTT", "CRO", "OKB", "GT", "KCS",
    ]
    .into_iter()
    .collect()
});

// Full name to ticker mapping
static NAME_MAPPINGS: Lazy<Vec<(&'static str, &'static str)>> = Lazy::new(|| {
    vec![
        ("bitcoin", "BTC"),
        ("ethereum", "ETH"),
        ("solana", "SOL"),
        ("ripple", "XRP"),
        ("cardano", "ADA"),
        ("dogecoin", "DOGE"),
        ("polkadot", "DOT"),
        ("avalanche", "AVAX"),
        ("polygon", "MATIC"),
        ("chainlink", "LINK"),
        ("uniswap", "UNI"),
        ("cosmos", "ATOM"),
        ("litecoin", "LTC"),
        ("near protocol", "NEAR"),
        ("aptos", "APT"),
        ("arbitrum", "ARB"),
        ("optimism", "OP"),
        ("injective", "INJ"),
        ("celestia", "TIA"),
    ]
});

// Regex patterns
static DOLLAR_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\$([A-Z]{2,10})\b").unwrap()
});

static STANDALONE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b([A-Z]{2,6})\b").unwrap()
});

/// Extract coin mentions from post text
pub fn extract_coins(text: &str) -> Vec<CoinMention> {
    let mut mentions = Vec::new();
    let mut seen = HashSet::new();

    // Pattern 1: $SYMBOL format (highest confidence)
    for cap in DOLLAR_PATTERN.captures_iter(text) {
        let symbol = &cap[1];
        if KNOWN_TICKERS.contains(symbol) && !seen.contains(symbol) {
            seen.insert(symbol.to_string());
            mentions.push(CoinMention {
                symbol: symbol.to_string(),
                name: None,
                confidence: 0.95,
            });
        }
    }

    // Pattern 2: Standalone known tickers (medium confidence)
    for cap in STANDALONE_PATTERN.captures_iter(text) {
        let symbol = &cap[1];
        if KNOWN_TICKERS.contains(symbol) && !seen.contains(symbol) {
            seen.insert(symbol.to_string());
            mentions.push(CoinMention {
                symbol: symbol.to_string(),
                name: None,
                confidence: 0.75,
            });
        }
    }

    // Pattern 3: Full names (good confidence)
    let text_lower = text.to_lowercase();
    for (name, symbol) in NAME_MAPPINGS.iter() {
        if text_lower.contains(name) && !seen.contains(*symbol) {
            seen.insert(symbol.to_string());
            mentions.push(CoinMention {
                symbol: symbol.to_string(),
                name: Some(name.to_string()),
                confidence: 0.85,
            });
        }
    }

    mentions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dollar_pattern() {
        let mentions = extract_coins("I'm bullish on $BTC and $ETH");
        assert_eq!(mentions.len(), 2);
        assert!(mentions.iter().any(|m| m.symbol == "BTC"));
        assert!(mentions.iter().any(|m| m.symbol == "ETH"));
    }

    #[test]
    fn test_standalone_pattern() {
        let mentions = extract_coins("BTC is going to moon, ETH too");
        assert_eq!(mentions.len(), 2);
    }

    #[test]
    fn test_full_name() {
        let mentions = extract_coins("I love bitcoin and ethereum");
        assert_eq!(mentions.len(), 2);
        assert!(mentions.iter().any(|m| m.symbol == "BTC"));
        assert!(mentions.iter().any(|m| m.symbol == "ETH"));
    }

    #[test]
    fn test_no_duplicates() {
        let mentions = extract_coins("$BTC BTC bitcoin Bitcoin BITCOIN");
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].symbol, "BTC");
    }

    #[test]
    fn test_unknown_ticker() {
        let mentions = extract_coins("$FAKECOIN is a scam");
        assert!(mentions.is_empty());
    }
}
