use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;


/// GPT-4 tokenization pattern adapted for Rust regex (without lookahead)
const GPT4_SPLIT_PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+";

/// Calculate pair frequencies in a sequence of tokens
fn pair_freq(text_bytes: &[u32], pair_freq_counts: Option<&mut HashMap<(u32, u32), usize>>) -> HashMap<(u32, u32), usize> {
    let mut freq = if let Some(counts) = pair_freq_counts {
        counts.clone()
    } else {
        HashMap::new()
    };

    for window in text_bytes.windows(2) {
        let pair = (window[0], window[1]);
        *freq.entry(pair).or_insert(0) += 1;
    }

    freq
}

/// Increase frequency count for a pair
fn increase_frequency(freq: &mut HashMap<(u32, u32), usize>, pair: (u32, u32)) {
    *freq.entry(pair).or_insert(0) += 1;
}

/// Decrease frequency count for a pair
fn decrease_frequency(freq: &mut HashMap<(u32, u32), usize>, pair: (u32, u32)) {
    if let Some(count) = freq.get_mut(&pair) {
        if *count > 0 {
            *count -= 1;
        }
    }
}

/// Replace control characters with their Unicode escape sequences
fn replace_control_characters(s: &str) -> String {
    s.chars()
        .map(|ch| {
            if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
                format!("\\u{:04x}", ch as u32)
            } else {
                ch.to_string()
            }
        })
        .collect()
}

/// Render a token as a string, escaping control characters
fn render_token(token: &[u8]) -> String {
    let s = String::from_utf8_lossy(token);
    replace_control_characters(&s)
}

/// Naive merge implementation - linear search through tokens
fn merge_naive(text_bytes: &[u32], pair: (u32, u32), replacement_id: u32) -> Vec<u32> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < text_bytes.len() {
        if i < text_bytes.len() - 1 && text_bytes[i] == pair.0 && text_bytes[i + 1] == pair.1 {
            result.push(replacement_id);
            i += 2;
        } else {
            result.push(text_bytes[i]);
            i += 1;
        }
    }

    result
}

/// Optimized merge implementation with frequency tracking
fn merge(freq: &mut HashMap<(u32, u32), usize>, text_bytes: &[u32], pair: (u32, u32), replacement_id: u32) -> Vec<u32> {
    let mut result = Vec::new();
    freq.insert((replacement_id, replacement_id), 0);
    let mut i = 0;

    while i < text_bytes.len() {
        if i < text_bytes.len() - 1 && text_bytes[i] == pair.0 && text_bytes[i + 1] == pair.1 {
            result.push(replacement_id);

            // Update frequencies for left neighbor
            if i >= 1 {
                let lnpo = (text_bytes[i - 1], text_bytes[i]);
                let lrp = (text_bytes[i - 1], replacement_id);
                decrease_frequency(freq, lnpo);

                if i >= 2 && text_bytes[i - 2] == pair.0 && text_bytes[i - 1] == pair.1 {
                    decrease_frequency(freq, lnpo);
                    decrease_frequency(freq, (replacement_id, text_bytes[i]));
                } else {
                    increase_frequency(freq, lrp);
                }
            }

            // Update frequencies for right neighbor
            if i + 3 < text_bytes.len() {
                let rnpo = (text_bytes[i + 1], text_bytes[i + 2]);
                let rrp = (replacement_id, text_bytes[i + 2]);
                decrease_frequency(freq, rnpo);

                if text_bytes[i + 2] != pair.0 || text_bytes[i + 3] != pair.1 {
                    increase_frequency(freq, rrp);
                } else {
                    increase_frequency(freq, (replacement_id, replacement_id));
                }
            } else if i + 2 < text_bytes.len() {
                let rnpo = (text_bytes[i + 1], text_bytes[i + 2]);
                let rrp = (replacement_id, text_bytes[i + 2]);
                decrease_frequency(freq, rnpo);
                increase_frequency(freq, rrp);
            }

            i += 2;
        } else {
            result.push(text_bytes[i]);
            i += 1;
        }
    }

    freq.insert(pair, 0);
    result
}

/// Byte Pair Encoding (BPE) Tokenizer
///
/// This tokenizer implements BPE algorithm for text compression and tokenization.
/// It supports both naive and optimized training methods, and can save/load models.
pub struct Tokenizer {
    pattern: String,
    compiled_pattern: Regex,
    merges: HashMap<(u32, u32), u32>,
    special_tokens: HashMap<String, u32>,
    vocab: HashMap<u32, Vec<u8>>,
    mergeable_ranks: HashMap<(u32, u32), usize>,
}

impl Tokenizer {
    /// Create a new tokenizer with optional regex pattern
    pub fn new(pattern: Option<String>) -> Self {
        let pattern = pattern.unwrap_or_else(|| GPT4_SPLIT_PATTERN.to_string());
        let compiled_pattern = Regex::new(&pattern).expect("Invalid regex pattern");

        let mut tokenizer = Tokenizer {
            pattern,
            compiled_pattern,
            merges: HashMap::new(),
            special_tokens: HashMap::new(),
            vocab: HashMap::new(),
            mergeable_ranks: HashMap::new(),
        };

        tokenizer.vocab = tokenizer.build_vocab();
        tokenizer
    }

    /// Build vocabulary from merges and special tokens
    fn build_vocab(&self) -> HashMap<u32, Vec<u8>> {
        let mut vocab = HashMap::new();

        // Initialize with single bytes
        for idx in 0..256u32 {
            vocab.insert(idx, vec![idx as u8]);
        }

        // Sort merges by their token index to apply them in order
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &idx)| idx);

        // Apply merges in order
        for ((p0, p1), idx) in sorted_merges {
            if let (Some(bytes0), Some(bytes1)) = (vocab.get(p0), vocab.get(p1)) {
                let mut merged = bytes0.clone();
                merged.extend(bytes1);
                vocab.insert(*idx, merged);
            }
        }

        // Add special tokens
        for (special, idx) in &self.special_tokens {
            vocab.insert(*idx, special.as_bytes().to_vec());
        }

        vocab
    }

    /// Train tokenizer using naive method (O(3mn) time complexity)
    ///
    /// This method recalculates pair frequencies after each merge
    pub fn train_naive(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        assert!(vocab_size >= 256, "Vocab size must be at least 256");

        let num_merges = vocab_size - self.vocab.len();

        // Split text into chunks using regex pattern
        let text_chunks: Vec<&str> = self.compiled_pattern.find_iter(text)
            .map(|m| m.as_str())
            .collect();

        // Convert to token sequences
        let mut ids: Vec<Vec<u32>> = text_chunks.iter()
            .map(|chunk| chunk.bytes().map(|b| b as u32).collect())
            .collect();

        println!("Training on {} chunks of text...", ids.len());

        let pb = ProgressBar::new((vocab_size - 256) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"));
        pb.set_message("Training BPE");

        let mut tmp_vocab = self.vocab.clone();
        let mut tmp_merges = self.merges.clone();
        let mut i = 0;

        while tmp_vocab.len() < vocab_size {
            // Calculate pair frequencies
            let mut text_bytes_freq = HashMap::new();
            for chunk_ids in &ids {
                text_bytes_freq = pair_freq(chunk_ids, Some(&mut text_bytes_freq));
            }

            if text_bytes_freq.is_empty() {
                break;
            }

            // Find most frequent pair
            let freq_pair = text_bytes_freq
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(pair, _)| *pair)
                .unwrap();

            let idx = 256 + i as u32;

            // Merge the pair in all chunks using parallel processing
            ids = ids.par_iter()
                .map(|chunk_ids| merge_naive(chunk_ids, freq_pair, idx))
                .collect();

            tmp_merges.insert(freq_pair, idx);
            if let (Some(bytes0), Some(bytes1)) = (tmp_vocab.get(&freq_pair.0), tmp_vocab.get(&freq_pair.1)) {
                let mut merged = bytes0.clone();
                merged.extend(bytes1);
                tmp_vocab.insert(idx, merged);
            }

            if verbose {
                println!(
                    "merge {}/{}: {:?} -> {} (had {} occurrences)",
                    i + 1,
                    num_merges,
                    freq_pair,
                    idx,
                    text_bytes_freq[&freq_pair]
                );
            }

            pb.inc(1);
            i += 1;
        }

        pb.finish_with_message("Training complete");

        self.merges = tmp_merges;
        self.vocab = tmp_vocab;
    }

    /// Train tokenizer using optimized method (O(2mn) time complexity)
    ///
    /// This method tracks frequency changes during merging
    pub fn train(&mut self, text: &str, vocab_size: usize, _verbose: bool) {
        // Split text into chunks
        let text_chunks: Vec<&str> = self.compiled_pattern.find_iter(text)
            .map(|m| m.as_str())
            .collect();

        // Convert to token sequences
        let mut ids: Vec<Vec<u32>> = text_chunks.iter()
            .map(|chunk| chunk.bytes().map(|b| b as u32).collect())
            .collect();

        let mut tmp_vocab = self.vocab.clone();
        let mut tmp_merges = self.merges.clone();

        println!("Training on {} chunks of text...", ids.len());

        let pb = ProgressBar::new((vocab_size - 256) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"));
        pb.set_message("Training BPE");

        // Calculate initial pair frequencies
        self.mergeable_ranks.clear();
        for chunk_ids in &ids {
            self.mergeable_ranks = pair_freq(chunk_ids, Some(&mut self.mergeable_ranks));
        }

        let mut index_i = 0;

        while index_i < vocab_size - 256 {
            // Find most frequent pair
            let freq_pair = match self.mergeable_ranks
                .iter()
                .max_by_key(|&(_, &count)| count) {
                Some((pair, &count)) if count > 1 => *pair,
                _ => break,
            };

            let idx = 256 + index_i as u32;

            // Merge in all chunks and update frequencies
            ids = ids.into_iter()
                .map(|chunk_ids| merge(&mut self.mergeable_ranks, &chunk_ids, freq_pair, idx))
                .collect();

            tmp_merges.insert(freq_pair, idx);
            if let (Some(bytes0), Some(bytes1)) = (tmp_vocab.get(&freq_pair.0), tmp_vocab.get(&freq_pair.1)) {
                let mut merged = bytes0.clone();
                merged.extend(bytes1);
                tmp_vocab.insert(idx, merged);
            }

            pb.inc(1);
            index_i += 1;
        }

        pb.finish_with_message("Training complete");

        self.merges = tmp_merges;
        self.vocab = tmp_vocab;
    }

    /// Encode a chunk of bytes into tokens
    fn encode_chunk(&self, text_bytes: &[u8]) -> Vec<u32> {
        // For HuggingFace tokenizers, we need to map bytes to their token IDs
        // Build a reverse mapping from bytes to token IDs
        let mut byte_to_token_id: HashMap<u8, u32> = HashMap::new();

        // Find the token IDs for single-byte tokens (0-255)
        for (token_id, token_bytes) in &self.vocab {
            if token_bytes.len() == 1 && *token_id < 256 {
                byte_to_token_id.insert(token_bytes[0], *token_id);
            }
        }

        // Convert bytes to their corresponding token IDs
        let mut ids: Vec<u32> = Vec::new();
        for &byte in text_bytes {
            if let Some(&token_id) = byte_to_token_id.get(&byte) {
                ids.push(token_id);
            } else {
                // Fallback to byte value if no mapping found
                ids.push(byte as u32);
            }
        }



        // Apply merges iteratively
        while ids.len() >= 2 {
            let freq = pair_freq(&ids, None);
            if freq.is_empty() {
                break;
            }

            // Find the pair with minimum merge index (earliest merge)
            let recent_pair = freq.keys()
                .filter_map(|pair| self.merges.get(pair).map(|&idx| (pair, idx)))
                .min_by_key(|(_, idx)| *idx)
                .map(|(pair, _)| pair)
                .cloned();

            if let Some(pair) = recent_pair {
                if let Some(&idx) = self.merges.get(&pair) {
                    ids = merge_naive(&ids, pair, idx);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        ids
    }

    /// Encode text into a sequence of tokens
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // For HuggingFace tokenizers, we need to handle byte-level encoding
        let text_chunks: Vec<&str> = self.compiled_pattern.find_iter(text)
            .map(|m| m.as_str())
            .collect();

        let mut ids = Vec::new();
        for chunk in text_chunks {
            let chunk_bytes = chunk.as_bytes();
            let chunk_ids = self.encode_chunk(chunk_bytes);
            ids.extend(chunk_ids);
        }

        ids
    }

    /// Decode a sequence of tokens back into text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &id in ids {
            if let Some(token_bytes) = self.vocab.get(&id) {
                bytes.extend(token_bytes);
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Save tokenizer to model and vocab files
    pub fn save(&self, file_prefix: &str) -> std::io::Result<()> {
        // Save model file (for loading)
        let model_file = format!("{}.model", file_prefix);
        let mut f = File::create(&model_file)?;

        writeln!(f, "bpetokenizer v1")?;
        writeln!(f, "{}", self.pattern)?;
        writeln!(f, "{}", self.special_tokens.len())?;

        for (special, idx) in &self.special_tokens {
            writeln!(f, "{} {}", special, idx)?;
        }

        // Write merges with their corresponding token indices
        for ((idx1, idx2), token_idx) in &self.merges {
            writeln!(f, "{} {} {}", idx1, idx2, token_idx)?;
        }

        // Save vocab file (for human inspection)
        let vocab_file = format!("{}.vocab", file_prefix);
        let mut f = File::create(&vocab_file)?;

        let inverted_merges: HashMap<u32, (u32, u32)> = self.merges
            .iter()
            .map(|((p0, p1), idx)| (*idx, (*p0, *p1)))
            .collect();

        // Sort vocab entries by index
        let mut sorted_vocab: Vec<_> = self.vocab.iter().collect();
        sorted_vocab.sort_by_key(|&(&idx, _)| idx);

        for (idx, token) in sorted_vocab {
            let s = render_token(token);

            if let Some((idx0, idx1)) = inverted_merges.get(idx) {
                if let (Some(token0), Some(token1)) = (self.vocab.get(idx0), self.vocab.get(idx1)) {
                    let s0 = render_token(token0);
                    let s1 = render_token(token1);
                    writeln!(f, "[{}][{}] -> [{}] {}", s0, s1, s, idx)?;
                } else {
                    writeln!(f, "[{}] {}", s, idx)?;
                }
            } else {
                writeln!(f, "[{}] {}", s, idx)?;
            }
        }

        Ok(())
    }

    /// Load tokenizer from a model file
    pub fn load(&mut self, model_file: &str) -> std::io::Result<()> {
        assert!(model_file.ends_with(".model"), "Model file must end with .model");

        let file = File::open(model_file)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read version
        let version = lines.next().unwrap()?;
        assert_eq!(version, "bpetokenizer v1");

        // Read pattern
        self.pattern = lines.next().unwrap()?;
        self.compiled_pattern = Regex::new(&self.pattern).expect("Invalid regex pattern");

        // Read special tokens
        let num_special: usize = lines.next().unwrap()?.parse().unwrap();
        self.special_tokens.clear();

        for _ in 0..num_special {
            let line = lines.next().unwrap()?;
            let parts: Vec<&str> = line.split(' ').collect();
            self.special_tokens.insert(parts[0].to_string(), parts[1].parse().unwrap());
        }

        // Read merges
        self.merges.clear();
        let mut idx = 256u32;

        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() >= 3 {
                let idx1: u32 = parts[0].parse().unwrap();
                let idx2: u32 = parts[1].parse().unwrap();
                let token_idx: u32 = parts[2].parse().unwrap();
                self.merges.insert((idx1, idx2), token_idx);
            } else if parts.len() >= 2 {
                // Fallback for old format without explicit token indices
                let idx1: u32 = parts[0].parse().unwrap();
                let idx2: u32 = parts[1].parse().unwrap();
                self.merges.insert((idx1, idx2), idx);
                idx += 1;
            }
        }

        self.vocab = self.build_vocab();
        Ok(())
    }

    /// Display tokenizer statistics
    pub fn stats(&self) {
        println!("\n=== Tokenizer Statistics ===");
        println!("Vocabulary size: {}", self.vocab.len());
        println!("Number of merges: {}", self.merges.len());
        println!("Number of special tokens: {}", self.special_tokens.len());
        println!("Pattern: {}...", &self.pattern[..50.min(self.pattern.len())]);

        // Show some example tokens
        println!("\nExample tokens (last 10):");
        let mut vocab_items: Vec<_> = self.vocab.iter().collect();
        vocab_items.sort_by_key(|&(&k, _)| k);
        for &(idx, bytes) in vocab_items.iter().rev().take(10) {
            println!("  Token {}: {}", idx, render_token(bytes));
        }
    }

    /// Load vocabulary and merges from a HuggingFace tokenizer.json file or URL
    pub fn load_from_huggingface(&mut self, source: &str) -> std::io::Result<()> {
        // Determine if source is a URL or local path
        let tokenizer_json: Value = if source.starts_with("http://") || source.starts_with("https://") {
            // Download from URL
            println!("Downloading tokenizer from URL: {}", source);
            let response = ureq::get(source)
                .call()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            let reader = response.into_reader();
            serde_json::from_reader(reader)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        } else {
            // Read from local file
            let file = File::open(source)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        };

        // Check tokenizer type
        let tokenizer_type = tokenizer_json
            .get("model")
            .and_then(|m| m.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or_else(|| {
                // If no type field, try to infer from the structure
                if tokenizer_json.get("model")
                    .and_then(|m| m.get("merges"))
                    .is_some() {
                    "BPE"
                } else {
                    "Unknown"
                }
            });

        println!("Loading HuggingFace {} tokenizer", tokenizer_type);

        // Currently only support BPE tokenizers
        if tokenizer_type != "BPE" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported tokenizer type: {}. Currently only BPE tokenizers are supported.", tokenizer_type)
            ));
        }

        // Clear existing data
        self.merges.clear();
        self.special_tokens.clear();
        self.vocab.clear();

        // Update regex pattern based on pre-tokenizer configuration
        if let Some(pre_tokenizer) = tokenizer_json.get("pre_tokenizer") {
            if let Some(pre_type) = pre_tokenizer.get("type").and_then(|t| t.as_str()) {
                match pre_type {
                    "ByteLevel" => {
                        // Use GPT-4 pattern for byte-level tokenization
                        self.pattern = GPT4_SPLIT_PATTERN.to_string();
                    }
                    _ => {
                        // Keep current pattern or use a generic one
                        println!("Warning: Pre-tokenizer type '{}' may not be fully supported", pre_type);
                    }
                }
            }
        }
        self.compiled_pattern = Regex::new(&self.pattern).expect("Invalid regex pattern");

        // Build byte-level mapping for HuggingFace encoding
        let byte_to_unicode = self.build_byte_to_unicode_map();
        let unicode_to_byte: HashMap<String, u8> = byte_to_unicode
            .iter()
            .map(|(byte, unicode)| (unicode.clone(), *byte))
            .collect();

        // Load vocabulary and merges from HuggingFace format
        if let Some(model) = tokenizer_json.get("model") {
            // First pass: build string to ID mapping from vocab
            let mut token_str_to_id: HashMap<String, u32> = HashMap::new();

            if let Some(vocab) = model.get("vocab").and_then(|v| v.as_object()) {
                for (token_str, id_val) in vocab {
                    if let Some(id) = id_val.as_u64() {
                        let id = id as u32;
                        token_str_to_id.insert(token_str.clone(), id);
                    }
                }
                println!("Found {} vocabulary entries in HuggingFace tokenizer", vocab.len());
            }

            // Load merges and build our internal merge structure
            if let Some(merges) = model.get("merges").and_then(|v| v.as_array()) {
                println!("Processing {} merges from HuggingFace tokenizer", merges.len());

                for (_merge_idx, merge) in merges.iter().enumerate() {
                    let (token1_str, token2_str) = if let Some(merge_arr) = merge.as_array() {
                        // Array format: ["token1", "token2"]
                        if merge_arr.len() == 2 {
                            match (merge_arr[0].as_str(), merge_arr[1].as_str()) {
                                (Some(t1), Some(t2)) => (t1, t2),
                                _ => continue,
                            }
                        } else {
                            continue;
                        }
                    } else if let Some(merge_str) = merge.as_str() {
                        // String format: "token1 token2"
                        let parts: Vec<&str> = merge_str.split(' ').collect();
                        if parts.len() == 2 {
                            (parts[0], parts[1])
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                    let merged_str = format!("{}{}", token1_str, token2_str);

                    // Look up the IDs
                    if let (Some(&id1), Some(&id2), Some(&merged_id)) = (
                        token_str_to_id.get(token1_str),
                        token_str_to_id.get(token2_str),
                        token_str_to_id.get(&merged_str)
                    ) {
                        self.merges.insert((id1, id2), merged_id);

                        // Build vocab entry for merged token
                        if let (Some(bytes1), Some(bytes2)) = (
                            self.decode_hf_token_to_bytes(token1_str, &unicode_to_byte),
                            self.decode_hf_token_to_bytes(token2_str, &unicode_to_byte)
                        ) {
                            let mut merged_bytes = bytes1;
                            merged_bytes.extend(bytes2);
                            self.vocab.insert(merged_id, merged_bytes);
                        }
                    }
                }
                println!("Loaded {} valid merges", self.merges.len());
            }

            // Build vocabulary for all tokens by decoding their byte representation
            for (token_str, &id) in &token_str_to_id {
                if let Some(bytes) = self.decode_hf_token_to_bytes(token_str, &unicode_to_byte) {
                    self.vocab.insert(id, bytes);
                }
            }
        }

        // Handle special/added tokens
        if let Some(added_tokens) = tokenizer_json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                if let (Some(content), Some(id)) = (
                    token.get("content").and_then(|v| v.as_str()),
                    token.get("id").and_then(|v| v.as_u64())
                ) {
                    let id = id as u32;
                    self.special_tokens.insert(content.to_string(), id);
                    self.vocab.insert(id, content.as_bytes().to_vec());
                }
            }
            println!("Loaded {} special tokens", self.special_tokens.len());
        }

        println!("\nSuccessfully loaded HuggingFace {} tokenizer:", tokenizer_type);
        println!("  - Vocabulary size: {}", self.vocab.len());
        println!("  - Number of merges: {}", self.merges.len());
        println!("  - Special tokens: {}", self.special_tokens.len());
        println!("  - Source: {}", source);

        Ok(())
    }

    /// Build the byte-to-unicode mapping used by HuggingFace
    fn build_byte_to_unicode_map(&self) -> HashMap<u8, String> {
        let mut byte_to_unicode = HashMap::new();

        // GPT-2 uses specific ranges and mappings
        let mut bytes: Vec<u8> = Vec::new();

        // Add printable ASCII
        bytes.extend(b'!'..=b'~');
        bytes.extend(161u8..=172u8);
        bytes.extend(174u8..=255u8);

        let mut chars: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        // For remaining bytes, use Unicode Private Use Area starting at 256
        let mut n = 0u32;
        for byte in 0..=255u8 {
            if !bytes.contains(&byte) {
                bytes.push(byte);
                chars.push(256 + n);
                n += 1;
            }
        }

        // Build the mapping
        for (i, &byte) in bytes.iter().enumerate() {
            byte_to_unicode.insert(byte, char::from_u32(chars[i]).unwrap().to_string());
        }

        byte_to_unicode
    }

    /// Decode a HuggingFace token string back to bytes
    fn decode_hf_token_to_bytes(&self, token: &str, unicode_to_byte: &HashMap<String, u8>) -> Option<Vec<u8>> {
        let mut bytes = Vec::new();

        for ch in token.chars() {
            // Check if this character maps to a byte
            let ch_str = ch.to_string();
            if let Some(&byte) = unicode_to_byte.get(&ch_str) {
                bytes.push(byte);
            } else if ch.is_ascii() && !unicode_to_byte.values().any(|&b| b == ch as u8) {
                // For ASCII characters that aren't remapped, use them directly
                bytes.push(ch as u8);
            } else {
                // For any character we can't decode, fail the entire token
                return None;
            }
        }

        Some(bytes)
    }
}

/// Check if a file exists locally
fn check_file_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

fn main() {
    println!("=== Rust BPE Tokenizer Implementation ===\n");

    // Training data
    let training_text = "Hello world! This is a test. Hello world! Hello world! \
                         The quick brown fox jumps over the lazy dog. \
                         Machine learning is fascinating. \
                         Rust is a systems programming language.";

    // Create and train tokenizer
    let mut tokenizer = Tokenizer::new(None);
    println!("Training tokenizer on sample text...");
    tokenizer.train_naive(training_text, 300, false);
    tokenizer.stats();

    // Test encoding and decoding
    println!("\n=== Encoding/Decoding Test ===");
    let test_texts = vec![
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Rust programming",
        "New text that wasn't in training!",
    ];

    for text in &test_texts {
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        println!("Original: \"{}\"", text);
        println!("Encoded:  {:?} ({} tokens)", encoded, encoded.len());
        println!("Decoded:  \"{}\"", decoded);
        println!("Lossless: {}\n", text == &decoded);
    }

    // Compression statistics
    println!("=== Compression Statistics ===");
    let original_bytes = training_text.len();
    let encoded_tokens = tokenizer.encode(training_text).len();
    let compression_ratio = original_bytes as f64 / encoded_tokens as f64;
    println!("Original text: {} bytes", original_bytes);
    println!("Encoded tokens: {} tokens", encoded_tokens);
    println!("Compression ratio: {:.2}x", compression_ratio);

    // Save and load test
    println!("\n=== Save/Load Test ===");
    if let Err(e) = tokenizer.save("rust_tokenizer") {
        eprintln!("Failed to save tokenizer: {}", e);
    } else {
        println!("Tokenizer saved to rust_tokenizer.model");

        let mut loaded = Tokenizer::new(None);
        if let Err(e) = loaded.load("rust_tokenizer.model") {
            eprintln!("Failed to load tokenizer: {}", e);
        } else {
            println!("Tokenizer loaded successfully");

            // Verify loaded tokenizer
            let test_text = "Hello world!";
            let original_encoding = tokenizer.encode(test_text);
            let loaded_encoding = loaded.encode(test_text);
            let loaded_decoding = loaded.decode(&loaded_encoding);

            println!("\nVerification:");
            println!("Original: {} -> {:?}", test_text, original_encoding);
            println!("Loaded:   {} -> {:?} -> {}", test_text, loaded_encoding, loaded_decoding);
            println!("Match: {}", original_encoding == loaded_encoding && test_text == loaded_decoding);
        }

        // Clean up
        let _ = std::fs::remove_file("rust_tokenizer.model");
        let _ = std::fs::remove_file("rust_tokenizer.vocab");
    }

    // Test loading from HuggingFace tokenizer.json (if available)
    println!("\n=== HuggingFace Tokenizer Loading Test ===");

    // Test with different sources
    let test_sources = vec![
        ("local", "gpt2_tokenizer.json"),
        ("url", "https://huggingface.co/openai/gpt-oss-120b/resolve/main/tokenizer.json"),
    ];

    for (source_type, source) in test_sources {
        if source_type == "local" && !check_file_exists(source) {
            println!("\nSkipping local file test - file not found: {}", source);
            continue;
        }

        println!("\nTesting {} source: {}", source_type, source);
        let mut hf_tokenizer = Tokenizer::new(None);
        match hf_tokenizer.load_from_huggingface(source) {
            Ok(_) => {
                println!("Successfully loaded HuggingFace tokenizer!");
                hf_tokenizer.stats();

                // Test encoding/decoding
                let test_texts = vec![
                    "Hello world!",
                    "Hello world! This is a test.",
                    "The quick brown fox",
                    "gpt-oss-120b tokenizer",
                ];

                for test_text in &test_texts {
                    let encoded = hf_tokenizer.encode(test_text);
                    let decoded = hf_tokenizer.decode(&encoded);
                    println!("\nTest encoding/decoding:");
                    println!("Original: \"{}\"", test_text);
                    println!("Encoded: {:?} ({} tokens)", encoded, encoded.len());
                    println!("Decoded: \"{}\"", decoded);
                    println!("Lossless: {}", test_text == &decoded);
                }
            }
            Err(e) => {
                println!("Could not load HuggingFace tokenizer: {}", e);
            }
        }

        // Only test URL on first source to avoid redundant downloads
        if source_type == "url" {
            break;
        }
    }

    println!("\n=== Test Complete ===");
}
