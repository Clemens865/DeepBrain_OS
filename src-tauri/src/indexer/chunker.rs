//! Text chunker for DeepBrain
//!
//! Splits text into overlapping chunks for embedding.

/// Split text into chunks of approximately `chunk_size` tokens
/// with `overlap` token overlap between consecutive chunks.
///
/// Uses word boundaries for natural splits.
/// Token count is approximated as word count (roughly 0.75 tokens per word).
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return Vec::new();
    }

    // If text is small enough for one chunk, return as-is
    if words.len() <= chunk_size {
        return vec![words.join(" ")];
    }

    let mut chunks = Vec::new();
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");

        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }

        start += step;

        // Avoid tiny trailing chunks
        if start + overlap >= words.len() && start < words.len() {
            let final_chunk = words[start..].join(" ");
            if !final_chunk.trim().is_empty() && final_chunk.split_whitespace().count() > overlap / 2 {
                chunks.push(final_chunk);
            }
            break;
        }
    }

    chunks
}

/// Split text into chunks respecting paragraph boundaries
pub fn chunk_by_paragraphs(text: &str, max_chunk_size: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for para in paragraphs {
        let para_words = para.split_whitespace().count();

        if current_chunk.split_whitespace().count() + para_words > max_chunk_size
            && !current_chunk.is_empty()
        {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = String::new();
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para);
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_small_text() {
        let text = "Hello world this is a test";
        let chunks = chunk_text(text, 512, 128);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunk_overlapping() {
        // Create text with 100 words
        let words: Vec<String> = (0..100).map(|i| format!("word{}", i)).collect();
        let text = words.join(" ");

        let chunks = chunk_text(&text, 30, 10);

        // Should have multiple chunks
        assert!(chunks.len() > 1);

        // Each chunk should have at most 30 words
        for chunk in &chunks {
            assert!(chunk.split_whitespace().count() <= 30);
        }
    }

    #[test]
    fn test_chunk_empty() {
        let chunks = chunk_text("", 512, 128);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_by_paragraphs() {
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph.";
        let chunks = chunk_by_paragraphs(text, 100);
        assert_eq!(chunks.len(), 1); // All fit in one chunk

        let chunks = chunk_by_paragraphs(text, 5);
        assert!(chunks.len() >= 2); // Split across multiple
    }
}
