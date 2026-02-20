//! File content parser for DeepBrain
//!
//! Extracts text content from supported file types.

use std::path::Path;

/// Supported file extensions
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "md", "txt", "rs", "ts", "tsx", "js", "jsx", "py", "json", "toml", "yaml", "yml", "html",
    "css", "sh", "bash", "zsh", "fish", "swift", "go", "java", "c", "cpp", "h", "hpp", "rb",
    "lua", "sql", "xml", "csv", "log", "conf", "cfg", "ini", "env", "pdf", "emlx", "eml",
];

/// Check if a file extension is supported for indexing
pub fn is_supported(ext: &str) -> bool {
    SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Parse a file and extract its text content
pub fn parse_file(path: &Path) -> Result<String, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if !is_supported(&ext) {
        return Err(format!("Unsupported file type: {}", ext));
    }

    // PDF gets special binary handling
    if ext == "pdf" {
        return parse_pdf(path);
    }

    // Apple Mail .emlx format
    if ext == "emlx" {
        return parse_emlx(path);
    }

    // Standard .eml format (RFC 822)
    if ext == "eml" {
        return parse_eml(path);
    }

    // Read file content as text
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {:?}: {}", path, e))?;

    // Strip content based on file type
    match ext.as_str() {
        "json" => parse_json(&content),
        "html" | "xml" => parse_markup(&content),
        _ => Ok(clean_text(&content)),
    }
}

/// Parse a PDF file and extract text
fn parse_pdf(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Failed to read PDF {:?}: {}", path, e))?;

    let text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| format!("Failed to extract PDF text: {}", e))?;

    Ok(clean_text(&text))
}

/// Clean raw text content
fn clean_text(content: &str) -> String {
    // Remove excessive whitespace and empty lines
    content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse JSON and extract meaningful text values
fn parse_json(content: &str) -> Result<String, String> {
    // For JSON, we extract string values that likely contain meaningful text
    // Simple approach: just return the raw content cleaned up
    Ok(clean_text(content))
}

/// Parse HTML/XML and strip tags
fn parse_markup(content: &str) -> Result<String, String> {
    // Simple tag stripping
    let mut result = String::with_capacity(content.len());
    let mut in_tag = false;
    let mut in_script = false;

    for ch in content.chars() {
        match ch {
            '<' => {
                in_tag = true;
                // Check if we're entering a script or style block
                let rest = &content[content.find('<').unwrap_or(0)..];
                if rest.starts_with("<script") || rest.starts_with("<style") {
                    in_script = true;
                }
                if rest.starts_with("</script") || rest.starts_with("</style") {
                    in_script = false;
                }
            }
            '>' => {
                in_tag = false;
            }
            _ => {
                if !in_tag && !in_script {
                    result.push(ch);
                }
            }
        }
    }

    Ok(clean_text(&result))
}

/// Parse an Apple Mail .emlx file
///
/// Format: first line is byte count of the RFC 822 message,
/// followed by the raw email, then an Apple plist with metadata.
fn parse_emlx(path: &Path) -> Result<String, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read emlx {:?}: {}", path, e))?;

    // First line is the byte count — skip it
    let email_body = if let Some(pos) = raw.find('\n') {
        &raw[pos + 1..]
    } else {
        &raw
    };

    // The email ends before the Apple plist (starts with <?xml or <!DOCTYPE)
    let email_part = if let Some(plist_start) = email_body.find("<?xml") {
        &email_body[..plist_start]
    } else {
        email_body
    };

    parse_email_content(email_part)
}

/// Parse a standard .eml file (RFC 822 format)
fn parse_eml(path: &Path) -> Result<String, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read eml {:?}: {}", path, e))?;

    parse_email_content(&content)
}

/// Extract structured text from an RFC 822 email message
fn parse_email_content(raw: &str) -> Result<String, String> {
    let mut headers = Vec::new();
    let mut body_start = 0;

    // Parse headers (end at first blank line)
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            // Headers end at first blank line; body follows
            body_start = raw.lines().take(i + 1).map(|l| l.len() + 1).sum::<usize>();
            break;
        }

        let lower = line.to_lowercase();
        if lower.starts_with("from:") {
            headers.push(line.trim().to_string());
        } else if lower.starts_with("to:") {
            headers.push(line.trim().to_string());
        } else if lower.starts_with("subject:") {
            headers.push(line.trim().to_string());
        } else if lower.starts_with("date:") {
            headers.push(line.trim().to_string());
        }
    }

    let body_raw = if body_start > 0 && body_start < raw.len() {
        &raw[body_start..]
    } else {
        ""
    };

    // Extract plain text body: skip MIME boundaries, HTML, and base64
    let body_text = extract_plain_text_body(body_raw);

    let mut result = String::new();
    for header in &headers {
        result.push_str(header);
        result.push('\n');
    }
    if !result.is_empty() && !body_text.is_empty() {
        result.push('\n');
    }
    result.push_str(&body_text);

    if result.trim().is_empty() {
        return Err("Empty email content".to_string());
    }

    Ok(clean_text(&result))
}

/// Extract plain text from potentially MIME-encoded email body
fn extract_plain_text_body(body: &str) -> String {
    let mut result = String::new();
    let mut in_html = false;
    let mut in_base64 = false;
    let mut found_plain = false;

    for line in body.lines() {
        let trimmed = line.trim();

        // MIME boundary detection
        if trimmed.starts_with("--") && trimmed.len() > 4 {
            in_html = false;
            in_base64 = false;
            found_plain = false;
            continue;
        }

        // Content-Type headers within MIME parts
        let lower = trimmed.to_lowercase();
        if lower.starts_with("content-type:") {
            if lower.contains("text/plain") {
                found_plain = true;
                in_html = false;
            } else if lower.contains("text/html") {
                in_html = true;
                found_plain = false;
            }
            continue;
        }

        if lower.starts_with("content-transfer-encoding:") {
            if lower.contains("base64") {
                in_base64 = true;
            }
            continue;
        }

        // Skip Content-* continuation headers
        if lower.starts_with("content-") {
            continue;
        }

        // Skip base64-encoded blocks
        if in_base64 && trimmed.len() > 60 && !trimmed.contains(' ') {
            continue;
        }

        // Skip HTML content
        if in_html {
            continue;
        }

        // Collect plain text (either explicitly marked or default)
        if found_plain || (!in_html && !in_base64) {
            if !trimmed.is_empty() {
                result.push_str(trimmed);
                result.push('\n');
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_extensions() {
        assert!(is_supported("rs"));
        assert!(is_supported("ts"));
        assert!(is_supported("py"));
        assert!(is_supported("md"));
        assert!(is_supported("pdf"));
        assert!(!is_supported("exe"));
        assert!(!is_supported("png"));
    }

    #[test]
    fn test_clean_text() {
        let input = "  hello  \n\n\n  world  \n  ";
        let result = clean_text(input);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_supported_email_extensions() {
        assert!(is_supported("emlx"));
        assert!(is_supported("eml"));
    }

    #[test]
    fn test_parse_email_content() {
        let email = "From: alice@example.com\nTo: bob@example.com\nSubject: Hello\nDate: Mon, 1 Jan 2024 12:00:00 +0000\n\nThis is the body of the email.\nSecond line here.";
        let result = parse_email_content(email).unwrap();
        assert!(result.contains("From: alice@example.com"));
        assert!(result.contains("Subject: Hello"));
        assert!(result.contains("This is the body of the email."));
    }

    #[test]
    fn test_parse_markup() {
        let html = "<p>Hello <b>world</b></p>";
        let result = parse_markup(html).unwrap();
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
        assert!(!result.contains("<p>"));
    }
}
