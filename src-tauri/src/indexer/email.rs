//! Background email indexer for DeepBrain
//!
//! Periodically fetches read emails from Apple Mail via AppleScript,
//! filters out newsletters and unimportant messages, then chunks and
//! embeds them into the same SQLite index used for file search.
//!
//! Design decisions:
//! - Small batches with timeouts to avoid AppleScript hangs
//! - Deduplication by Mail.app message ID
//! - Emails stored with synthetic path `mail://<message_id>` so the
//!   existing FileIndexer search works seamlessly

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::brain::embeddings::EmbeddingModel;
use crate::deepbrain::vector_store::{DeepBrainVectorStore, VectorFilter, VectorMetadata};
use super::chunker;

/// An email fetched from Apple Mail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedEmail {
    pub message_id: i64,
    pub subject: String,
    pub sender: String,
    pub date: String,
    pub body: String,
    pub account: String,
    pub mailbox: String,
}

/// Statistics about the email index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailIndexStats {
    pub indexed_count: u32,
    pub chunk_count: u32,
    pub is_indexing: bool,
}

/// Mailbox names to skip (EN + DE)
const SKIP_MAILBOXES: &[&str] = &[
    // English
    "Junk", "Trash", "Sent", "Sent Messages", "Drafts",
    "Spam", "Deleted Messages", "Deleted Items", "Archive",
    // German
    "Papierkorb", "Gesendet", "Gesendete Nachrichten", "Entwürfe",
    "Werbung", "Gelöschte Nachrichten", "Archiv",
];

/// Strong sender patterns (score 3) — near-certain newsletter/automated mail
const SENDER_STRONG: &[&str] = &[
    "noreply@", "no-reply@", "newsletter@", "marketing@",
    "bulk@", "campaign@", "mailer-daemon@", "postmaster@",
    "bounce@", "donotreply@", "do-not-reply@", "promo@",
    "announce@", "news@",
];

/// Medium sender patterns (score 2)
const SENDER_MEDIUM: &[&str] = &[
    "notifications@", "notification@", "updates@", "digest@",
    "automated@", "alerts@", "deals@",
];

/// Weak sender patterns (score 1)
const SENDER_WEAK: &[&str] = &[
    "info@", "hello@", "team@", "support@", "store@",
];

/// Subject patterns that indicate newsletters (score 2 each)
const NEWSLETTER_SUBJECT_PATTERNS: &[&str] = &[
    "unsubscribe", "newsletter", "weekly digest", "daily digest",
    "monthly update", "promotional", "weekly roundup", "your weekly",
    "your monthly", "your daily", "price drop", "flash sale",
    "top stories", "release notes",
];

pub struct EmailIndexer {
    db_path: PathBuf,
    embeddings: Arc<EmbeddingModel>,
    is_indexing: RwLock<bool>,
    /// RVF-backed vector store (None = brute-force SQLite fallback).
    vector_store: Option<Arc<DeepBrainVectorStore>>,
    /// If non-empty, only index these email accounts (matched by account name).
    allowed_accounts: RwLock<Vec<String>>,
}

impl EmailIndexer {
    pub fn new(db_path: PathBuf, embeddings: Arc<EmbeddingModel>) -> Result<Self, String> {
        let indexer = Self {
            db_path,
            embeddings,
            is_indexing: RwLock::new(false),
            vector_store: None,
            allowed_accounts: RwLock::new(Vec::new()),
        };
        indexer.initialize_db()?;
        Ok(indexer)
    }

    /// Create a new email indexer backed by a DeepBrain RVF vector store.
    pub fn with_vector_store(
        db_path: PathBuf,
        embeddings: Arc<EmbeddingModel>,
        store: Arc<DeepBrainVectorStore>,
    ) -> Result<Self, String> {
        let indexer = Self {
            db_path,
            embeddings,
            is_indexing: RwLock::new(false),
            vector_store: Some(store),
            allowed_accounts: RwLock::new(Vec::new()),
        };
        indexer.initialize_db()?;
        Ok(indexer)
    }

    /// Set allowed email accounts. If non-empty, only these accounts are scanned.
    pub fn set_allowed_accounts(&self, accounts: Vec<String>) {
        *self.allowed_accounts.write() = accounts;
    }

    fn open_connection(&self) -> Result<Connection, String> {
        Connection::open(&self.db_path).map_err(|e| format!("Email DB open failed: {}", e))
    }

    fn initialize_db(&self) -> Result<(), String> {
        let conn = self.open_connection()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS email_index (
                message_id INTEGER PRIMARY KEY,
                subject TEXT NOT NULL,
                sender TEXT NOT NULL,
                date_received TEXT NOT NULL,
                account TEXT NOT NULL,
                mailbox TEXT NOT NULL,
                indexed_at INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0
            );
            ",
        )
        .map_err(|e| format!("Email DB init failed: {}", e))?;
        Ok(())
    }

    /// Check if a message has already been indexed
    fn is_indexed(&self, message_id: i64) -> bool {
        let conn = match self.open_connection() {
            Ok(c) => c,
            Err(_) => return false,
        };
        conn.query_row(
            "SELECT 1 FROM email_index WHERE message_id = ?1",
            params![message_id],
            |_| Ok(()),
        )
        .is_ok()
    }

    /// Fetch read emails from Apple Mail in a single small batch.
    /// Uses a per-account, per-mailbox approach with strict limits
    /// and a timeout to prevent AppleScript hangs.
    async fn fetch_read_emails(&self, batch_size: u32) -> Result<Vec<IndexedEmail>, String> {
        // Build account filter for AppleScript
        let allowed = self.allowed_accounts.read().clone();
        let account_filter = if allowed.is_empty() {
            // No filter — scan all accounts
            String::new()
        } else {
            // Build AppleScript list: {"account1", "account2"}
            let items: Vec<String> = allowed.iter().map(|a| format!("\"{}\"", a)).collect();
            format!("set allowedAccounts to {{{}}}", items.join(", "))
        };
        let account_check = if allowed.is_empty() {
            String::new()
        } else {
            // Skip accounts not in the allowed list
            "if allowedAccounts does not contain acctName then\n      -- skip this account\n    else".to_string()
        };
        let account_check_end = if allowed.is_empty() {
            String::new()
        } else {
            "end if".to_string()
        };

        // AppleScript: iterate accounts → mailboxes, get read messages
        // We fetch full body (up to 2000 chars) for indexing
        let script = format!(
            r#"tell application "Mail"
  set output to ""
  set resultCount to 0
  set maxResults to {batch_size}
  {account_filter}
  repeat with acct in accounts
    set acctName to name of acct
    {account_check}
    repeat with mbox in mailboxes of acct
      set mboxName to name of mbox
      try
        set msgs to (messages of mbox whose read status is true and date received > (current date) - 30 * days)
        set msgCount to count of msgs
        if msgCount > 0 then
          set endIdx to msgCount
          if endIdx > 20 then set endIdx to 20
          repeat with i from 1 to endIdx
            if resultCount >= maxResults then exit repeat
            set m to item i of msgs
            set subj to subject of m
            set sndr to sender of m
            set dt to (date received of m) as string
            set msgId to (id of m) as string
            set bodyText to ""
            try
              set bodyText to (content of m)
              if (length of bodyText) > 2000 then set bodyText to (text 1 through 2000 of bodyText)
            end try
            set output to output & msgId & "␞" & subj & "␞" & sndr & "␞" & dt & "␞" & bodyText & "␞" & acctName & "␞" & mboxName & "␟"
            set resultCount to resultCount + 1
          end repeat
        end if
      end try
      if resultCount >= maxResults then exit repeat
    end repeat
    if resultCount >= maxResults then exit repeat
    {account_check_end}
  end repeat
  return output
end tell"#,
            batch_size = batch_size,
            account_filter = account_filter,
            account_check = account_check,
            account_check_end = account_check_end,
        );

        // Run with a timeout to prevent hangs
        let result = tokio::time::timeout(
            Duration::from_secs(60),
            tokio::process::Command::new("osascript")
                .arg("-e")
                .arg(&script)
                .output(),
        )
        .await;

        let output = match result {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Err(format!("osascript failed: {}", e)),
            Err(_) => return Err("Email fetch timed out after 60s".to_string()),
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Mail fetch failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let emails: Vec<IndexedEmail> = stdout
            .split('␟')
            .filter(|s| !s.trim().is_empty())
            .filter_map(|record| {
                let fields: Vec<&str> = record.split('␞').collect();
                if fields.len() >= 7 {
                    Some(IndexedEmail {
                        message_id: fields[0].trim().parse().unwrap_or(0),
                        subject: fields[1].trim().to_string(),
                        sender: fields[2].trim().to_string(),
                        date: fields[3].trim().to_string(),
                        body: fields[4].trim().replace('\r', ""),
                        account: fields[5].trim().to_string(),
                        mailbox: fields[6].trim().to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(emails)
    }

    /// Score how likely an email is a newsletter or automated message.
    /// Returns a weighted score; emails scoring >= 3 should be skipped.
    fn newsletter_score(email: &IndexedEmail) -> u32 {
        let sender_lower = email.sender.to_lowercase();
        let subject_lower = email.subject.to_lowercase();
        let body_lower = email.body.to_lowercase();
        let mut score: u32 = 0;

        // --- Sender signals ---
        for pattern in SENDER_STRONG {
            if sender_lower.contains(pattern) {
                score += 3;
                break; // one match is enough
            }
        }
        if score == 0 {
            for pattern in SENDER_MEDIUM {
                if sender_lower.contains(pattern) {
                    score += 2;
                    break;
                }
            }
        }
        if score == 0 {
            for pattern in SENDER_WEAK {
                if sender_lower.contains(pattern) {
                    score += 1;
                    break;
                }
            }
        }

        // --- Subject signals (score 2) ---
        for pattern in NEWSLETTER_SUBJECT_PATTERNS {
            if subject_lower.contains(pattern) {
                score += 2;
                break;
            }
        }

        // --- Body signals ---
        if body_lower.contains("unsubscribe") || body_lower.contains("abmelden") {
            score += 2;
        }
        if body_lower.contains("view in browser") || body_lower.contains("im browser anzeigen") {
            score += 2;
        }
        if body_lower.contains("manage preferences") || body_lower.contains("opt out")
            || body_lower.contains("email preferences")
        {
            score += 1;
        }

        score
    }

    /// Check if a mailbox should be skipped
    fn should_skip_mailbox(mailbox: &str) -> bool {
        let mailbox_lower = mailbox.to_lowercase();
        SKIP_MAILBOXES
            .iter()
            .any(|skip| mailbox_lower == skip.to_lowercase())
    }

    /// Run a single indexing pass: fetch, filter, embed, store.
    /// Returns the number of newly indexed emails.
    pub async fn index_pass(&self, batch_size: u32) -> Result<u32, String> {
        {
            let is_indexing = self.is_indexing.read();
            if *is_indexing {
                return Ok(0);
            }
        }
        *self.is_indexing.write() = true;

        let result = self.index_pass_inner(batch_size).await;

        *self.is_indexing.write() = false;
        result
    }

    async fn index_pass_inner(&self, batch_size: u32) -> Result<u32, String> {
        tracing::info!("Email indexing pass starting (batch_size={})", batch_size);

        let emails = match self.fetch_read_emails(batch_size).await {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Email fetch failed: {}", e);
                return Err(e);
            }
        };

        tracing::info!("Fetched {} emails from Mail.app", emails.len());

        let mut indexed = 0u32;

        for email in &emails {
            // Skip if already indexed
            if self.is_indexed(email.message_id) {
                continue;
            }

            // Skip unwanted mailboxes
            if Self::should_skip_mailbox(&email.mailbox) {
                continue;
            }

            // Skip newsletters (score >= 3 = almost certainly a newsletter)
            let nl_score = Self::newsletter_score(email);
            if nl_score >= 3 {
                tracing::debug!("Skipping newsletter (score={}): {} from {}", nl_score, email.subject, email.sender);
                continue;
            }

            // Skip empty bodies
            if email.body.trim().is_empty() {
                continue;
            }

            // Build indexable text: combine subject + sender + body
            let full_text = format!(
                "Email from: {}\nSubject: {}\nDate: {}\n\n{}",
                email.sender, email.subject, email.date, email.body
            );

            // Chunk the email content
            let chunks = chunker::chunk_text(&full_text, 512, 128);
            if chunks.is_empty() {
                continue;
            }

            // Embed each chunk and store
            let synthetic_path = format!("mail://{}", email.message_id);
            let display_name = format!("{} ({})", email.subject, email.sender);

            let mut chunk_vectors = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                match self.embeddings.embed(chunk).await {
                    Ok(vec) => chunk_vectors.push((chunk.clone(), vec)),
                    Err(e) => {
                        tracing::debug!("Embedding failed for email {}: {}", email.message_id, e);
                        continue;
                    }
                }
            }

            if chunk_vectors.is_empty() {
                continue;
            }

            // Store in database
            if let Err(e) = self.store_email(email, &synthetic_path, &display_name, &chunk_vectors) {
                tracing::warn!("Failed to store email {}: {}", email.message_id, e);
                continue;
            }

            indexed += 1;
            tracing::debug!("Indexed email: {} ({} chunks)", email.subject, chunk_vectors.len());
        }

        if indexed > 0 {
            tracing::info!("Email indexing pass complete: {} new emails indexed", indexed);
        }

        Ok(indexed)
    }

    /// Store an email and its chunks in the database
    fn store_email(
        &self,
        email: &IndexedEmail,
        synthetic_path: &str,
        display_name: &str,
        chunks: &[(String, Vec<f32>)],
    ) -> Result<(), String> {
        let conn = self.open_connection()?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        // Record in email_index
        conn.execute(
            "INSERT OR REPLACE INTO email_index (message_id, subject, sender, date_received, account, mailbox, indexed_at, chunk_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                email.message_id,
                email.subject,
                email.sender,
                email.date,
                email.account,
                email.mailbox,
                now,
                chunks.len() as u32,
            ],
        )
        .map_err(|e| format!("Store email index failed: {}", e))?;

        // Store in the shared file_index + file_chunks tables so
        // the existing FileIndexer.search() finds emails too
        conn.execute(
            "INSERT OR REPLACE INTO file_index (path, name, ext, modified, chunk_count)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![synthetic_path, display_name, "email", now, chunks.len() as u32],
        )
        .map_err(|e| format!("Store file_index for email failed: {}", e))?;

        // Remove old chunks
        conn.execute(
            "DELETE FROM file_chunks WHERE file_path = ?1",
            params![synthetic_path],
        )
        .map_err(|e| format!("Delete old email chunks failed: {}", e))?;

        // Insert new chunks (SQLite + RVF vector store)
        for (i, (content, vector)) in chunks.iter().enumerate() {
            let vector_bytes = vector_to_bytes(vector);
            conn.execute(
                "INSERT INTO file_chunks (file_path, chunk_index, content, vector)
                 VALUES (?1, ?2, ?3, ?4)",
                params![synthetic_path, i as u32, content, vector_bytes],
            )
            .map_err(|e| format!("Store email chunk failed: {}", e))?;

            // Also insert into RVF vector store for indexed search.
            if let Some(ref store) = self.vector_store {
                let chunk_id = format!("email::{}::{}", synthetic_path, i);
                let meta = VectorMetadata {
                    source: "email".to_string(),
                    path: synthetic_path.to_string(),
                    timestamp: now,
                    ..Default::default()
                };
                if let Err(e) = store.store_vector(&chunk_id, vector, meta) {
                    tracing::warn!("Failed to store email chunk in RVF: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Search indexed emails by semantic similarity.
    /// Uses RVF vector store for O(log n) search when available, falls back to brute-force SQL.
    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<EmailSearchResult>, String> {
        let query_vector = self.embeddings.embed(query).await?;

        // Try RVF vector store first (fast O(log n) — major perf upgrade from brute-force).
        if let Some(ref store) = self.vector_store {
            return self.search_rvf(store, &query_vector, limit);
        }

        // Fallback: brute-force SQL scan (original implementation).
        self.search_brute_force(&query_vector, limit)
    }

    /// RVF-accelerated email search.
    fn search_rvf(
        &self,
        store: &DeepBrainVectorStore,
        query_vector: &[f32],
        limit: u32,
    ) -> Result<Vec<EmailSearchResult>, String> {
        let fetch_k = (limit as usize) * 3;
        let filter = Some(VectorFilter::Source("email".to_string()));

        let rvf_results = store
            .search(query_vector, fetch_k, filter)
            .map_err(|e| format!("RVF email search failed: {}", e))?;

        let conn = self.open_connection()?;

        let mut results: Vec<EmailSearchResult> = Vec::with_capacity(rvf_results.len());
        for vr in &rvf_results {
            if vr.similarity < 0.15 {
                continue;
            }

            // Parse the RVF ID: "email::mail://<message_id>::<chunk_index>"
            let id_str = &vr.id;
            let stripped = id_str.strip_prefix("email::").unwrap_or(id_str);
            let parts: Vec<&str> = stripped.rsplitn(2, "::").collect();
            if parts.len() != 2 {
                continue;
            }
            let synthetic_path = parts[1]; // "mail://<message_id>"

            // Look up email metadata from SQLite.
            let row_result = conn.query_row(
                "SELECT ei.subject, ei.sender, ei.date_received, ei.mailbox, fc.content
                 FROM email_index ei
                 JOIN file_index fi ON fi.path = 'mail://' || CAST(ei.message_id AS TEXT)
                 JOIN file_chunks fc ON fc.file_path = fi.path
                 WHERE fi.path = ?1
                 LIMIT 1",
                params![synthetic_path],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            );

            if let Ok((subject, sender, date, mailbox, chunk)) = row_result {
                results.push(EmailSearchResult {
                    subject,
                    sender,
                    date,
                    mailbox,
                    chunk,
                    similarity: vr.similarity,
                });
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit as usize);

        Ok(results)
    }

    /// Brute-force email search (fallback when vector store not available).
    fn search_brute_force(
        &self,
        query_vector: &[f32],
        limit: u32,
    ) -> Result<Vec<EmailSearchResult>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare(
                "SELECT fc.content, fc.vector, fi.name, fi.path, ei.subject, ei.sender, ei.date_received, ei.mailbox
                 FROM file_chunks fc
                 JOIN file_index fi ON fc.file_path = fi.path
                 JOIN email_index ei ON fi.path = 'mail://' || CAST(ei.message_id AS TEXT)
                 WHERE fi.ext = 'email'",
            )
            .map_err(|e| format!("Email search query failed: {}", e))?;

        let mut results: Vec<EmailSearchResult> = stmt
            .query_map([], |row| {
                let content: String = row.get(0)?;
                let vector_bytes: Vec<u8> = row.get(1)?;
                let _name: String = row.get(2)?;
                let _path: String = row.get(3)?;
                let subject: String = row.get(4)?;
                let sender: String = row.get(5)?;
                let date: String = row.get(6)?;
                let mailbox: String = row.get(7)?;

                let vector = bytes_to_vector(&vector_bytes);
                let similarity =
                    crate::brain::utils::cosine_similarity(query_vector, &vector) as f64;

                Ok(EmailSearchResult {
                    subject,
                    sender,
                    date,
                    mailbox,
                    chunk: content,
                    similarity,
                })
            })
            .map_err(|e| format!("Email search failed: {}", e))?
            .filter_map(|r| r.ok())
            .filter(|r| r.similarity > 0.15)
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit as usize);

        Ok(results)
    }

    /// Get index statistics
    pub fn stats(&self) -> Result<EmailIndexStats, String> {
        let conn = self.open_connection()?;

        let indexed_count: u32 = conn
            .query_row("SELECT COUNT(*) FROM email_index", [], |row| row.get(0))
            .unwrap_or(0);

        let chunk_count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM file_chunks fc JOIN file_index fi ON fc.file_path = fi.path WHERE fi.ext = 'email'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let is_indexing = *self.is_indexing.read();

        Ok(EmailIndexStats {
            indexed_count,
            chunk_count,
            is_indexing,
        })
    }
}

/// Email search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSearchResult {
    pub subject: String,
    pub sender: String,
    pub date: String,
    pub mailbox: String,
    pub chunk: String,
    pub similarity: f64,
}

fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for &val in vector {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
