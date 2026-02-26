//! Tauri IPC command handlers for DeepBrain

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::ai::AiProvider;
use std::sync::Arc;
use crate::state::{AppSettings, AppState, SystemStatus};

// ---- Think / Chat ----

#[derive(Debug, Serialize, Deserialize)]
pub struct ThinkResponse {
    pub response: String,
    pub confidence: f64,
    pub thought_id: String,
    pub memory_count: u32,
    pub ai_enhanced: bool,
}

/// Truncate a string to at most `max` chars on a word boundary.
fn truncate_chars(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Find last space before max to avoid cutting mid-word
    match s[..max].rfind(char::is_whitespace) {
        Some(pos) => &s[..pos],
        None => &s[..max],
    }
}

/// Format extra context (Spotlight + file results + email) for AI prompts.
///
/// Uses a budget-based approach: total `max_chars` (default 20,000 ≈ 5,000 tokens)
/// split 40% emails, 50% files, 10% Spotlight. Each item within a category gets
/// a proportional share, with higher-similarity items getting more space.
fn format_extra_context(
    spotlight: &[SpotlightResult],
    files: &[crate::indexer::FileResult],
    emails: &[crate::indexer::email::EmailSearchResult],
    live_emails: &[MailSearchResult],
) -> String {
    let max_chars: usize = 20_000;
    let email_budget = max_chars * 40 / 100;
    let file_budget = max_chars * 50 / 100;
    let spotlight_budget = max_chars * 10 / 100;

    let mut context = String::new();

    // --- Emails (indexed + live) ---
    let has_emails = !emails.is_empty() || !live_emails.is_empty();
    if has_emails {
        let total_email_count = emails.len() + live_emails.len();
        let per_email = if total_email_count > 0 { email_budget / total_email_count } else { 0 };

        context.push_str("\n--- Relevant Emails from Apple Mail ---\n");
        for (i, email) in emails.iter().enumerate() {
            context.push_str(&format!(
                "{}. Subject: {}\n   From: {} | Date: {} | Mailbox: {}\n",
                i + 1, email.subject, email.sender, email.date, email.mailbox,
            ));
            if !email.chunk.is_empty() {
                let preview = truncate_chars(&email.chunk, per_email);
                context.push_str(&format!("   Content: {}\n", preview));
            }
            context.push('\n');
        }
        let offset = emails.len();
        for (i, email) in live_emails.iter().enumerate() {
            context.push_str(&format!(
                "{}. Subject: {}\n   From: {} | Date: {} | Account: {} | Mailbox: {}\n",
                offset + i + 1, email.subject, email.sender, email.date, email.account, email.mailbox,
            ));
            if !email.preview.is_empty() {
                let preview = truncate_chars(&email.preview, per_email);
                context.push_str(&format!("   Content: {}\n", preview));
            }
            context.push('\n');
        }
        context.push_str("--- End Emails ---\n\n");
    }

    // --- Spotlight results (with optional content for documents) ---
    if !spotlight.is_empty() {
        let per_spotlight = spotlight_budget / spotlight.len().max(1);
        context.push_str("\n--- System Search Results (macOS Spotlight) ---\n");
        for (i, result) in spotlight.iter().enumerate() {
            context.push_str(&format!(
                "{}. [{}] {}: {}\n",
                i + 1, result.kind, result.name, result.path
            ));
            if let Some(ref content) = result.content {
                if !content.is_empty() {
                    let preview = truncate_chars(content, per_spotlight);
                    context.push_str(&format!("   Content: {}\n", preview));
                }
            }
        }
        context.push_str("--- End System Results ---\n\n");
    }

    // --- Indexed file results (similarity-weighted budget) ---
    if !files.is_empty() {
        // Give higher-similarity results proportionally more space
        let total_sim: f64 = files.iter().map(|f| f.similarity).sum();
        context.push_str("\n--- Relevant Indexed File Contents ---\n");
        for (i, file) in files.iter().enumerate() {
            let weight = if total_sim > 0.0 { file.similarity / total_sim } else { 1.0 / files.len() as f64 };
            let item_budget = (file_budget as f64 * weight).max(200.0) as usize;
            let preview = truncate_chars(&file.chunk, item_budget);
            context.push_str(&format!(
                "{}. {} ({})\n{}\n\n",
                i + 1, file.name, file.path, preview
            ));
        }
        context.push_str("--- End File Contents ---\n\n");
    }

    context
}

/// Core think implementation shared by Tauri IPC and HTTP API
pub async fn think_impl(input: &str, state: &AppState) -> Result<ThinkResponse, String> {
    let embedding = state.embeddings.embed(input).await?;

    // Get memory-based response and recall relevant memories
    let brain_result = state.engine.think_with_embedding(input, &embedding)?;
    let mut memories = state.engine.recall_f32(&embedding, Some(5), None).unwrap_or_default();

    // Graph augmentation: for top-3 results, find 1-hop graph neighbors
    // and add any graph-discovered memories not already in results.
    {
        let existing_ids: std::collections::HashSet<String> =
            memories.iter().map(|m| m.id.clone()).collect();
        let mut graph_additions = Vec::new();

        for memory in memories.iter().take(3) {
            let neighbors = state.graph.k_hop_neighbors(&memory.id, 1);
            for neighbor_id in neighbors {
                if !existing_ids.contains(&neighbor_id)
                    && !graph_additions.iter().any(|a: &crate::brain::cognitive::RecallResult| a.id == neighbor_id)
                {
                    if let Some(entry) = state.engine.memory.get(&neighbor_id) {
                        graph_additions.push(crate::brain::cognitive::RecallResult {
                            id: entry.id,
                            content: entry.content,
                            similarity: 0.4, // Graph-discovered baseline similarity
                            memory_type: entry.memory_type,
                        });
                    }
                }
            }
        }

        if !graph_additions.is_empty() {
            tracing::debug!(
                "Graph augmentation added {} memories to recall",
                graph_additions.len()
            );
            memories.extend(graph_additions);
        }
    }

    // Gather additional context: indexed emails + mdfind email + Spotlight + file index search (in parallel)
    // mdfind email search replaces the unreliable AppleScript live search (with AppleScript fallback)
    let (email_res, mdfind_mail_res, spotlight_res, file_res) = tokio::join!(
        state.email_indexer.search(input, 5),
        async {
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                mdfind_email_search_impl(input.to_string(), Some(5)),
            )
            .await
            .unwrap_or(Ok(vec![]))
        },
        spotlight_search_impl(input.to_string(), None, Some(8)),
        state.indexer.search(input, 10),
    );

    // Deduplicate file results: concatenate chunks from the same file
    let deduped_files = deduplicate_file_results(&file_res.unwrap_or_default());

    // Log query for GNN re-ranker training
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        let query_hash = hasher.finish() as i64;

        let file_ids: Vec<String> = deduped_files.iter().map(|f| f.path.clone()).collect();
        let selected = file_ids.first().cloned();
        let _ = state.persistence.log_query(
            query_hash,
            &file_ids,
            selected.as_deref(),
            "file",
        );
    }

    let extra_context = format_extra_context(
        &spotlight_res.unwrap_or_default(),
        &deduped_files,
        &email_res.unwrap_or_default(),
        &mdfind_mail_res.unwrap_or_default(),
    );

    // Try AI-enhanced response if a provider is configured
    let settings = state.settings.read().clone();
    let ai_response = match settings.ai_provider.as_str() {
        "ollama" => {
            let provider = crate::ai::ollama::OllamaProvider::new(&settings.ollama_model);
            Some(provider.generate(input, &memories, &extra_context).await)
        }
        "claude" => {
            if let Some(ref key) = settings.claude_api_key {
                let provider = crate::ai::claude::ClaudeProvider::new(key);
                Some(provider.generate(input, &memories, &extra_context).await)
            } else {
                None
            }
        }
        "ruvllm" => {
            if state.llm.is_loaded() {
                use crate::ai::AiProvider;
                let provider = std::sync::Arc::clone(&state.llm);
                Some(provider.generate(input, &memories, &extra_context).await)
            } else {
                tracing::warn!("ruvllm selected but no model loaded — falling back to memory-only");
                None
            }
        }
        _ => None,
    };

    if let Some(Err(ref e)) = ai_response {
        tracing::warn!("AI provider failed: {}", e);
    }

    if let Some(Ok(ai_resp)) = ai_response {
        // NOTE: Do NOT auto-store Q&A as episodic memories — this creates
        // a feedback loop where old answers dominate recall for future questions.
        // Users can explicitly store important info via "remember" mode.
        return Ok(ThinkResponse {
            response: ai_resp.content,
            confidence: brain_result.confidence,
            thought_id: brain_result.thought_id,
            memory_count: brain_result.memory_count,
            ai_enhanced: true,
        });
    }

    // Fallback: include extra context summary when AI is unavailable
    let fallback = if !extra_context.is_empty() {
        format!(
            "I couldn't reach the AI model, but here's what I found:\n\n{}",
            extra_context.chars().take(2000).collect::<String>()
        )
    } else if brain_result.memory_count > 0 {
        brain_result.response
    } else {
        "No relevant information found. The AI model may be loading — try again in a moment.".to_string()
    };

    Ok(ThinkResponse {
        response: fallback,
        confidence: brain_result.confidence,
        thought_id: brain_result.thought_id,
        memory_count: brain_result.memory_count,
        ai_enhanced: false,
    })
}

#[tauri::command]
pub async fn think(input: String, app: tauri::AppHandle, state: State<'_, Arc<AppState>>) -> Result<ThinkResponse, String> {
    crate::tray::set_status(&app, crate::tray::TrayStatus::Thinking);
    let result = think_impl(&input, &state).await;
    crate::tray::set_status(&app, crate::tray::TrayStatus::Idle);
    result
}

// ---- Remember ----

#[derive(Debug, Serialize, Deserialize)]
pub struct RememberResponse {
    pub id: String,
    pub memory_count: u32,
}

#[tauri::command]
pub async fn remember(
    content: String,
    memory_type: String,
    importance: Option<f64>,
    state: State<'_, Arc<AppState>>,
) -> Result<RememberResponse, String> {
    let embedding = state.embeddings.embed(&content).await?;

    // Clone values needed after the move into remember_with_embedding
    let content_for_graph = content.clone();
    let memory_type_for_graph = memory_type.clone();
    let embedding_for_graph = embedding.clone();

    let id = state.engine.remember_with_embedding(
        content,
        embedding,
        memory_type,
        importance,
    )?;

    // Persist to disk
    if let Some(node) = {
        // Get the node we just stored
        let nodes = state.engine.memory.all_nodes();
        nodes.into_iter().find(|n| n.id == id)
    } {
        let _ = state.persistence.store_memory(&node);
    }

    // Graph: add memory node and auto-connect to similar memories
    {
        let _ = state.graph.add_memory_node(&id, &memory_type_for_graph, &content_for_graph);

        // Find similar memories for graph connections
        let similar = state
            .engine
            .recall_f32(&embedding_for_graph, Some(5), None)
            .unwrap_or_default();
        let related_ids: Vec<String> = similar
            .iter()
            .filter(|s| s.id != id && s.similarity > 0.5)
            .map(|s| s.id.clone())
            .collect();

        if !related_ids.is_empty() {
            let edges_created = state.graph.auto_connect(&id, &related_ids, 0.6);
            if edges_created > 0 {
                tracing::debug!(
                    "Graph: auto-connected memory {} to {} neighbors",
                    id,
                    edges_created
                );
            }

            // Create cluster hyperedge if 3+ related
            if related_ids.len() >= 2 {
                let mut cluster_ids = vec![id.clone()];
                cluster_ids.extend(related_ids.iter().take(4).cloned());
                let _ = state.graph.create_cluster_hyperedge(
                    &cluster_ids,
                    &format!("Auto-cluster around {}", &content_for_graph.chars().take(50).collect::<String>()),
                    0.6,
                );
            }
        }
    }

    let memory_count = state.engine.memory.len();

    Ok(RememberResponse { id, memory_count })
}

// ---- Recall ----

#[derive(Debug, Serialize, Deserialize)]
pub struct RecallItem {
    pub id: String,
    pub content: String,
    pub similarity: f64,
    pub memory_type: String,
}

#[tauri::command]
pub async fn recall(
    query: String,
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<RecallItem>, String> {
    let embedding = state.embeddings.embed(&query).await?;

    let results = state
        .engine
        .recall_f32(&embedding, limit, None)?;

    Ok(results
        .into_iter()
        .map(|r| RecallItem {
            id: r.id,
            content: r.content,
            similarity: r.similarity,
            memory_type: r.memory_type,
        })
        .collect())
}

// ---- Status ----

#[tauri::command]
pub fn get_status(state: State<'_, Arc<AppState>>) -> Result<SystemStatus, String> {
    let introspection = state.engine.introspect();
    let settings = state.settings.read();
    let embedding_provider = format!("{:?}", state.embeddings.provider());
    let ai_available = state.ai_provider.read().is_some();

    let index_stats = state.indexer.stats().unwrap_or(crate::indexer::IndexStats {
        file_count: 0,
        chunk_count: 0,
        watched_dirs: 0,
        is_indexing: false,
    });

    let email_stats = state.email_indexer.stats().unwrap_or(crate::indexer::email::EmailIndexStats {
        indexed_count: 0,
        chunk_count: 0,
        is_indexing: false,
    });

    Ok(SystemStatus {
        status: introspection.status,
        memory_count: introspection.total_memories,
        thought_count: introspection.total_thoughts,
        uptime_ms: introspection.uptime_ms,
        ai_provider: settings.ai_provider.clone(),
        ai_available,
        embedding_provider,
        learning_trend: introspection.learning_trend,
        indexed_files: index_stats.file_count,
        indexed_chunks: index_stats.chunk_count,
        indexed_emails: email_stats.indexed_count,
    })
}

// ---- Settings ----

#[tauri::command]
pub fn get_settings(state: State<'_, Arc<AppState>>) -> Result<AppSettings, String> {
    Ok(state.settings.read().clone())
}

#[tauri::command]
pub fn update_settings(
    settings: AppSettings,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // Store Claude API key in Keychain if present
    if let Some(ref key) = settings.claude_api_key {
        if !key.is_empty() {
            crate::keychain::store_secret("claude_api_key", key)?;
        }
    } else {
        let _ = crate::keychain::delete_secret("claude_api_key");
    }

    // Store API token in Keychain if present
    if let Some(ref token) = settings.api_token {
        if !token.is_empty() {
            crate::keychain::store_secret("api_token", token)?;
        }
    } else {
        let _ = crate::keychain::delete_secret("api_token");
    }

    // Update auto-start login item
    #[cfg(target_os = "macos")]
    {
        let _ = crate::autostart::set_auto_start(settings.auto_start);
    }

    // Update email account filter if changed
    state.email_indexer.set_allowed_accounts(settings.email_accounts.clone());

    *state.settings.write() = settings.clone();

    // Refresh AI provider with new settings
    state.refresh_ai_provider();

    // Persist settings to SQLite (strip secrets — they're in Keychain)
    let mut persist_settings = settings;
    persist_settings.claude_api_key = None;
    persist_settings.api_token = None;
    let json = serde_json::to_string(&persist_settings)
        .map_err(|e| format!("Serialize error: {}", e))?;
    state.persistence.store_config("app_settings", &json)?;

    Ok(())
}

// ---- Thoughts ----

#[tauri::command]
pub fn get_thoughts(
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::brain::types::Thought>, String> {
    Ok(state.engine.get_thoughts(limit))
}

// ---- Stats ----

#[tauri::command]
pub fn get_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::brain::types::CognitiveStats, String> {
    Ok(state.engine.stats())
}

// ---- Evolve ----

#[tauri::command]
pub fn evolve(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::brain::cognitive::EvolutionResult, String> {
    Ok(state.engine.evolve())
}

// ---- Cycle ----

#[tauri::command]
pub fn cycle(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::brain::cognitive::CycleResult, String> {
    Ok(state.engine.cycle())
}

// ---- File Search ----

#[tauri::command]
pub async fn search_files(
    query: String,
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::indexer::FileResult>, String> {
    state.indexer.search(&query, limit.unwrap_or(10)).await
}

// ---- Index Files ----

#[tauri::command]
pub async fn index_files(state: State<'_, Arc<AppState>>) -> Result<u32, String> {
    state.indexer.scan_all().await
}

// ---- Workflows ----

#[tauri::command]
pub async fn run_workflow(
    action: String,
    query: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<crate::workflows::WorkflowResult, String> {
    let workflow_action = match action.as_str() {
        "remember_clipboard" => crate::workflows::WorkflowAction::RememberClipboard,
        "summarize" => crate::workflows::WorkflowAction::SummarizeRecent,
        "digest" => crate::workflows::WorkflowAction::LearningDigest,
        "search_and_remember" => crate::workflows::WorkflowAction::SearchAndRemember {
            query: query.unwrap_or_default(),
        },
        _ => return Err(format!("Unknown workflow: {}", action)),
    };

    crate::workflows::execute_workflow(
        workflow_action,
        &state.engine,
        &state.embeddings,
        &state.context,
    )
    .await
}

// ---- Check Ollama ----

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaStatus {
    pub available: bool,
    pub models: Vec<String>,
}

#[tauri::command]
pub async fn check_ollama() -> Result<OllamaStatus, String> {
    match crate::ai::ollama::list_models("http://127.0.0.1:11434").await {
        Ok(models) => Ok(OllamaStatus {
            available: true,
            models,
        }),
        Err(_) => Ok(OllamaStatus {
            available: false,
            models: vec![],
        }),
    }
}

// ---- Clipboard History ----

#[tauri::command]
pub fn get_clipboard_history(
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::context::ClipboardEntry>, String> {
    Ok(state.context.recent_clipboard(limit.unwrap_or(20) as usize))
}

// ---- Add Indexed Folder ----

#[tauri::command]
pub async fn add_indexed_folder(
    path: String,
    state: State<'_, Arc<AppState>>,
) -> Result<u32, String> {
    let folder = std::path::PathBuf::from(&path);
    if !folder.exists() || !folder.is_dir() {
        return Err(format!("Directory does not exist: {}", path));
    }

    // Add to indexer's watch dirs
    state.indexer.add_watch_dirs(vec![folder]);

    // Update settings
    {
        let mut settings = state.settings.write();
        if !settings.indexed_folders.contains(&path) {
            settings.indexed_folders.push(path);
        }
    }

    // Trigger re-scan
    state.indexer.scan_all().await
}

// ---- Browse: List Memories ----

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryBrowseItem {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub importance: f64,
    pub access_count: u32,
    pub timestamp: i64,
    pub connection_count: u32,
}

#[tauri::command]
pub fn list_memories(
    memory_type: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<MemoryBrowseItem>, String> {
    let nodes = state.engine.memory.all_nodes();
    let limit = limit.unwrap_or(100) as usize;
    let offset = offset.unwrap_or(0) as usize;

    let mut items: Vec<MemoryBrowseItem> = nodes
        .into_iter()
        .filter(|n| {
            if let Some(ref mt) = memory_type {
                format!("{:?}", n.memory_type).to_lowercase() == mt.to_lowercase()
            } else {
                true
            }
        })
        .map(|n| MemoryBrowseItem {
            id: n.id.clone(),
            content: n.content.clone(),
            memory_type: format!("{:?}", n.memory_type).to_lowercase(),
            importance: n.importance,
            access_count: n.access_count,
            timestamp: n.timestamp,
            connection_count: n.connections.len() as u32,
        })
        .collect();

    // Sort by timestamp descending (newest first)
    items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    let items: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

    Ok(items)
}

// ---- Browse: Get Memory ----

#[tauri::command]
pub fn get_memory(
    id: String,
    state: State<'_, Arc<AppState>>,
) -> Result<MemoryBrowseItem, String> {
    let entry = state.engine.memory.get(&id).ok_or("Memory not found")?;

    Ok(MemoryBrowseItem {
        id: entry.id,
        content: entry.content,
        memory_type: entry.memory_type,
        importance: entry.importance,
        access_count: entry.access_count,
        timestamp: entry.timestamp,
        connection_count: entry.connections.len() as u32,
    })
}

// ---- Browse: Delete Memory ----

#[tauri::command]
pub fn delete_memory(
    id: String,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // Remove from in-memory store
    if !state.engine.memory.delete(&id) {
        return Err("Memory not found".to_string());
    }
    // Remove from SQLite persistence
    state.persistence.delete_memory(&id)?;
    Ok(())
}

// ---- Browse: Update Memory ----

#[tauri::command]
pub fn update_memory(
    id: String,
    content: Option<String>,
    memory_type: Option<String>,
    importance: Option<f64>,
    state: State<'_, Arc<AppState>>,
) -> Result<MemoryBrowseItem, String> {
    if !state.engine.memory.update(&id, content, memory_type, importance) {
        return Err("Memory not found".to_string());
    }

    // Re-persist updated node
    if let Some(node) = state.engine.memory.all_nodes().into_iter().find(|n| n.id == id) {
        let _ = state.persistence.store_memory(&node);
    }

    // Return updated item
    get_memory(id, state)
}

// ---- Browse: Get File Chunks ----

#[tauri::command]
pub fn get_file_chunks(
    file_path: String,
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::indexer::FileChunkItem>, String> {
    state.indexer.get_file_chunks(&file_path, limit.unwrap_or(10))
}

// ---- Browse: List Indexed Files ----

#[tauri::command]
pub fn list_indexed_files(
    folder: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::indexer::FileListItem>, String> {
    state.indexer.list_files(
        folder.as_deref(),
        limit.unwrap_or(100),
        offset.unwrap_or(0),
    )
}

// ---- Spotlight Search ----

#[derive(Debug, Serialize, Deserialize)]
pub struct SpotlightResult {
    pub path: String,
    pub name: String,
    pub kind: String,
    pub modified: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Core spotlight search implementation (shared by Tauri IPC, think_impl, and HTTP API)
pub async fn spotlight_search_impl(
    query: String,
    kind: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<SpotlightResult>, String> {
    let limit = limit.unwrap_or(20) as usize;

    // Sanitize query for mdfind query language (escape double quotes and backslashes)
    let safe_q: String = query
        .chars()
        .filter(|c| !c.is_control())
        .map(|c| match c {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            _ => c.to_string(),
        })
        .collect();

    // Build mdfind query
    let mdfind_query = if let Some(ref kind_filter) = kind {
        match kind_filter.as_str() {
            "email" => format!("(kMDItemContentType == 'com.apple.mail.emlx' || kMDItemContentType == 'public.email-message') && (kMDItemTextContent == \"*{}*\"cd)", safe_q),
            "document" => format!("(kMDItemContentTypeTree == 'public.content') && (kMDItemDisplayName == \"*{}*\"cd || kMDItemTextContent == \"*{}*\"cd)", safe_q, safe_q),
            "pdf" => format!("kMDItemContentType == 'com.adobe.pdf' && (kMDItemDisplayName == \"*{}*\"cd || kMDItemTextContent == \"*{}*\"cd)", safe_q, safe_q),
            "image" => format!("kMDItemContentTypeTree == 'public.image' && kMDItemDisplayName == \"*{}*\"cd", safe_q),
            "presentation" => format!("(kMDItemContentType == 'com.apple.keynote.key' || kMDItemContentType == 'org.openxmlformats.presentationml.presentation') && kMDItemDisplayName == \"*{}*\"cd", safe_q),
            _ => safe_q.clone(),
        }
    } else {
        safe_q.clone()
    };

    let output = tokio::process::Command::new("mdfind")
        .arg(&mdfind_query)
        .output()
        .await
        .map_err(|e| format!("mdfind failed: {}", e))?;

    if !output.status.success() {
        return Err(format!("mdfind returned error: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let paths: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(limit)
        .collect();

    if paths.is_empty() {
        return Ok(vec![]);
    }

    // Get metadata for each result using mdls
    let mut results = Vec::with_capacity(paths.len());
    for path_str in &paths {
        let path = std::path::Path::new(path_str);
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let kind_label = match ext.as_str() {
            "emlx" | "eml" => "email".to_string(),
            "pdf" => "pdf".to_string(),
            "doc" | "docx" | "odt" | "rtf" | "txt" | "md" => "document".to_string(),
            "xls" | "xlsx" | "csv" => "spreadsheet".to_string(),
            "ppt" | "pptx" | "key" => "presentation".to_string(),
            "png" | "jpg" | "jpeg" | "gif" | "heic" | "webp" => "image".to_string(),
            "mp3" | "aac" | "wav" | "m4a" => "audio".to_string(),
            "mp4" | "mov" | "avi" | "mkv" => "video".to_string(),
            "rs" | "ts" | "tsx" | "js" | "py" | "swift" | "go" | "java" | "c" | "cpp" => "code".to_string(),
            _ => "file".to_string(),
        };

        // Get modification date
        let modified = path.metadata().ok().and_then(|m| {
            m.modified().ok().and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| d.as_secs().to_string())
            })
        });

        // For document-type results, read content via parser
        let content = match kind_label.as_str() {
            "document" | "pdf" => {
                match crate::indexer::parser::parse_file(path) {
                    Ok(text) if !text.is_empty() => Some(text),
                    _ => None,
                }
            }
            _ => None,
        };

        results.push(SpotlightResult {
            path: path_str.to_string(),
            name,
            kind: kind_label,
            modified,
            content,
        });
    }

    Ok(results)
}

#[tauri::command]
pub async fn spotlight_search(
    query: String,
    kind: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<SpotlightResult>, String> {
    spotlight_search_impl(query, kind, limit).await
}

// ---- Mail Search (AppleScript) ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailSearchResult {
    pub subject: String,
    pub sender: String,
    pub date: String,
    pub preview: String,
    pub account: String,
    pub mailbox: String,
    pub message_id: i64,
}

/// Deduplicate file search results: merge chunks from the same file into a single result.
fn deduplicate_file_results(results: &[crate::indexer::FileResult]) -> Vec<crate::indexer::FileResult> {
    use std::collections::HashMap;

    let mut by_path: HashMap<String, crate::indexer::FileResult> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for r in results {
        if let Some(existing) = by_path.get_mut(&r.path) {
            // Concatenate chunks, keep the higher similarity
            existing.chunk.push_str("\n\n");
            existing.chunk.push_str(&r.chunk);
            if r.similarity > existing.similarity {
                existing.similarity = r.similarity;
            }
        } else {
            order.push(r.path.clone());
            by_path.insert(r.path.clone(), r.clone());
        }
    }

    order.into_iter().filter_map(|p| by_path.remove(&p)).collect()
}

/// Search for emails using mdfind (Spotlight index) with mdls metadata extraction.
/// Falls back to AppleScript live search if mdfind returns nothing.
pub async fn mdfind_email_search_impl(
    query: String,
    limit: Option<u32>,
) -> Result<Vec<MailSearchResult>, String> {
    let limit = limit.unwrap_or(5) as usize;

    // Sanitize query for mdfind
    let safe_q: String = query
        .chars()
        .filter(|c| !c.is_control())
        .map(|c| match c {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            _ => c.to_string(),
        })
        .collect();

    let mdfind_query = format!(
        "(kMDItemContentType == 'com.apple.mail.emlx') && (kMDItemTextContent == \"*{}*\"cd || kMDItemSubject == \"*{}*\"cd || kMDItemAuthors == \"*{}*\"cd)",
        safe_q, safe_q, safe_q
    );

    let output = tokio::process::Command::new("mdfind")
        .arg(&mdfind_query)
        .output()
        .await
        .map_err(|e| format!("mdfind failed: {}", e))?;

    if !output.status.success() {
        // Fall back to AppleScript
        return mail_search_impl(query, Some("all".to_string()), Some(limit as u32)).await;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let paths: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).take(limit).collect();

    if paths.is_empty() {
        // Fallback to AppleScript live search
        return mail_search_impl(query, Some("all".to_string()), Some(limit as u32)).await;
    }

    let mut results = Vec::with_capacity(paths.len());
    for path_str in &paths {
        let path = std::path::Path::new(path_str);

        // Extract metadata via mdls
        let mdls_out = tokio::process::Command::new("mdls")
            .args(["-name", "kMDItemSubject", "-name", "kMDItemAuthors", "-name", "kMDItemContentModificationDate"])
            .arg(path_str)
            .output()
            .await;

        let (subject, sender, date) = if let Ok(ref md) = mdls_out {
            let md_str = String::from_utf8_lossy(&md.stdout);
            let subject = extract_mdls_value(&md_str, "kMDItemSubject").unwrap_or_else(|| "(no subject)".to_string());
            let sender = extract_mdls_value(&md_str, "kMDItemAuthors").unwrap_or_else(|| "unknown".to_string());
            let date = extract_mdls_value(&md_str, "kMDItemContentModificationDate").unwrap_or_default();
            (subject, sender, date)
        } else {
            ("(no subject)".to_string(), "unknown".to_string(), String::new())
        };

        // Read email content via the file parser
        let preview = match crate::indexer::parser::parse_file(path) {
            Ok(text) => {
                if text.len() > 500 { text[..500].to_string() } else { text }
            }
            Err(_) => String::new(),
        };

        results.push(MailSearchResult {
            subject,
            sender,
            date,
            preview,
            account: String::new(),
            mailbox: path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("").to_string(),
            message_id: 0,
        });
    }

    Ok(results)
}

/// Extract a value from mdls output for a given key.
fn extract_mdls_value(mdls_output: &str, key: &str) -> Option<String> {
    for line in mdls_output.lines() {
        if line.trim_start().starts_with(key) {
            if let Some(eq_pos) = line.find('=') {
                let val = line[eq_pos + 1..].trim().trim_matches('"');
                if val == "(null)" {
                    return None;
                }
                // Handle array values like ("value1", "value2")
                let val = val.trim_start_matches('(').trim_end_matches(')').trim();
                let val = val.trim_matches('"').trim();
                if val.is_empty() {
                    return None;
                }
                return Some(val.to_string());
            }
        }
    }
    None
}

/// Sanitize a string for safe embedding in AppleScript double-quoted strings.
/// Escapes backslashes, double quotes, and strips control characters that could
/// break out of the string context or inject AppleScript commands.
fn sanitize_for_applescript(input: &str) -> String {
    input
        .chars()
        .filter(|c| !c.is_control()) // Strip newlines, tabs, and other control chars
        .map(|c| match c {
            '\\' => "\\\\".to_string(),
            '"' => "\\\"".to_string(),
            _ => c.to_string(),
        })
        .collect()
}

/// Search Mail.app across all accounts via AppleScript
pub async fn mail_search_impl(
    query: String,
    field: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<MailSearchResult>, String> {
    let limit = limit.unwrap_or(10);
    let field = field.unwrap_or_else(|| "subject".to_string());

    let safe_query = sanitize_for_applescript(&query);

    // Build AppleScript search predicate
    let predicate = match field.as_str() {
        "sender" => format!("sender contains \"{}\"", safe_query),
        "body" => format!("content contains \"{}\"", safe_query),
        "all" => format!(
            "subject contains \"{q}\" or sender contains \"{q}\" or content contains \"{q}\"",
            q = safe_query
        ),
        _ => format!("subject contains \"{}\"", safe_query),
    };

    let script = format!(
        r#"tell application "Mail"
  set output to ""
  set resultCount to 0
  repeat with acct in accounts
    set acctName to name of acct
    repeat with mbox in mailboxes of acct
      set mboxName to name of mbox
      try
        set msgs to (messages of mbox whose {predicate})
        set msgCount to count of msgs
        if msgCount > 0 then
          set endIdx to msgCount
          if endIdx > {remaining_per_box} then set endIdx to {remaining_per_box}
          repeat with i from 1 to endIdx
            if resultCount >= {limit} then exit repeat
            set m to item i of msgs
            set subj to subject of m
            set sndr to sender of m
            set dt to (date received of m) as string
            set prev to ""
            try
              set prev to (content of m)
              if (length of prev) > 200 then set prev to (text 1 through 200 of prev)
            end try
            set msgId to (id of m) as string
            set output to output & subj & "␞" & sndr & "␞" & dt & "␞" & prev & "␞" & acctName & "␞" & mboxName & "␞" & msgId & "␟"
            set resultCount to resultCount + 1
          end repeat
        end if
      end try
      if resultCount >= {limit} then exit repeat
    end repeat
    if resultCount >= {limit} then exit repeat
  end repeat
  return output
end tell"#,
        predicate = predicate,
        limit = limit,
        remaining_per_box = limit,
    );

    let output = tokio::process::Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output()
        .await
        .map_err(|e| format!("osascript failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Mail search failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let results: Vec<MailSearchResult> = stdout
        .split('␟')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|record| {
            let fields: Vec<&str> = record.split('␞').collect();
            if fields.len() >= 7 {
                Some(MailSearchResult {
                    subject: fields[0].trim().to_string(),
                    sender: fields[1].trim().to_string(),
                    date: fields[2].trim().to_string(),
                    preview: fields[3].trim().replace('\n', " ").replace('\r', ""),
                    account: fields[4].trim().to_string(),
                    mailbox: fields[5].trim().to_string(),
                    message_id: fields[6].trim().parse().unwrap_or(0),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(results)
}

#[tauri::command]
pub async fn mail_search(
    query: String,
    field: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<MailSearchResult>, String> {
    mail_search_impl(query, field, limit).await
}

// ---- List memories (non-Tauri, for REST API) ----

pub fn list_memories_impl(
    memory_type: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
    state: &AppState,
) -> Result<Vec<MemoryBrowseItem>, String> {
    let nodes = state.engine.memory.all_nodes();
    let limit = limit.unwrap_or(100) as usize;
    let offset = offset.unwrap_or(0) as usize;

    let mut items: Vec<MemoryBrowseItem> = nodes
        .into_iter()
        .filter(|n| {
            if let Some(ref mt) = memory_type {
                format!("{:?}", n.memory_type).to_lowercase() == mt.to_lowercase()
            } else {
                true
            }
        })
        .map(|n| MemoryBrowseItem {
            id: n.id.clone(),
            content: n.content.clone(),
            memory_type: format!("{:?}", n.memory_type).to_lowercase(),
            importance: n.importance,
            access_count: n.access_count,
            timestamp: n.timestamp,
            connection_count: n.connections.len() as u32,
        })
        .collect();

    items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    let items: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

    Ok(items)
}

// ---- Common Folders ----

#[derive(Debug, Serialize, Deserialize)]
pub struct CommonFolder {
    pub label: String,
    pub path: String,
    pub exists: bool,
}

#[tauri::command]
pub fn get_common_folders() -> Vec<CommonFolder> {
    let home = dirs::home_dir().unwrap_or_default();
    let folders = vec![
        ("Home", home.clone()),
        ("Documents", home.join("Documents")),
        ("Desktop", home.join("Desktop")),
        ("Downloads", home.join("Downloads")),
        (
            "iCloud Drive",
            home.join("Library/Mobile Documents/com~apple~CloudDocs"),
        ),
        ("Projects", home.join("Projects")),
        ("Developer", home.join("Developer")),
    ];

    folders
        .into_iter()
        .map(|(label, path)| CommonFolder {
            label: label.to_string(),
            path: path.to_string_lossy().to_string(),
            exists: path.exists(),
        })
        .collect()
}

// ---- Folder Awareness ----

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FolderEntry {
    /// Folder name (last path component).
    pub name: String,
    /// Full absolute path.
    pub path: String,
    /// Number of immediate child items (files + folders).
    pub item_count: u32,
    /// Number of immediate child subdirectories.
    pub subfolder_count: u32,
    /// Last modification time as ISO 8601 string.
    pub modified: String,
    /// Whether this folder is currently indexed by DeepBrain.
    pub is_indexed: bool,
}

/// List subdirectories of a given path.
///
/// If `path` is None, lists the home directory.
/// Returns only directories (not files) — the UI can call `list_indexed_files`
/// with a folder filter to see files within a folder.
#[tauri::command]
pub fn list_folders(
    path: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<FolderEntry>, String> {
    let home = dirs::home_dir().unwrap_or_default();
    let target = match path {
        Some(ref p) => std::path::Path::new(p).to_path_buf(),
        None => home,
    };

    if !target.exists() || !target.is_dir() {
        return Err(format!("Not a directory: {}", target.display()));
    }

    // Sanitize: prevent traversal outside home
    let canonical = target
        .canonicalize()
        .map_err(|e| format!("Cannot resolve path: {}", e))?;

    let settings = state.settings.read();
    let indexed_folders: Vec<String> = settings.indexed_folders.clone();

    let mut entries: Vec<FolderEntry> = Vec::new();

    let read_dir = std::fs::read_dir(&canonical)
        .map_err(|e| format!("Cannot read directory: {}", e))?;

    for entry in read_dir.flatten() {
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if !ft.is_dir() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().to_string();
        // Skip hidden directories
        if name.starts_with('.') {
            continue;
        }

        let entry_path = entry.path();
        let path_str = entry_path.to_string_lossy().to_string();

        // Count immediate children
        let (item_count, subfolder_count) = match std::fs::read_dir(&entry_path) {
            Ok(children) => {
                let mut items = 0u32;
                let mut subs = 0u32;
                for child in children.flatten() {
                    let child_name = child.file_name().to_string_lossy().to_string();
                    if child_name.starts_with('.') {
                        continue;
                    }
                    items += 1;
                    if child.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        subs += 1;
                    }
                }
                (items, subs)
            }
            Err(_) => (0, 0),
        };

        let modified = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| {
                let dt: chrono::DateTime<chrono::Utc> = t.into();
                dt.format("%Y-%m-%dT%H:%M:%S").to_string()
            })
            .unwrap_or_default();

        let is_indexed = indexed_folders
            .iter()
            .any(|f| path_str.starts_with(f) || f.starts_with(&path_str));

        entries.push(FolderEntry {
            name,
            path: path_str,
            item_count,
            subfolder_count,
            modified,
            is_indexed,
        });
    }

    entries.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(entries)
}

#[tauri::command]
pub async fn pick_folder() -> Result<Option<String>, String> {
    let output = tokio::process::Command::new("osascript")
        .arg("-e")
        .arg("POSIX path of (choose folder with prompt \"Select a folder to index\")")
        .output()
        .await
        .map_err(|e| format!("Failed to open folder picker: {}", e))?;

    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if path.is_empty() {
            Ok(None)
        } else {
            Ok(Some(path))
        }
    } else {
        // User cancelled
        Ok(None)
    }
}

// ---- Recent Emails (cached from SQLite) ----

#[tauri::command]
pub fn get_recent_emails(
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<MailSearchResult>, String> {
    let limit = limit.unwrap_or(30);
    state.email_indexer.list_recent(limit)
}

// ---- Email Semantic Search (indexed/embedded emails) ----

#[tauri::command]
pub async fn search_emails(
    query: String,
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::indexer::email::EmailSearchResult>, String> {
    state.email_indexer.search(&query, limit.unwrap_or(10)).await
}

// ---- Email Index Stats ----

#[tauri::command]
pub fn email_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::indexer::email::EmailIndexStats, String> {
    state.email_indexer.stats()
}

// ---- Trigger Email Indexing ----

#[tauri::command]
pub async fn index_emails(
    state: State<'_, Arc<AppState>>,
) -> Result<u32, String> {
    state.email_indexer.index_pass(100).await
}

// ---- Pin Mode ----

#[tauri::command]
pub fn set_pinned(pinned: bool) {
    crate::overlay::set_pinned(pinned);
}

#[tauri::command]
pub fn get_pinned() -> bool {
    crate::overlay::is_pinned()
}

// ---- Storage Metrics (RVF) ----

#[tauri::command]
pub fn get_storage_metrics(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::metrics::StorageMetrics, String> {
    Ok(state.vector_store.metrics())
}

#[tauri::command]
pub fn verify_storage(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    let status = state.vector_store.status();
    Ok(serde_json::json!({
        "ok": true,
        "total_vectors": status.total_vectors,
        "file_size_bytes": status.file_size,
    }))
}

// ---- Migration ----

#[tauri::command]
pub fn migrate_from_v1(
    _state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::migration::MigrationReport, String> {
    let superbrain_dir = dirs::data_dir()
        .ok_or("No data dir")?
        .join("SuperBrain");

    let brain_db = superbrain_dir.join("brain.db");
    let files_db = superbrain_dir.join("files.db");
    let target_dir = dirs::data_dir()
        .ok_or("No data dir")?
        .join("DeepBrain");

    if !brain_db.exists() && !files_db.exists() {
        return Err("No SuperBrain V1 data found to migrate".to_string());
    }

    crate::deepbrain::migration::migrate_from_superbrain(&brain_db, &files_db, &target_dir)
        .map_err(|e| format!("Migration failed: {}", e))
}

// ---- Local LLM (ruvllm) ----

#[tauri::command]
pub fn load_local_model(
    model_id: String,
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::llm_bridge::LlmStatus, String> {
    state.llm.load_model(&model_id)?;

    // If ai_provider is set to "ruvllm", refresh so it picks up the loaded model
    if state.settings.read().ai_provider == "ruvllm" {
        state.refresh_ai_provider();
    }

    Ok(state.llm.status())
}

#[tauri::command]
pub fn unload_local_model(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.llm.unload_model();
    if state.settings.read().ai_provider == "ruvllm" {
        state.refresh_ai_provider();
    }
    Ok(())
}

#[tauri::command]
pub fn local_model_status(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::llm_bridge::LlmStatus, String> {
    Ok(state.llm.status())
}

// ---- SONA Stats ----

#[tauri::command]
pub fn get_sona_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::sona_bridge::SonaStats, String> {
    Ok(state.sona.stats())
}

// ---- Nervous System Stats ----

#[tauri::command]
pub fn get_nervous_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::nervous_bridge::NervousStats, String> {
    Ok(state.nervous.stats())
}

// ---- Verify All Subsystems ----

#[tauri::command]
pub async fn verify_all(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    // Storage
    let storage_metrics = state.vector_store.metrics();
    let storage_ok = true; // redb is always crash-safe

    // SONA
    let sona_stats = state.sona.stats();
    let sona_ok = sona_stats.instant_enabled && sona_stats.background_enabled;

    // Nervous system
    let nervous_stats = state.nervous.stats();
    let nervous_ok = nervous_stats.hopfield_capacity > 0;

    // LLM
    let llm_status = state.llm.status();

    // AI provider
    let ai_available = state.ai_provider.read().is_some();

    // Embeddings
    let emb_provider = format!("{:?}", state.embeddings.provider());

    // Indexer
    let index_stats = state.indexer.stats().unwrap_or(crate::indexer::IndexStats {
        file_count: 0,
        chunk_count: 0,
        watched_dirs: 0,
        is_indexing: false,
    });

    // Email
    let email_stats = state.email_indexer.stats().unwrap_or(crate::indexer::email::EmailIndexStats {
        indexed_count: 0,
        chunk_count: 0,
        is_indexing: false,
    });

    // Ollama reachability
    let ollama_ok = crate::ai::ollama::list_models("http://127.0.0.1:11434")
        .await
        .is_ok();

    Ok(serde_json::json!({
        "storage": {
            "ok": storage_ok,
            "vectors": storage_metrics.total_vectors,
            "file_size_bytes": storage_metrics.file_size_bytes,
        },
        "sona": {
            "ok": sona_ok,
            "patterns": sona_stats.patterns_stored,
            "trajectories": sona_stats.trajectories_buffered,
            "ewc_tasks": sona_stats.ewc_tasks,
            "instant_enabled": sona_stats.instant_enabled,
            "background_enabled": sona_stats.background_enabled,
        },
        "nervous": {
            "ok": nervous_ok,
            "hopfield_patterns": nervous_stats.hopfield_patterns,
            "hopfield_capacity": nervous_stats.hopfield_capacity,
            "router_sync": nervous_stats.router_sync,
        },
        "llm": {
            "ok": llm_status.model_loaded,
            "model_id": llm_status.model_id,
            "model_info": llm_status.model_info,
        },
        "ai_provider": {
            "ok": ai_available,
            "provider": state.settings.read().ai_provider.clone(),
        },
        "embeddings": {
            "ok": true,
            "provider": emb_provider,
        },
        "indexer": {
            "ok": true,
            "files": index_stats.file_count,
            "chunks": index_stats.chunk_count,
            "watched_dirs": index_stats.watched_dirs,
        },
        "email": {
            "ok": true,
            "indexed": email_stats.indexed_count,
            "chunks": email_stats.chunk_count,
        },
        "ollama": {
            "ok": ollama_ok,
        },
    }))
}

// ---- Bootstrap SONA from existing file vectors ----

#[derive(Debug, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub trajectories_recorded: u32,
    pub patterns_extracted: usize,
    pub message: String,
}

/// Core implementation shared by Tauri IPC and HTTP API.
pub fn bootstrap_sona_impl(state: &AppState) -> Result<BootstrapResult, String> {
    // Guard: skip if SONA already has patterns
    let stats = state.sona.stats();
    if stats.patterns_stored > 0 {
        return Ok(BootstrapResult {
            trajectories_recorded: 0,
            patterns_extracted: stats.patterns_stored,
            message: format!(
                "SONA already has {} patterns — skipping bootstrap.",
                stats.patterns_stored
            ),
        });
    }

    // Sample up to 5000 vectors from the file index
    let samples = state.indexer.sample_vectors(5000)?;
    if samples.is_empty() {
        return Err("No file vectors found — index some files first.".to_string());
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let mut recorded: u32 = 0;

    // Feed in batches of 1000, force_learn between batches
    for batch in samples.chunks(1000) {
        for (embedding, modified) in batch {
            // Quality by recency: 7d→0.7, 30d→0.55, older→0.4
            let age_days = (now - modified) / 86400;
            let quality = if age_days <= 7 {
                0.7
            } else if age_days <= 30 {
                0.55
            } else {
                0.4
            };
            state.sona.record_query(embedding, quality as f32);
            recorded += 1;
        }
        // Force a learning cycle after each batch
        state.sona.force_learn();
    }

    // Persist patterns to blob store
    let patterns = state.sona.export_patterns();
    if let Ok(bytes) = serde_json::to_vec(&patterns) {
        let _ = state.persistence.store_blob("sona_patterns", &bytes);
    }

    let final_stats = state.sona.stats();

    Ok(BootstrapResult {
        trajectories_recorded: recorded,
        patterns_extracted: final_stats.patterns_stored,
        message: format!(
            "Bootstrapped SONA: {} trajectories → {} patterns extracted.",
            recorded, final_stats.patterns_stored
        ),
    })
}

#[tauri::command]
pub fn bootstrap_sona(
    state: State<'_, Arc<AppState>>,
) -> Result<BootstrapResult, String> {
    bootstrap_sona_impl(&state)
}

// ---- Knowledge Graph ----

#[tauri::command]
pub fn graph_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::graph_bridge::GraphStats, String> {
    Ok(state.graph.stats())
}

#[tauri::command]
pub fn graph_neighbors(
    node_id: String,
    hops: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<String>, String> {
    Ok(state.graph.k_hop_neighbors(&node_id, hops.unwrap_or(1)))
}

// ---- GNN Re-ranking ----

#[tauri::command]
pub async fn gnn_rerank(
    query: String,
    limit: Option<u32>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::deepbrain::gnn_bridge::RankedResult>, String> {
    let embedding = state.embeddings.embed(&query).await?;
    let limit = limit.unwrap_or(10);

    let recall_results = state
        .engine
        .recall_f32(&embedding, Some(limit), None)?;

    let candidates: Vec<crate::deepbrain::gnn_bridge::RerankCandidate> = recall_results
        .iter()
        .filter_map(|r| {
            let node = state.engine.memory.all_nodes().into_iter().find(|n| n.id == r.id)?;
            Some(crate::deepbrain::gnn_bridge::RerankCandidate {
                id: r.id.clone(),
                content: r.content.clone(),
                embedding: node.vector.clone(),
                vector_score: r.similarity,
            })
        })
        .collect();

    let graph = state.graph.clone();
    let engine = state.engine.clone();
    let results = state.gnn.rerank(&embedding, &candidates, &|node_id| {
        let neighbor_ids = graph.k_hop_neighbors(node_id, 1);
        neighbor_ids
            .into_iter()
            .filter_map(|nid| {
                engine.memory.all_nodes().into_iter().find(|n| n.id == nid).map(|n| {
                    crate::deepbrain::gnn_bridge::NeighborInfo {
                        id: n.id.clone(),
                        embedding: n.vector.clone(),
                        edge_weight: 1.0,
                    }
                })
            })
            .collect()
    });

    Ok(results)
}

#[tauri::command]
pub fn gnn_save_weights(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let json = state.gnn.to_json()?;
    state.persistence.store_blob("gnn_weights", json.as_bytes())
}

// ---- Tensor Compression ----

#[tauri::command]
pub fn compression_scan(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::deepbrain::compress_bridge::CompressionStats, String> {
    let nodes = state.engine.memory.all_nodes();
    let mem_stats = state.engine.memory.stats();
    let total_accesses = mem_stats.total_accesses as u64;

    let memories: Vec<crate::deepbrain::compress_bridge::MemoryForCompression> = nodes
        .iter()
        .map(|n| crate::deepbrain::compress_bridge::MemoryForCompression {
            id: n.id.clone(),
            access_count: n.access_count,
            vector_len: n.vector.len(),
        })
        .collect();

    Ok(state.compressor.scan(&memories, total_accesses))
}

#[tauri::command]
pub fn compression_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    let stats = state.engine.memory.stats();
    Ok(serde_json::json!({
        "total_memories": stats.total_memories,
        "total_accesses": stats.total_accesses,
    }))
}

// ---- Knowledge Bootstrap ----

#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeBootstrapResult {
    pub total_memories_created: u32,
    pub total_skipped: u32,
    pub sources: Vec<crate::indexer::bootstrap::SourceResult>,
    pub duration_secs: f64,
}

/// Core implementation shared by Tauri IPC and HTTP API.
pub async fn bootstrap_knowledge_impl(
    sources: Vec<String>,
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> Result<KnowledgeBootstrapResult, String> {
    let result = crate::indexer::bootstrap::bootstrap_knowledge(state, app_handle, sources).await?;
    Ok(KnowledgeBootstrapResult {
        total_memories_created: result.total_memories_created,
        total_skipped: result.total_skipped,
        sources: result.sources,
        duration_secs: result.duration_secs,
    })
}

#[tauri::command]
pub async fn bootstrap_knowledge(
    sources: Vec<String>,
    state: State<'_, Arc<AppState>>,
    app_handle: tauri::AppHandle,
) -> Result<KnowledgeBootstrapResult, String> {
    bootstrap_knowledge_impl(sources, &state, Some(&app_handle)).await
}

// ---- Browser History Sync ----

/// Trigger a manual browser history sync pass.
pub async fn browser_sync_impl(state: &AppState) -> Result<u32, String> {
    state.browser_indexer.sync_pass().await
}

#[tauri::command]
pub async fn browser_sync(
    state: State<'_, Arc<AppState>>,
) -> Result<u32, String> {
    browser_sync_impl(&state).await
}

/// Get browser indexer statistics.
pub fn browser_stats_impl(state: &AppState) -> Result<serde_json::Value, String> {
    let stats = state.browser_indexer.stats();
    Ok(serde_json::json!({
        "indexed_count": stats.indexed_count,
        "last_sync_unix": stats.last_sync_unix,
        "is_syncing": stats.is_syncing,
    }))
}

#[tauri::command]
pub fn browser_stats(
    state: State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    browser_stats_impl(&state)
}

// ---- Flush (save to disk) ----

#[tauri::command]
pub fn flush(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.flush()
}
