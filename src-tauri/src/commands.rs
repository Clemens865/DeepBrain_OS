//! Tauri IPC command handlers for SuperBrain

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::ai::AiProvider;
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

#[tauri::command]
pub async fn think(input: String, app: tauri::AppHandle, state: State<'_, AppState>) -> Result<ThinkResponse, String> {
    crate::tray::set_status(&app, crate::tray::TrayStatus::Thinking);
    let embedding = state.embeddings.embed(&input).await?;

    // Get memory-based response and recall relevant memories
    let brain_result = state.engine.think_with_embedding(&input, &embedding)?;
    let memories = state.engine.recall_f32(&embedding, Some(5), None).unwrap_or_default();

    // Try AI-enhanced response if a provider is configured
    let ai_provider_name = state.ai_provider.read().as_ref().map(|p| p.name().to_string());
    if let Some(_provider_name) = ai_provider_name {
        // Clone what we need, then drop the lock before awaiting
        let ai_result = {
            let provider_guard = state.ai_provider.read();
            if let Some(ref provider) = *provider_guard {
                // We need to drop the guard before awaiting, so check availability first
                let provider_ref: &dyn crate::ai::AiProvider = provider.as_ref();
                // Unfortunately we can't hold the guard across await, so we build
                // a quick non-async check here and do the generate outside
                Some(provider_ref.name().to_string())
            } else {
                None
            }
        };

        if ai_result.is_some() {
            // Re-acquire and generate (the provider is behind RwLock, can't hold across await)
            // Instead, extract what we need to call the provider
            let settings = state.settings.read().clone();
            let ai_response = match settings.ai_provider.as_str() {
                "ollama" => {
                    let provider = crate::ai::ollama::OllamaProvider::new(&settings.ollama_model);
                    provider.generate(&input, &memories).await
                }
                "claude" => {
                    if let Some(ref key) = settings.claude_api_key {
                        let provider = crate::ai::claude::ClaudeProvider::new(key);
                        provider.generate(&input, &memories).await
                    } else {
                        Err("No Claude API key".to_string())
                    }
                }
                _ => Err("No AI provider".to_string()),
            };

            if let Ok(ai_resp) = ai_response {
                // Store the AI interaction as an episodic memory
                let _ = state.engine.remember_with_embedding(
                    format!("Q: {} A: {}", input, &ai_resp.content[..ai_resp.content.len().min(200)]),
                    embedding,
                    "episodic".to_string(),
                    Some(0.5),
                );

                crate::tray::set_status(&app, crate::tray::TrayStatus::Idle);
                return Ok(ThinkResponse {
                    response: ai_resp.content,
                    confidence: brain_result.confidence,
                    thought_id: brain_result.thought_id,
                    memory_count: brain_result.memory_count,
                    ai_enhanced: true,
                });
            }
        }
    }

    // Fallback: memory-only response
    crate::tray::set_status(&app, crate::tray::TrayStatus::Idle);
    Ok(ThinkResponse {
        response: brain_result.response,
        confidence: brain_result.confidence,
        thought_id: brain_result.thought_id,
        memory_count: brain_result.memory_count,
        ai_enhanced: false,
    })
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
    state: State<'_, AppState>,
) -> Result<RememberResponse, String> {
    let embedding = state.embeddings.embed(&content).await?;

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
    state: State<'_, AppState>,
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
pub fn get_status(state: State<'_, AppState>) -> Result<SystemStatus, String> {
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
    })
}

// ---- Settings ----

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Result<AppSettings, String> {
    Ok(state.settings.read().clone())
}

#[tauri::command]
pub fn update_settings(
    settings: AppSettings,
    state: State<'_, AppState>,
) -> Result<(), String> {
    // Store Claude API key in Keychain if present
    if let Some(ref key) = settings.claude_api_key {
        if !key.is_empty() {
            crate::keychain::store_secret("claude_api_key", key)?;
        }
    } else {
        let _ = crate::keychain::delete_secret("claude_api_key");
    }

    // Update auto-start login item
    #[cfg(target_os = "macos")]
    {
        let _ = crate::autostart::set_auto_start(settings.auto_start);
    }

    *state.settings.write() = settings.clone();

    // Refresh AI provider with new settings
    state.refresh_ai_provider();

    // Persist settings to SQLite (strip API key — it's in Keychain)
    let mut persist_settings = settings;
    persist_settings.claude_api_key = None;
    let json = serde_json::to_string(&persist_settings)
        .map_err(|e| format!("Serialize error: {}", e))?;
    state.persistence.store_config("app_settings", &json)?;

    Ok(())
}

// ---- Thoughts ----

#[tauri::command]
pub fn get_thoughts(
    limit: Option<u32>,
    state: State<'_, AppState>,
) -> Result<Vec<crate::brain::types::Thought>, String> {
    Ok(state.engine.get_thoughts(limit))
}

// ---- Stats ----

#[tauri::command]
pub fn get_stats(
    state: State<'_, AppState>,
) -> Result<crate::brain::types::CognitiveStats, String> {
    Ok(state.engine.stats())
}

// ---- Evolve ----

#[tauri::command]
pub fn evolve(
    state: State<'_, AppState>,
) -> Result<crate::brain::cognitive::EvolutionResult, String> {
    Ok(state.engine.evolve())
}

// ---- Cycle ----

#[tauri::command]
pub fn cycle(
    state: State<'_, AppState>,
) -> Result<crate::brain::cognitive::CycleResult, String> {
    Ok(state.engine.cycle())
}

// ---- File Search ----

#[tauri::command]
pub async fn search_files(
    query: String,
    limit: Option<u32>,
    state: State<'_, AppState>,
) -> Result<Vec<crate::indexer::FileResult>, String> {
    state.indexer.search(&query, limit.unwrap_or(10)).await
}

// ---- Index Files ----

#[tauri::command]
pub async fn index_files(state: State<'_, AppState>) -> Result<u32, String> {
    state.indexer.scan_all().await
}

// ---- Workflows ----

#[tauri::command]
pub async fn run_workflow(
    action: String,
    query: Option<String>,
    state: State<'_, AppState>,
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
    match crate::ai::ollama::list_models("http://localhost:11434").await {
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

// ---- Flush (save to disk) ----

#[tauri::command]
pub fn flush(state: State<'_, AppState>) -> Result<(), String> {
    state.flush()
}
