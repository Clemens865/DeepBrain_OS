//! Local HTTP REST API for DeepBrain
//!
//! Exposes cognitive engine endpoints on 127.0.0.1:19519 so external
//! tools (CLI, Notebook_LLM, shell scripts) can interact with DeepBrain.

use std::sync::Arc;

use axum::{
    extract::{Path, State as AxumState},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};

use crate::commands::{self, RecallItem, RememberResponse, ThinkResponse};
use crate::state::{AppSettings, AppState, SystemStatus};

// ---- Request / Response types ----

#[derive(Deserialize)]
struct ThinkRequest {
    input: String,
}

#[derive(Deserialize)]
struct RememberRequest {
    content: String,
    memory_type: Option<String>,
    importance: Option<f64>,
}

#[derive(Deserialize)]
struct RecallRequest {
    query: String,
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct FileSearchRequest {
    query: String,
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct WorkflowRequest {
    action: String,
    query: Option<String>,
}

#[derive(Deserialize)]
struct EmailSearchRequest {
    query: String,
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct MailLiveSearchRequest {
    query: String,
    field: Option<String>,
    limit: Option<u32>,
}

// ---- New request types for browser UI ----

#[derive(Deserialize)]
struct ListMemoriesRequest {
    memory_type: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
}

#[derive(Deserialize)]
struct UpdateMemoryRequest {
    content: Option<String>,
    memory_type: Option<String>,
    importance: Option<f64>,
}

#[derive(Deserialize)]
struct SpotlightRequest {
    query: String,
    kind: Option<String>,
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct LoadModelRequest {
    model_id: String,
}

#[derive(Deserialize)]
struct AddFolderRequest {
    path: String,
}

#[derive(Deserialize)]
struct ListFilesRequest {
    folder: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
}

#[derive(Deserialize)]
struct FileChunksRequest {
    file_path: String,
    limit: Option<u32>,
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ---- Auth middleware ----

/// Constant-time string comparison to prevent timing attacks
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

async fn auth_middleware(
    AxumState(state): AxumState<Arc<AppState>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> impl IntoResponse {
    // Load token: prefer Keychain, fall back to settings
    let expected = crate::keychain::get_secret("api_token")
        .ok()
        .flatten()
        .or_else(|| state.settings.read().api_token.clone())
        .unwrap_or_default();

    if expected.is_empty() {
        // No token configured — auto-generate one and store in Keychain
        let generated = uuid::Uuid::new_v4().to_string();
        let _ = crate::keychain::store_secret("api_token", &generated);
        tracing::info!(
            "No API token found. Generated and stored in Keychain. Token: {}",
            generated
        );
        // Allow this first request through so the user can discover the token
        // via `security find-generic-password -s DeepBrain -a api_token -w`
        return Ok(next.run(request).await);
    }

    let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(t) if constant_time_eq(t, &expected) => Ok(next.run(request).await),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid or missing Bearer token. Use: Authorization: Bearer <token>".to_string(),
            }),
        )),
    }
}

// ---- Handlers ----

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { ok: true })
}

async fn api_think(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<ThinkRequest>,
) -> Result<Json<ThinkResponse>, (StatusCode, Json<ErrorResponse>)> {
    commands::think_impl(&body.input, &state)
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

async fn api_remember(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, (StatusCode, Json<ErrorResponse>)> {
    let memory_type = body.memory_type.unwrap_or_else(|| "semantic".to_string());
    let embedding = state
        .embeddings
        .embed(&body.content)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })?;

    let id = state
        .engine
        .remember_with_embedding(body.content, embedding, memory_type, body.importance)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })?;

    // Persist to disk
    if let Some(node) = {
        let nodes = state.engine.memory.all_nodes();
        nodes.into_iter().find(|n| n.id == id)
    } {
        let _ = state.persistence.store_memory(&node);
    }

    let memory_count = state.engine.memory.len();
    Ok(Json(RememberResponse { id, memory_count }))
}

async fn api_recall(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<RecallRequest>,
) -> Result<Json<Vec<RecallItem>>, (StatusCode, Json<ErrorResponse>)> {
    let embedding = state.embeddings.embed(&body.query).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e }),
        )
    })?;

    let results = state
        .engine
        .recall_f32(&embedding, body.limit, None)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })?;

    Ok(Json(
        results
            .into_iter()
            .map(|r| RecallItem {
                id: r.id,
                content: r.content,
                similarity: r.similarity,
                memory_type: r.memory_type,
            })
            .collect(),
    ))
}

async fn api_status(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Result<Json<SystemStatus>, (StatusCode, Json<ErrorResponse>)> {
    let introspection = state.engine.introspect();
    let settings = state.settings.read();
    let embedding_provider = format!("{:?}", state.embeddings.provider());
    let ai_available = state.ai_provider.read().is_some();

    let index_stats = state
        .indexer
        .stats()
        .unwrap_or(crate::indexer::IndexStats {
            file_count: 0,
            chunk_count: 0,
            watched_dirs: 0,
            is_indexing: false,
        });

    let email_stats = state
        .email_indexer
        .stats()
        .unwrap_or(crate::indexer::email::EmailIndexStats {
            indexed_count: 0,
            chunk_count: 0,
            is_indexing: false,
        });

    Ok(Json(SystemStatus {
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
    }))
}

async fn api_search_files(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<FileSearchRequest>,
) -> Result<Json<Vec<crate::indexer::FileResult>>, (StatusCode, Json<ErrorResponse>)> {
    state
        .indexer
        .search(&body.query, body.limit.unwrap_or(10))
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

async fn api_workflow(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<WorkflowRequest>,
) -> Result<Json<crate::workflows::WorkflowResult>, (StatusCode, Json<ErrorResponse>)> {
    let workflow_action = match body.action.as_str() {
        "remember_clipboard" => crate::workflows::WorkflowAction::RememberClipboard,
        "summarize" => crate::workflows::WorkflowAction::SummarizeRecent,
        "digest" => crate::workflows::WorkflowAction::LearningDigest,
        "search_and_remember" => crate::workflows::WorkflowAction::SearchAndRemember {
            query: body.query.unwrap_or_default(),
        },
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Unknown workflow: {}", other),
                }),
            ));
        }
    };

    crate::workflows::execute_workflow(workflow_action, &state.engine, &state.embeddings, &state.context)
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

async fn api_clipboard(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<Vec<crate::context::ClipboardEntry>> {
    Json(state.context.recent_clipboard(20))
}

async fn api_settings(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<AppSettings> {
    let mut settings = state.settings.read().clone();
    // Strip secrets from the response
    settings.claude_api_key = None;
    settings.api_token = None;
    Json(settings)
}

// ---- Email: semantic search of indexed/embedded emails ----

async fn api_search_emails(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<EmailSearchRequest>,
) -> Result<Json<Vec<crate::indexer::email::EmailSearchResult>>, (StatusCode, Json<ErrorResponse>)> {
    state
        .email_indexer
        .search(&body.query, body.limit.unwrap_or(10))
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

// ---- Email: index stats ----

async fn api_email_stats(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Result<Json<crate::indexer::email::EmailIndexStats>, (StatusCode, Json<ErrorResponse>)> {
    state
        .email_indexer
        .stats()
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

// ---- Email: trigger manual indexing ----

async fn api_index_emails(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state
        .email_indexer
        .index_pass(100)
        .await
        .map(|count| Json(serde_json::json!({ "indexed": count })))
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

// ---- Storage: health metrics (RVF) ----

async fn api_storage_health(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::deepbrain::metrics::StorageMetrics> {
    Json(state.vector_store.metrics())
}

async fn api_storage_verify(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let status = state.vector_store.status();
    let metrics = state.vector_store.metrics();
    Json(serde_json::json!({
        "ok": true,
        "total_vectors": status.total_vectors,
        "file_size_bytes": status.file_size,
        "current_epoch": status.current_epoch,
        "dead_space_ratio": status.dead_space_ratio,
        "id_map_count": metrics.id_map_count,
    }))
}

// ---- Migration: SQLite → RvfStore ----

async fn api_migrate(
    AxumState(_state): AxumState<Arc<AppState>>,
) -> Result<Json<crate::deepbrain::migration::MigrationReport>, (StatusCode, Json<ErrorResponse>)> {
    let superbrain_dir = dirs::data_dir()
        .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "No data dir".to_string() })))?
        .join("SuperBrain");

    let brain_db = superbrain_dir.join("brain.db");
    let files_db = superbrain_dir.join("files.db");
    let target_dir = dirs::data_dir()
        .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "No data dir".to_string() })))?
        .join("DeepBrain");

    crate::deepbrain::migration::migrate_from_superbrain(&brain_db, &files_db, &target_dir)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{}", e) })))
}

// ---- Mail: live AppleScript search ----

async fn api_mail_search(
    Json(body): Json<MailLiveSearchRequest>,
) -> Result<Json<Vec<commands::MailSearchResult>>, (StatusCode, Json<ErrorResponse>)> {
    commands::mail_search_impl(body.query, body.field, body.limit)
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })
}

// ---- Memory CRUD ----

async fn api_list_memories(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<ListMemoriesRequest>,
) -> Result<Json<Vec<commands::MemoryBrowseItem>>, (StatusCode, Json<ErrorResponse>)> {
    commands::list_memories_impl(body.memory_type, body.limit, body.offset, &state)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

async fn api_get_memory(
    AxumState(state): AxumState<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<commands::MemoryBrowseItem>, (StatusCode, Json<ErrorResponse>)> {
    let entry = state.engine.memory.get(&id).ok_or_else(|| {
        (StatusCode::NOT_FOUND, Json(ErrorResponse { error: "Memory not found".to_string() }))
    })?;
    Ok(Json(commands::MemoryBrowseItem {
        id: entry.id,
        content: entry.content,
        memory_type: entry.memory_type,
        importance: entry.importance,
        access_count: entry.access_count,
        timestamp: entry.timestamp,
        connection_count: entry.connections.len() as u32,
    }))
}

async fn api_delete_memory(
    AxumState(state): AxumState<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    if !state.engine.memory.delete(&id) {
        return Err((StatusCode::NOT_FOUND, Json(ErrorResponse { error: "Memory not found".to_string() })));
    }
    let _ = state.persistence.delete_memory(&id);
    Ok(StatusCode::NO_CONTENT)
}

async fn api_update_memory(
    AxumState(state): AxumState<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<UpdateMemoryRequest>,
) -> Result<Json<commands::MemoryBrowseItem>, (StatusCode, Json<ErrorResponse>)> {
    if !state.engine.memory.update(&id, body.content, body.memory_type, body.importance) {
        return Err((StatusCode::NOT_FOUND, Json(ErrorResponse { error: "Memory not found".to_string() })));
    }
    if let Some(node) = state.engine.memory.all_nodes().into_iter().find(|n| n.id == id) {
        let _ = state.persistence.store_memory(&node);
    }
    let entry = state.engine.memory.get(&id).ok_or_else(|| {
        (StatusCode::NOT_FOUND, Json(ErrorResponse { error: "Memory not found".to_string() }))
    })?;
    Ok(Json(commands::MemoryBrowseItem {
        id: entry.id,
        content: entry.content,
        memory_type: entry.memory_type,
        importance: entry.importance,
        access_count: entry.access_count,
        timestamp: entry.timestamp,
        connection_count: entry.connections.len() as u32,
    }))
}

// ---- Thoughts & Stats ----

async fn api_get_thoughts(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<Vec<crate::brain::types::Thought>> {
    Json(state.engine.get_thoughts(Some(200)))
}

async fn api_get_stats(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::brain::types::CognitiveStats> {
    Json(state.engine.stats())
}

// ---- Settings (write) ----

async fn api_update_settings(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(settings): Json<AppSettings>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Store secrets in Keychain
    if let Some(ref key) = settings.claude_api_key {
        if !key.is_empty() {
            let _ = crate::keychain::store_secret("claude_api_key", key);
        }
    }
    if let Some(ref token) = settings.api_token {
        if !token.is_empty() {
            let _ = crate::keychain::store_secret("api_token", token);
        }
    }
    *state.settings.write() = settings.clone();
    state.refresh_ai_provider();
    // Persist (strip secrets)
    let mut persist = settings;
    persist.claude_api_key = None;
    persist.api_token = None;
    let json = serde_json::to_string(&persist)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{}", e) })))?;
    state.persistence.store_config("app_settings", &json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

// ---- Brain actions ----

async fn api_evolve(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::brain::cognitive::EvolutionResult> {
    Json(state.engine.evolve())
}

async fn api_cycle(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::brain::cognitive::CycleResult> {
    Json(state.engine.cycle())
}

async fn api_flush(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state.flush()
        .map(|_| Json(serde_json::json!({ "ok": true })))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

// ---- LLM management ----

async fn api_load_model(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<LoadModelRequest>,
) -> Result<Json<crate::deepbrain::llm_bridge::LlmStatus>, (StatusCode, Json<ErrorResponse>)> {
    state.llm.load_model(&body.model_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))?;
    if state.settings.read().ai_provider == "ruvllm" {
        state.refresh_ai_provider();
    }
    Ok(Json(state.llm.status()))
}

async fn api_unload_model(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<serde_json::Value> {
    state.llm.unload_model();
    if state.settings.read().ai_provider == "ruvllm" {
        state.refresh_ai_provider();
    }
    Json(serde_json::json!({ "ok": true }))
}

async fn api_llm_status(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::deepbrain::llm_bridge::LlmStatus> {
    Json(state.llm.status())
}

// ---- DeepBrain subsystem stats ----

async fn api_sona_stats(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::deepbrain::sona_bridge::SonaStats> {
    Json(state.sona.stats())
}

async fn api_nervous_stats(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<crate::deepbrain::nervous_bridge::NervousStats> {
    Json(state.nervous.stats())
}

async fn api_verify_all(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let storage_metrics = state.vector_store.metrics();
    let sona_stats = state.sona.stats();
    let nervous_stats = state.nervous.stats();
    let llm_status = state.llm.status();
    let ai_available = state.ai_provider.read().is_some();
    let emb_provider = format!("{:?}", state.embeddings.provider());
    let index_stats = state.indexer.stats().unwrap_or(crate::indexer::IndexStats {
        file_count: 0, chunk_count: 0, watched_dirs: 0, is_indexing: false,
    });
    let email_stats = state.email_indexer.stats().unwrap_or(crate::indexer::email::EmailIndexStats {
        indexed_count: 0, chunk_count: 0, is_indexing: false,
    });
    let ollama_ok = crate::ai::ollama::list_models("http://127.0.0.1:11434").await.is_ok();

    Json(serde_json::json!({
        "storage": { "ok": storage_metrics.dead_space_ratio < 0.5, "vectors": storage_metrics.total_vectors },
        "sona": { "ok": sona_stats.instant_enabled && sona_stats.background_enabled, "patterns": sona_stats.patterns_stored },
        "nervous": { "ok": nervous_stats.hopfield_capacity > 0, "hopfield_patterns": nervous_stats.hopfield_patterns },
        "llm": { "ok": llm_status.model_loaded, "model_id": llm_status.model_id },
        "ai_provider": { "ok": ai_available, "provider": state.settings.read().ai_provider.clone() },
        "embeddings": { "ok": true, "provider": emb_provider },
        "indexer": { "ok": true, "files": index_stats.file_count, "chunks": index_stats.chunk_count },
        "email": { "ok": true, "indexed": email_stats.indexed_count },
        "ollama": { "ok": ollama_ok },
    }))
}

// ---- SONA bootstrap ----

async fn api_bootstrap_sona(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Result<Json<commands::BootstrapResult>, (StatusCode, Json<ErrorResponse>)> {
    commands::bootstrap_sona_impl(&state)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

// ---- Spotlight search ----

async fn api_spotlight(
    Json(body): Json<SpotlightRequest>,
) -> Result<Json<Vec<commands::SpotlightResult>>, (StatusCode, Json<ErrorResponse>)> {
    commands::spotlight_search_impl(body.query, body.kind, body.limit)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

// ---- File indexing ----

async fn api_list_files(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<ListFilesRequest>,
) -> Result<Json<Vec<crate::indexer::FileListItem>>, (StatusCode, Json<ErrorResponse>)> {
    state.indexer.list_files(body.folder.as_deref(), body.limit.unwrap_or(200), body.offset.unwrap_or(0))
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

async fn api_file_chunks(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<FileChunksRequest>,
) -> Result<Json<Vec<crate::indexer::FileChunkItem>>, (StatusCode, Json<ErrorResponse>)> {
    state.indexer.get_file_chunks(&body.file_path, body.limit.unwrap_or(10))
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

async fn api_index_files(
    AxumState(state): AxumState<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state.indexer.scan_all().await
        .map(|count| Json(serde_json::json!({ "indexed": count })))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

async fn api_add_folder(
    AxumState(state): AxumState<Arc<AppState>>,
    Json(body): Json<AddFolderRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let folder = std::path::PathBuf::from(&body.path);
    if !folder.exists() || !folder.is_dir() {
        return Err((StatusCode::BAD_REQUEST, Json(ErrorResponse {
            error: format!("Directory does not exist: {}", body.path),
        })));
    }
    state.indexer.add_watch_dirs(vec![folder]);
    {
        let mut settings = state.settings.write();
        if !settings.indexed_folders.contains(&body.path) {
            settings.indexed_folders.push(body.path);
        }
    }
    let count = state.indexer.scan_all().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))?;
    Ok(Json(serde_json::json!({ "indexed": count })))
}

// ---- Ollama status ----

async fn api_check_ollama() -> Json<commands::OllamaStatus> {
    match crate::ai::ollama::list_models("http://127.0.0.1:11434").await {
        Ok(models) => Json(commands::OllamaStatus { available: true, models }),
        Err(_) => Json(commands::OllamaStatus { available: false, models: vec![] }),
    }
}

// ---- Common folders ----

async fn api_common_folders() -> Json<Vec<commands::CommonFolder>> {
    Json(commands::get_common_folders())
}

// ---- Recent emails ----

async fn api_recent_emails() -> Result<Json<Vec<commands::MailSearchResult>>, (StatusCode, Json<ErrorResponse>)> {
    // Use mdfind-based search for recent emails (no AppleScript requirement)
    commands::mdfind_email_search_impl("".to_string(), Some(30))
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })))
}

// ---- Server startup ----

/// Start the HTTP API server. Call from main.rs via tokio::spawn.
pub async fn start_api_server(state: Arc<AppState>, port: u16) {
    // Routes that skip auth
    let public = Router::new().route("/api/health", get(health));

    // Routes that require auth (if token is set)
    let protected = Router::new()
        // Core cognition
        .route("/api/think", post(api_think))
        .route("/api/remember", post(api_remember))
        .route("/api/recall", post(api_recall))
        // Status & settings
        .route("/api/status", get(api_status))
        .route("/api/settings", get(api_settings).post(api_update_settings))
        // Search
        .route("/api/search/files", post(api_search_files))
        .route("/api/search/emails", post(api_search_emails))
        .route("/api/spotlight/search", post(api_spotlight))
        .route("/api/mail/search", post(api_mail_search))
        // Memory CRUD
        .route("/api/memories", post(api_list_memories))
        .route("/api/memories/:id", get(api_get_memory).delete(api_delete_memory).patch(api_update_memory))
        // Files & indexing
        .route("/api/files/list", post(api_list_files))
        .route("/api/files/chunks", post(api_file_chunks))
        .route("/api/indexer/scan", post(api_index_files))
        .route("/api/indexer/folder", post(api_add_folder))
        // Thoughts & stats
        .route("/api/thoughts", get(api_get_thoughts))
        .route("/api/stats", get(api_get_stats))
        // Brain actions
        .route("/api/brain/evolve", post(api_evolve))
        .route("/api/brain/cycle", post(api_cycle))
        .route("/api/brain/flush", post(api_flush))
        // Email
        .route("/api/email/stats", get(api_email_stats))
        .route("/api/email/index", post(api_index_emails))
        .route("/api/email/recent", get(api_recent_emails))
        // Clipboard
        .route("/api/clipboard", get(api_clipboard))
        // Workflow
        .route("/api/workflow", post(api_workflow))
        // Storage
        .route("/api/storage/health", get(api_storage_health))
        .route("/api/storage/verify", get(api_storage_verify))
        .route("/api/storage/migrate", post(api_migrate))
        // DeepBrain subsystems
        .route("/api/sona/stats", get(api_sona_stats))
        .route("/api/sona/bootstrap", post(api_bootstrap_sona))
        .route("/api/nervous/stats", get(api_nervous_stats))
        .route("/api/verify", post(api_verify_all))
        // LLM
        .route("/api/llm/load", post(api_load_model))
        .route("/api/llm/unload", post(api_unload_model))
        .route("/api/llm/status", get(api_llm_status))
        // Ollama
        .route("/api/ollama/status", get(api_check_ollama))
        // Folders
        .route("/api/folders/common", get(api_common_folders))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // CORS layer for browser access
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Static file serving: serve the React dist/ directory with SPA fallback
    let dist_dir = std::env::var("DEEPBRAIN_DIST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            // Search in multiple locations (dev mode vs production)
            let candidates: Vec<std::path::PathBuf> = vec![
                // Next to executable (production)
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.join("dist")))
                    .unwrap_or_default(),
                // CWD/dist
                std::path::PathBuf::from("dist"),
                // Project root (dev mode: CWD is src-tauri/, dist is ../dist)
                std::path::PathBuf::from("../dist"),
                // Cargo manifest dir (dev mode)
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../dist"),
            ];
            candidates
                .into_iter()
                .find(|p| p.join("index.html").exists())
                .unwrap_or_else(|| std::path::PathBuf::from("dist"))
        });

    let app = if dist_dir.exists() {
        tracing::info!("Serving web UI from {}", dist_dir.display());
        let spa_fallback = ServeFile::new(dist_dir.join("index.html"));
        let serve_dir = ServeDir::new(&dist_dir).not_found_service(spa_fallback);
        public
            .merge(protected)
            .with_state(state)
            .layer(cors)
            .fallback_service(serve_dir)
    } else {
        tracing::info!("No dist/ directory found — API-only mode (set DEEPBRAIN_DIST_DIR to serve web UI)");
        public.merge(protected).with_state(state).layer(cors)
    };

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!("DeepBrain API listening on http://{}", addr);

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("Failed to bind API server on port {}: {} (app continues without HTTP API)", port, e);
            return;
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        tracing::warn!("API server error: {}", e);
    }
}
