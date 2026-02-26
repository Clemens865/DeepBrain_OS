//! File Index connector — imports document-type files (md, pdf, txt, etc.) from files.db.

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    deterministic_hash, emit_progress, store_as_memory, BootstrapProgress, SourceResult,
    DOC_EXTENSIONS,
};
use crate::indexer::chunker;
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct FileIndexConnector;

impl FileIndexConnector {
    fn db_path() -> Option<std::path::PathBuf> {
        dirs::data_dir().map(|d| d.join("DeepBrain").join("files.db"))
    }
}

/// Read document rows from files.db synchronously (Connection is !Send).
fn read_file_index_docs(
    db_path: &std::path::Path,
) -> Result<Vec<(String, String, String, String)>, String> {
    if !db_path.exists() {
        return Err("files.db not found".to_string());
    }
    let conn = Connection::open(db_path).map_err(|e| format!("open files.db: {}", e))?;
    let ext_placeholders: Vec<String> = DOC_EXTENSIONS.iter().map(|_| "?".to_string()).collect();
    let sql = format!(
        "SELECT fi.path, fi.name, fi.ext, GROUP_CONCAT(fc.content, ' ') as full_content
         FROM file_index fi
         JOIN file_chunks fc ON fc.file_path = fi.path
         WHERE fi.ext IN ({})
         GROUP BY fi.path
         ORDER BY fi.modified DESC",
        ext_placeholders.join(",")
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| format!("prepare: {}", e))?;
    let params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = DOC_EXTENSIONS
        .iter()
        .map(|e| Box::new(e.to_string()) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

#[async_trait::async_trait]
impl Connector for FileIndexConnector {
    fn id(&self) -> &str {
        "file_index"
    }
    fn name(&self) -> &str {
        "Documents"
    }
    fn description(&self) -> &str {
        "md, pdf, txt, html..."
    }
    fn icon(&self) -> &str {
        "file"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "semantic"
    }
    fn default_importance(&self) -> f64 {
        0.5
    }

    fn detect(&self) -> DetectionResult {
        let path = Self::db_path();
        let available = path.as_ref().map(|p| p.exists()).unwrap_or(false);
        DetectionResult {
            available,
            path: path.map(|p| p.to_string_lossy().to_string()),
            details: None,
        }
    }

    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult {
        let mut result = SourceResult {
            source: "file_index".to_string(),
            ..Default::default()
        };

        let db_path = config
            .path_override
            .as_ref()
            .map(std::path::PathBuf::from)
            .or_else(Self::db_path);

        let db_path = match db_path {
            Some(p) => p,
            None => return result,
        };

        let rows = match read_file_index_docs(&db_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Bootstrap file_index: {}", e);
                return result;
            }
        };

        let total = rows.len() as u32;
        tracing::info!("Bootstrap file_index: found {} document files", total);
        let base_importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, (path, name, ext, content)) in rows.iter().enumerate() {
            result.items_scanned += 1;

            let path_hash = deterministic_hash(path);
            let memory_id = format!("bootstrap::file::{}", path_hash);

            if state.engine.memory.has_memory(&memory_id) {
                result.skipped_existing += 1;
                continue;
            }

            let truncated: String = content
                .split_whitespace()
                .take(4000)
                .collect::<Vec<&str>>()
                .join(" ");

            let enriched = format!("[Document: {} ({})] {}", name, path, truncated);
            let chunks = chunker::chunk_text(&enriched, 512, 128);
            if chunks.is_empty() {
                continue;
            }

            let importance = match ext.as_str() {
                "md" | "txt" | "rtf" | "org" | "rst" | "adoc" => base_importance,
                "pdf" | "docx" | "xlsx" | "xls" => base_importance + 0.05,
                _ => base_importance - 0.05,
            };

            for (ci, chunk) in chunks.iter().enumerate().take(5) {
                let chunk_id = format!("{}::{}", memory_id, ci);
                match store_as_memory(state, &chunk_id, chunk, "semantic", importance).await {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap file_index error for {}: {}", name, e);
                        result.errors += 1;
                    }
                }
            }

            emit_progress(
                app_handle,
                &BootstrapProgress {
                    source: "file_index".to_string(),
                    phase: "storing".to_string(),
                    current: i as u32 + 1,
                    total,
                    memories_created: result.memories_created,
                },
            );

            if i % 10 == 9 {
                tokio::task::yield_now().await;
            }
        }

        tracing::info!(
            "Bootstrap file_index complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
