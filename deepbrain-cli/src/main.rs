//! DeepBrain CLI — talk to the running DeepBrain app via its HTTP API.

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "deepbrain", about = "CLI for DeepBrain cognitive engine")]
struct Cli {
    /// API base URL
    #[arg(long, env = "DEEPBRAIN_URL", default_value = "http://127.0.0.1:19519")]
    url: String,

    /// Bearer token for authentication
    #[arg(long, env = "DEEPBRAIN_TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search memories by semantic similarity
    Search {
        query: String,
        #[arg(short, long, default_value = "5")]
        limit: u32,
    },
    /// Store a new memory
    Remember {
        content: String,
        #[arg(short = 't', long, default_value = "semantic")]
        memory_type: String,
        #[arg(short, long)]
        importance: Option<f64>,
    },
    /// AI-enhanced thinking
    Think { question: String },
    /// Search indexed files
    Files {
        query: String,
        #[arg(short, long, default_value = "10")]
        limit: u32,
    },
    /// Show system status
    Status,
    /// Run a workflow
    Workflow {
        /// Workflow action: digest, summarize, remember_clipboard, search_and_remember
        action: String,
        /// Query (for search_and_remember)
        query: Option<String>,
    },
    /// Show clipboard history
    Clipboard,
    /// Health check
    Health,
}

// ---- Response types (mirror the server) ----

#[derive(Deserialize)]
struct ThinkResponse {
    response: String,
    confidence: f64,
    ai_enhanced: bool,
}

#[derive(Deserialize)]
struct RememberResponse {
    id: String,
    memory_count: u32,
}

#[derive(Deserialize)]
struct RecallItem {
    content: String,
    similarity: f64,
    memory_type: String,
}

#[derive(Deserialize)]
struct SystemStatus {
    status: String,
    memory_count: u32,
    thought_count: u32,
    ai_provider: String,
    ai_available: bool,
    indexed_files: u32,
    indexed_chunks: u32,
}

#[derive(Deserialize)]
struct FileResult {
    path: String,
    name: String,
    chunk: String,
    similarity: f64,
}

#[derive(Deserialize)]
struct WorkflowResult {
    action: String,
    success: bool,
    message: String,
}

#[derive(Deserialize)]
struct ClipboardEntry {
    content: String,
    timestamp: i64,
}

#[derive(Deserialize)]
struct HealthResponse {
    ok: bool,
}

#[derive(Serialize)]
struct RecallReq {
    query: String,
    limit: Option<u32>,
}

#[derive(Serialize)]
struct RememberReq {
    content: String,
    memory_type: Option<String>,
    importance: Option<f64>,
}

#[derive(Serialize)]
struct ThinkReq {
    input: String,
}

#[derive(Serialize)]
struct FileSearchReq {
    query: String,
    limit: Option<u32>,
}

#[derive(Serialize)]
struct WorkflowReq {
    action: String,
    query: Option<String>,
}

fn client(token: &Option<String>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder();
    if let Some(ref t) = token {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", t).parse().unwrap(),
        );
        builder = builder.default_headers(headers);
    }
    builder.build().expect("Failed to build HTTP client")
}

async fn run(cli: Cli) -> Result<(), String> {
    let http = client(&cli.token);
    let base = cli.url.trim_end_matches('/');

    match cli.command {
        Commands::Health => {
            let resp: HealthResponse = http
                .get(format!("{}/api/health", base))
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            println!("{}", if resp.ok { "DeepBrain is running" } else { "DeepBrain is not healthy" });
            Ok(())
        }
        Commands::Status => {
            let resp: SystemStatus = http
                .get(format!("{}/api/status", base))
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            println!("Status:      {}", resp.status);
            println!("Memories:    {}", resp.memory_count);
            println!("Thoughts:    {}", resp.thought_count);
            println!("AI Provider: {} (available: {})", resp.ai_provider, resp.ai_available);
            println!("Files:       {} ({} chunks)", resp.indexed_files, resp.indexed_chunks);
            Ok(())
        }
        Commands::Search { query, limit } => {
            let items: Vec<RecallItem> = http
                .post(format!("{}/api/recall", base))
                .json(&RecallReq { query, limit: Some(limit) })
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            if items.is_empty() {
                println!("No matching memories found.");
            } else {
                for (i, item) in items.iter().enumerate() {
                    println!("{}. [{}] ({:.1}%) {}", i + 1, item.memory_type, item.similarity * 100.0, item.content);
                }
            }
            Ok(())
        }
        Commands::Remember { content, memory_type, importance } => {
            let resp: RememberResponse = http
                .post(format!("{}/api/remember", base))
                .json(&RememberReq {
                    content,
                    memory_type: Some(memory_type),
                    importance,
                })
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            println!("Stored memory {} (total: {})", &resp.id[..8], resp.memory_count);
            Ok(())
        }
        Commands::Think { question } => {
            let resp: ThinkResponse = http
                .post(format!("{}/api/think", base))
                .json(&ThinkReq { input: question })
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            if resp.ai_enhanced {
                println!("[AI] ");
            }
            println!("{}", resp.response);
            println!("\nConfidence: {:.1}%", resp.confidence * 100.0);
            Ok(())
        }
        Commands::Files { query, limit } => {
            let items: Vec<FileResult> = http
                .post(format!("{}/api/search/files", base))
                .json(&FileSearchReq { query, limit: Some(limit) })
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            if items.is_empty() {
                println!("No matching files found.");
            } else {
                for (i, item) in items.iter().enumerate() {
                    println!("{}. {} ({:.1}%)", i + 1, item.path, item.similarity * 100.0);
                    println!("   {}", item.chunk.chars().take(120).collect::<String>());
                }
            }
            Ok(())
        }
        Commands::Workflow { action, query } => {
            let resp: WorkflowResult = http
                .post(format!("{}/api/workflow", base))
                .json(&WorkflowReq { action, query })
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            println!("[{}] {}", if resp.success { "OK" } else { "FAIL" }, resp.message);
            Ok(())
        }
        Commands::Clipboard => {
            let items: Vec<ClipboardEntry> = http
                .get(format!("{}/api/clipboard", base))
                .send()
                .await
                .map_err(|e| format!("Connection failed: {}", e))?
                .json()
                .await
                .map_err(|e| format!("Parse error: {}", e))?;
            if items.is_empty() {
                println!("No clipboard history.");
            } else {
                for (i, item) in items.iter().enumerate() {
                    let preview: String = item.content.chars().take(80).collect();
                    println!("{}. {} (ts: {})", i + 1, preview, item.timestamp);
                }
            }
            Ok(())
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli).await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
