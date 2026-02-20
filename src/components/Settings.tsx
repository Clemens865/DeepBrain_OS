import { useEffect, useState, useCallback } from "react";
import { invoke, isTauri } from "../services/backend";
import { useAppStore, type LlmStatus } from "../store/appStore";

interface CommonFolder {
  label: string;
  path: string;
  exists: boolean;
}

interface SettingsProps {
  onBack: () => void;
}

export default function Settings({ onBack }: SettingsProps) {
  const { settings, loadSettings, updateSettings, status, addIndexedFolder } = useAppStore();
  const [localSettings, setLocalSettings] = useState(settings);
  const [commonFolders, setCommonFolders] = useState<CommonFolder[]>([]);

  // Default folders are Documents, Desktop, Downloads (always watched)
  const defaultLabels = ["Documents", "Desktop", "Downloads"];
  const defaultFolders = commonFolders
    .filter((f) => defaultLabels.includes(f.label))
    .map((f) => f.path);

  useEffect(() => {
    loadSettings();
    invoke<CommonFolder[]>("get_common_folders").then(setCommonFolders).catch(console.error);
  }, [loadSettings]);

  useEffect(() => {
    if (settings) {
      setLocalSettings(settings);
    }
  }, [settings]);

  const handleSave = useCallback(async () => {
    if (localSettings) {
      await updateSettings(localSettings);
      onBack();
    }
  }, [localSettings, updateSettings, onBack]);

  if (!localSettings) {
    return (
      <div className="p-4 text-brain-text/50 text-sm">Loading settings...</div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center h-14 px-4 border-b border-brain-border">
        <button
          onClick={onBack}
          className="text-brain-text/50 hover:text-white mr-3 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <span className="text-white font-medium">Settings</span>
        <button
          onClick={handleSave}
          className="ml-auto px-3 py-1.5 bg-brain-accent text-white text-xs rounded-lg hover:bg-brain-accent/80 transition-colors"
        >
          Save
        </button>
      </div>

      {/* Settings content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* AI Provider */}
        <Section title="AI Provider">
          <Select
            value={localSettings.ai_provider}
            onChange={(v) => setLocalSettings({ ...localSettings, ai_provider: v })}
            options={[
              { value: "ollama", label: "Ollama (Local)" },
              { value: "ruvllm", label: "RuVector LLM (Local)" },
              { value: "claude", label: "Claude (Cloud)" },
              { value: "none", label: "None (Memory Only)" },
            ]}
          />

          {localSettings.ai_provider === "ruvllm" && (
            <RuvllmSettings
              ruvllmModel={localSettings.ruvllm_model}
              onModelChange={(v) => setLocalSettings({ ...localSettings, ruvllm_model: v })}
            />
          )}

          {localSettings.ai_provider === "ollama" && (
            <div className="mt-3">
              <label className="block text-brain-text/50 text-xs mb-1">Model</label>
              <input
                type="text"
                value={localSettings.ollama_model}
                onChange={(e) =>
                  setLocalSettings({ ...localSettings, ollama_model: e.target.value })
                }
                className="w-full bg-brain-bg text-white text-sm px-3 py-2 rounded-lg border border-brain-border outline-none focus:border-brain-accent/50"
              />
            </div>
          )}

          {localSettings.ai_provider === "claude" && (
            <div className="mt-3">
              <label className="block text-brain-text/50 text-xs mb-1">API Key</label>
              <input
                type="password"
                value={localSettings.claude_api_key || ""}
                onChange={(e) =>
                  setLocalSettings({ ...localSettings, claude_api_key: e.target.value || null })
                }
                placeholder="sk-ant-..."
                className="w-full bg-brain-bg text-white text-sm px-3 py-2 rounded-lg border border-brain-border outline-none focus:border-brain-accent/50"
              />
            </div>
          )}
        </Section>

        {/* Hotkey */}
        <Section title="Global Shortcut">
          <div className="text-brain-text text-sm bg-brain-bg px-3 py-2 rounded-lg border border-brain-border">
            {localSettings.hotkey}
          </div>
        </Section>

        {/* Theme */}
        <Section title="Appearance">
          <Select
            value={localSettings.theme}
            onChange={(v) => setLocalSettings({ ...localSettings, theme: v })}
            options={[
              { value: "dark", label: "Dark" },
              { value: "light", label: "Light" },
              { value: "system", label: "System" },
            ]}
          />
        </Section>

        {/* Data Sources */}
        <Section title="Data Sources">
          <p className="text-brain-text/40 text-[11px] mb-3">
            Choose which folders DeepBrain can read and index.
          </p>

          {/* Quick-toggle folder buttons */}
          <div className="space-y-1.5 mb-3">
            {commonFolders.filter(f => f.exists).map((folder) => {
              const isActive = (localSettings?.indexed_folders ?? []).includes(folder.path)
                || defaultFolders.includes(folder.path);
              const isDefault = defaultFolders.includes(folder.path);
              return (
                <button
                  key={folder.path}
                  onClick={() => {
                    if (isDefault) return; // can't toggle defaults
                    const current = localSettings?.indexed_folders ?? [];
                    if (isActive) {
                      const updated = current.filter(f => f !== folder.path);
                      setLocalSettings({ ...localSettings!, indexed_folders: updated });
                    } else {
                      addIndexedFolder(folder.path);
                      setLocalSettings({ ...localSettings!, indexed_folders: [...current, folder.path] });
                    }
                  }}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg border text-xs transition-colors text-left ${
                    isActive
                      ? "bg-brain-accent/10 border-brain-accent/30 text-brain-accent"
                      : "bg-brain-bg border-brain-border text-brain-text/50 hover:border-brain-text/30"
                  }`}
                >
                  <div className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
                    isActive ? "bg-brain-accent border-brain-accent" : "border-brain-text/30"
                  }`}>
                    {isActive && (
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                  <span className="flex-1 truncate">{folder.label}</span>
                  {isDefault && <span className="text-[9px] text-brain-text/30">default</span>}
                </button>
              );
            })}
          </div>

          {/* Custom folders */}
          {(localSettings?.indexed_folders ?? [])
            .filter(f => !commonFolders.some(cf => cf.path === f))
            .map((folder, i) => (
              <div key={`custom-${i}`} className="flex items-center gap-2 mb-1.5">
                <div className="w-4 h-4 rounded bg-brain-accent border-brain-accent flex items-center justify-center flex-shrink-0">
                  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <span className="flex-1 text-brain-accent text-xs truncate">{folder.split("/").pop()}</span>
                <button
                  onClick={() => {
                    const updated = (localSettings?.indexed_folders ?? []).filter(f => f !== folder);
                    setLocalSettings({ ...localSettings!, indexed_folders: updated });
                  }}
                  className="text-brain-text/30 hover:text-red-400 text-[10px] transition-colors"
                >
                  remove
                </button>
              </div>
            ))}

          {/* Add custom folder */}
          <button
            onClick={async () => {
              try {
                let picked: string | null = null;
                if (isTauri()) {
                  picked = await invoke<string | null>("pick_folder");
                } else {
                  picked = prompt("Enter folder path to index:");
                }
                if (picked) {
                  const current = localSettings?.indexed_folders ?? [];
                  if (!current.includes(picked)) {
                    addIndexedFolder(picked);
                    setLocalSettings({ ...localSettings!, indexed_folders: [...current, picked] });
                  }
                }
              } catch (e) {
                console.error("Folder picker failed:", e);
              }
            }}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-dashed border-brain-text/20 text-brain-text/40 text-xs hover:border-brain-accent/30 hover:text-brain-accent transition-colors mt-2"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Choose folder...
          </button>
        </Section>

        {/* Auto-Start */}
        <Section title="Startup">
          <Toggle
            label="Start at login"
            checked={localSettings?.auto_start ?? false}
            onChange={(v) => setLocalSettings({ ...localSettings!, auto_start: v })}
          />
        </Section>

        {/* Privacy */}
        <Section title="Privacy">
          <Toggle
            label="Privacy Mode (disable cloud AI)"
            checked={localSettings.privacy_mode}
            onChange={(v) => setLocalSettings({ ...localSettings, privacy_mode: v })}
          />
        </Section>

        {/* Info */}
        <Section title="System Info">
          <div className="text-brain-text/50 text-xs space-y-1">
            <p>Embedding: {status?.embedding_provider || "..."}</p>
            <p>AI: {status?.ai_provider || "..."}</p>
            <p>Memories: {status?.memory_count ?? 0}</p>
            <p>Indexed Files: {status?.indexed_files ?? 0} ({status?.indexed_chunks ?? 0} chunks)</p>
            <p>Version: 0.1.0</p>
          </div>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-brain-text/70 text-xs font-medium uppercase tracking-wider mb-2">
        {title}
      </h3>
      {children}
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-brain-bg text-white text-sm px-3 py-2 rounded-lg border border-brain-border outline-none focus:border-brain-accent/50 appearance-none cursor-pointer"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between cursor-pointer">
      <span className="text-brain-text text-sm">{label}</span>
      <div
        onClick={() => onChange(!checked)}
        className={`w-10 h-5 rounded-full transition-colors relative ${
          checked ? "bg-brain-accent" : "bg-brain-border"
        }`}
      >
        <div
          className={`w-4 h-4 rounded-full bg-white absolute top-0.5 transition-transform ${
            checked ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </div>
    </label>
  );
}

function RuvllmSettings({
  ruvllmModel,
  onModelChange,
}: {
  ruvllmModel: string | null;
  onModelChange: (v: string | null) => void;
}) {
  const { loadModel, unloadModel } = useAppStore();
  const [llmStatus, setLlmStatus] = useState<LlmStatus | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    invoke<LlmStatus>("local_model_status").then(setLlmStatus).catch(console.error);
  }, []);

  const handleLoad = async () => {
    if (!ruvllmModel?.trim()) return;
    setLoading(true);
    await loadModel(ruvllmModel.trim());
    const status = await invoke<LlmStatus>("local_model_status").catch(() => null);
    setLlmStatus(status);
    setLoading(false);
  };

  const handleUnload = async () => {
    setLoading(true);
    await unloadModel();
    const status = await invoke<LlmStatus>("local_model_status").catch(() => null);
    setLlmStatus(status);
    setLoading(false);
  };

  return (
    <div className="mt-3 space-y-2">
      <label className="block text-brain-text/50 text-xs mb-1">Model ID or GGUF Path</label>
      <input
        type="text"
        value={ruvllmModel || ""}
        onChange={(e) => onModelChange(e.target.value || null)}
        placeholder="Qwen/Qwen2.5-1.5B-Instruct"
        className="w-full bg-brain-bg text-white text-sm px-3 py-2 rounded-lg border border-brain-border outline-none focus:border-brain-accent/50"
      />

      {/* Status indicator */}
      <div className="flex items-center gap-2 text-xs">
        <div className={`w-2 h-2 rounded-full ${llmStatus?.model_loaded ? "bg-brain-success" : "bg-brain-text/20"}`} />
        <span className="text-brain-text/50">
          {llmStatus?.model_loaded
            ? `Loaded: ${llmStatus.model_id ?? "unknown"}`
            : "No model loaded"}
        </span>
      </div>
      {llmStatus?.model_info && (
        <div className="text-brain-text/30 text-[10px]">{llmStatus.model_info}</div>
      )}

      {/* Load / Unload buttons */}
      <div className="flex gap-2">
        {llmStatus?.model_loaded ? (
          <button
            onClick={handleUnload}
            disabled={loading}
            className="px-3 py-1.5 bg-brain-error/20 text-brain-error text-xs rounded-lg hover:bg-brain-error/30 transition-colors disabled:opacity-50"
          >
            {loading ? "Unloading..." : "Unload Model"}
          </button>
        ) : (
          <button
            onClick={handleLoad}
            disabled={loading || !ruvllmModel?.trim()}
            className="px-3 py-1.5 bg-brain-accent/20 text-brain-accent text-xs rounded-lg hover:bg-brain-accent/30 transition-colors disabled:opacity-30"
          >
            {loading ? "Loading..." : "Load Model"}
          </button>
        )}
      </div>
    </div>
  );
}
