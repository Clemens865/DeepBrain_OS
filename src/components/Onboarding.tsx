import { useCallback, useEffect, useState } from "react";
import { invoke } from "../services/backend";
import { useAppStore } from "../store/appStore";

interface OllamaStatus {
  available: boolean;
  models: string[];
}

type Step = "welcome" | "ai" | "ready";

export default function Onboarding() {
  const { updateSettings, settings, loadSettings } = useAppStore();
  const [step, setStep] = useState<Step>("welcome");
  const [ollama, setOllama] = useState<OllamaStatus | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const checkOllama = useCallback(async () => {
    setChecking(true);
    try {
      const status = await invoke<OllamaStatus>("check_ollama");
      setOllama(status);
    } catch {
      setOllama({ available: false, models: [] });
    }
    setChecking(false);
  }, []);

  useEffect(() => {
    if (step === "ai") {
      checkOllama();
    }
  }, [step, checkOllama]);

  const [indexing, setIndexing] = useState(false);
  const [indexCount, setIndexCount] = useState<number | null>(null);

  const finish = useCallback(async () => {
    if (settings) {
      // Trigger initial file index
      setIndexing(true);
      try {
        const count = await invoke<number>("index_files");
        setIndexCount(count);
      } catch {
        // Non-fatal - indexing can happen in background
      }
      setIndexing(false);
      await updateSettings({ ...settings, onboarded: true });
    }
  }, [settings, updateSettings]);

  return (
    <div className="flex flex-col h-full bg-brain-bg">
      {step === "welcome" && (
        <div className="flex flex-col items-center justify-center flex-1 px-8 text-center animate-fade-in">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brain-accent to-purple-500 flex items-center justify-center mb-6">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h1 className="text-white text-xl font-semibold mb-2">Welcome to DeepBrain</h1>
          <p className="text-brain-text/60 text-sm mb-1">
            Your intelligent cognitive layer for macOS.
          </p>
          <p className="text-brain-text/40 text-xs mb-8 max-w-[280px]">
            Semantic memory, file search, and AI assistance — all from your menu bar.
          </p>
          <button
            onClick={() => setStep("ai")}
            className="px-6 py-2.5 bg-brain-accent text-white text-sm font-medium rounded-xl hover:bg-brain-accent/80 transition-colors"
          >
            Get Started
          </button>
        </div>
      )}

      {step === "ai" && (
        <div className="flex flex-col flex-1 px-8 py-6 animate-fade-in">
          <h2 className="text-white text-lg font-semibold mb-1">AI Provider</h2>
          <p className="text-brain-text/50 text-xs mb-5">
            DeepBrain can use a local or cloud AI for enhanced responses.
          </p>

          {/* Ollama status */}
          <div className="bg-brain-surface rounded-xl p-4 border border-brain-border mb-3">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${
                checking ? "bg-brain-warning animate-pulse" :
                ollama?.available ? "bg-brain-success" : "bg-brain-error"
              }`} />
              <span className="text-white text-sm font-medium">Ollama (Local AI)</span>
            </div>
            {checking ? (
              <p className="text-brain-text/50 text-xs">Checking...</p>
            ) : ollama?.available ? (
              <div>
                <p className="text-brain-success text-xs mb-1">Connected! {ollama.models.length} model(s) available</p>
                {ollama.models.length > 0 && (
                  <p className="text-brain-text/40 text-[10px]">
                    {ollama.models.slice(0, 3).join(", ")}
                    {ollama.models.length > 3 && ` +${ollama.models.length - 3} more`}
                  </p>
                )}
              </div>
            ) : (
              <div>
                <p className="text-brain-text/50 text-xs mb-1">Not detected</p>
                <p className="text-brain-text/40 text-[10px]">
                  Install from ollama.ai for local AI (no internet needed)
                </p>
              </div>
            )}
          </div>

          {/* None option */}
          <div className="bg-brain-surface rounded-xl p-4 border border-brain-border mb-5">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-brain-accent" />
              <span className="text-white text-sm font-medium">Memory Only</span>
            </div>
            <p className="text-brain-text/50 text-xs mt-1">
              Works great without AI — store, search, and recall memories using vector similarity.
            </p>
          </div>

          <div className="mt-auto flex gap-3">
            <button
              onClick={() => setStep("welcome")}
              className="px-4 py-2 text-brain-text/50 text-sm rounded-xl hover:bg-brain-surface transition-colors"
            >
              Back
            </button>
            <button
              onClick={() => setStep("ready")}
              className="flex-1 px-4 py-2.5 bg-brain-accent text-white text-sm font-medium rounded-xl hover:bg-brain-accent/80 transition-colors"
            >
              Continue
            </button>
          </div>
        </div>
      )}

      {step === "ready" && (
        <div className="flex flex-col items-center justify-center flex-1 px-8 text-center animate-fade-in">
          <div className="w-12 h-12 rounded-full bg-brain-success/20 flex items-center justify-center mb-5">
            <svg className="w-6 h-6 text-brain-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h2 className="text-white text-lg font-semibold mb-2">You're All Set!</h2>
          <p className="text-brain-text/50 text-xs mb-2 max-w-[280px]">
            Press <span className="text-brain-accent font-medium">Cmd+Shift+Space</span> anytime to open DeepBrain.
          </p>
          <p className="text-brain-text/40 text-[10px] mb-8 max-w-[250px]">
            {indexing
              ? "Indexing your files..."
              : indexCount !== null
              ? `Indexed ${indexCount} files from your Documents, Desktop, and Downloads.`
              : "Click below to index your files and start using DeepBrain."}
          </p>
          <button
            onClick={finish}
            disabled={indexing}
            className={`px-6 py-2.5 text-white text-sm font-medium rounded-xl transition-colors ${
              indexing
                ? "bg-brain-accent/50 cursor-wait"
                : "bg-brain-accent hover:bg-brain-accent/80"
            }`}
          >
            {indexing ? "Indexing..." : "Start Using DeepBrain"}
          </button>
        </div>
      )}
    </div>
  );
}
