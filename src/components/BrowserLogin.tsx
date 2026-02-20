import { useState } from "react";
import { setAuthToken } from "../services/backend";

interface BrowserLoginProps {
  onAuthenticated: () => void;
}

export default function BrowserLogin({ onAuthenticated }: BrowserLoginProps) {
  const [token, setToken] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = token.trim();
    if (!trimmed) {
      setError("Please enter your API token.");
      return;
    }

    // Test the token against the health endpoint (or any protected route)
    try {
      const resp = await fetch("/api/status", {
        headers: { Authorization: `Bearer ${trimmed}` },
      });
      if (resp.ok) {
        setAuthToken(trimmed);
        onAuthenticated();
      } else if (resp.status === 401) {
        setError("Invalid token. Please check and try again.");
      } else {
        setError(`Server error: ${resp.status}`);
      }
    } catch {
      setError("Cannot reach DeepBrain server. Is it running?");
    }
  };

  return (
    <div className="h-screen w-full bg-brain-bg flex items-center justify-center">
      <div className="w-full max-w-sm p-8">
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-brain-accent to-purple-500 flex items-center justify-center mb-4">
            <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h1 className="text-white text-xl font-semibold">DeepBrain</h1>
          <p className="text-brain-text/50 text-sm mt-1">Enter your API token to connect</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="password"
              value={token}
              onChange={(e) => {
                setToken(e.target.value);
                setError("");
              }}
              placeholder="Paste API token..."
              className="w-full bg-brain-surface text-white text-sm px-4 py-3 rounded-xl border border-brain-border outline-none focus:border-brain-accent/50 placeholder-brain-text/30"
              autoFocus
            />
          </div>

          {error && (
            <p className="text-brain-error text-xs">{error}</p>
          )}

          <button
            type="submit"
            className="w-full px-4 py-3 bg-brain-accent text-white text-sm font-medium rounded-xl hover:bg-brain-accent/80 transition-colors"
          >
            Connect
          </button>
        </form>

        <div className="mt-6 p-3 bg-brain-surface/50 rounded-lg border border-brain-border/50">
          <p className="text-brain-text/40 text-[11px] leading-relaxed">
            Find your token in Terminal:
          </p>
          <code className="block text-brain-accent/70 text-[10px] mt-1 break-all">
            security find-generic-password -s DeepBrain -a api_token -w
          </code>
        </div>
      </div>
    </div>
  );
}
