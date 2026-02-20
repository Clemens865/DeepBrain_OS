import { useRef, useCallback } from "react";
import { isTauri } from "../services/backend";
import { useAppStore } from "../store/appStore";

let debounceTimer: ReturnType<typeof setTimeout>;

export default function SearchBar() {
  const inputRef = useRef<HTMLInputElement>(null);
  const {
    query, setQuery, search, isSearching, clearResults,
    mode, setMode, remember, enterBrowseMode, exitBrowseMode,
    enterChatMode,
    browseFilterText, setBrowseFilterText,
    pinned, setPinned, expandedView, setExpandedView,
  } = useAppStore();

  const isBrowse = mode === "browse";

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;

      if (isBrowse) {
        setBrowseFilterText(value);
        return;
      }

      setQuery(value);

      if (mode === "remember") return; // Don't auto-search in remember mode

      clearTimeout(debounceTimer);
      if (value.trim().length === 0) {
        clearResults();
        return;
      }

      debounceTimer = setTimeout(() => {
        search(value.trim());
      }, 150);
    },
    [setQuery, search, clearResults, mode, isBrowse, setBrowseFilterText],
  );

  const handleKeyDown = useCallback(
    async (e: React.KeyboardEvent) => {
      // In browse mode, Escape exits browse; other keys handled globally
      if (isBrowse) return;

      if (e.key === "Escape") {
        if (query) {
          clearResults();
          setQuery("");
        } else if (isTauri()) {
          import("@tauri-apps/api/window").then(({ getCurrentWindow }) => getCurrentWindow().hide());
        }
      } else if (e.key === "Enter" && query.trim()) {
        if (mode === "remember") {
          await remember(query.trim(), "semantic", 0.7);
          setQuery("");
        } else {
          search(query.trim());
        }
      } else if (e.key === "Tab") {
        e.preventDefault();
        setMode(mode === "search" ? "remember" : "search");
      }
    },
    [query, search, clearResults, setQuery, mode, setMode, remember, isBrowse],
  );

  const isRemember = mode === "remember";

  return (
    <div className="flex items-center h-16 px-4 border-b border-brain-border">
      {/* Mode toggle */}
      <button
        onClick={() => {
          if (isBrowse) return;
          setMode(isRemember ? "search" : "remember");
        }}
        className={`mr-3 transition-colors ${
          isRemember ? "text-brain-success" : isBrowse ? "text-brain-accent" : "text-brain-text/50"
        }`}
        title={isBrowse ? "Browse mode (Cmd+B to exit)" : `${isRemember ? "Remember" : "Search"} mode (Tab to switch)`}
      >
        {isSearching ? (
          <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : isRemember ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        )}
      </button>

      {/* Input */}
      <input
        ref={inputRef}
        data-search-input
        type="text"
        value={isBrowse ? browseFilterText : query}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={
          isBrowse
            ? "Filter items... (Esc to exit browse)"
            : isRemember
            ? "Type something to remember... (Enter to save, Tab to switch)"
            : "Search, ask, or remember... (Tab to switch mode)"
        }
        className={`flex-1 bg-transparent text-lg outline-none ${
          isRemember
            ? "text-brain-success placeholder-brain-success/30"
            : isBrowse
            ? "text-brain-accent placeholder-brain-accent/30"
            : "text-white placeholder-brain-text/40"
        }`}
        autoFocus
        spellCheck={false}
      />

      {/* Mode badge */}
      <span className={`text-[10px] px-2 py-0.5 rounded-full ml-2 ${
        isBrowse
          ? "bg-brain-accent/20 text-brain-accent"
          : isRemember
          ? "bg-brain-success/20 text-brain-success"
          : "bg-brain-surface text-brain-text/40"
      }`}>
        {isBrowse ? "Browse" : isRemember ? "Remember" : "Search"}
      </span>

      {/* Pin toggle */}
      <button
        onClick={() => setPinned(!pinned)}
        className={`ml-2 p-1.5 rounded transition-colors ${
          pinned
            ? "text-brain-accent bg-brain-accent/15"
            : "text-brain-text/30 hover:text-brain-text/60 hover:bg-brain-surface"
        }`}
        title={pinned ? "Unpin window (Cmd+P)" : "Pin window (Cmd+P)"}
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill={pinned ? "currentColor" : "none"} stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M16 3l-4 4-4-1-4 4 6 6 4-4-1-4 4-4-1-1zm-5 11l-5 5" />
        </svg>
      </button>

      {/* Expand toggle */}
      <button
        onClick={() => setExpandedView(!expandedView)}
        className={`p-1.5 rounded transition-colors ${
          expandedView
            ? "text-brain-accent bg-brain-accent/15"
            : "text-brain-text/30 hover:text-brain-text/60 hover:bg-brain-surface"
        }`}
        title={expandedView ? "Collapse window (Cmd+Shift+F)" : "Expand window (Cmd+Shift+F)"}
      >
        {expandedView ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 9L4 4m0 0v4m0-4h4m6 6l5 5m0 0v-4m0 4h-4" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 3h6m0 0v6m0-6l-7 7M9 21H3m0 0v-6m0 6l7-7" />
          </svg>
        )}
      </button>

      {/* Chat toggle */}
      <button
        onClick={enterChatMode}
        className="p-1.5 rounded transition-colors text-brain-text/30 hover:text-brain-text/60 hover:bg-brain-surface"
        title="Open chat (Cmd+J)"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      </button>

      {/* Browse toggle */}
      <button
        onClick={() => {
          if (isBrowse) {
            exitBrowseMode();
          } else {
            enterBrowseMode();
          }
        }}
        className={`p-1.5 rounded transition-colors ${
          isBrowse
            ? "text-brain-accent bg-brain-accent/15"
            : "text-brain-text/30 hover:text-brain-text/60 hover:bg-brain-surface"
        }`}
        title="Toggle browse mode (Cmd+B)"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
        </svg>
      </button>

      {/* Clear button */}
      {(query || browseFilterText) && (
        <button
          onClick={() => {
            if (isBrowse) {
              setBrowseFilterText("");
            } else {
              clearResults();
            }
          }}
          className="text-brain-text/40 hover:text-brain-text ml-2 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
}
