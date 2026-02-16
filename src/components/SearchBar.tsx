import { useRef, useCallback } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useAppStore } from "../store/appStore";

let debounceTimer: ReturnType<typeof setTimeout>;

export default function SearchBar() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { query, setQuery, search, isSearching, clearResults } = useAppStore();

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setQuery(value);

      clearTimeout(debounceTimer);
      if (value.trim().length === 0) {
        clearResults();
        return;
      }

      debounceTimer = setTimeout(() => {
        search(value.trim());
      }, 150);
    },
    [setQuery, search, clearResults],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") {
        if (query) {
          clearResults();
          setQuery("");
        } else {
          // Hide window via Tauri
          getCurrentWindow().hide();
        }
      } else if (e.key === "Enter" && query.trim()) {
        search(query.trim());
      }
    },
    [query, search, clearResults],
  );

  return (
    <div className="flex items-center h-16 px-4 border-b border-brain-border">
      {/* Search icon */}
      <div className="text-brain-text/50 mr-3">
        {isSearching ? (
          <svg
            className="w-5 h-5 animate-spin"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        ) : (
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        )}
      </div>

      {/* Search input */}
      <input
        ref={inputRef}
        data-search-input
        type="text"
        value={query}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Search, ask, or remember anything..."
        className="flex-1 bg-transparent text-white text-lg outline-none placeholder-brain-text/40"
        autoFocus
        spellCheck={false}
      />

      {/* Clear button */}
      {query && (
        <button
          onClick={() => clearResults()}
          className="text-brain-text/40 hover:text-brain-text ml-2 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      )}
    </div>
  );
}
