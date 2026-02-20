import { useAppStore, type BrowseCategory } from "../store/appStore";

const CATEGORY_LABELS: Record<BrowseCategory, string> = {
  home: "Home",
  "all-memories": "All Memories",
  semantic: "Semantic",
  episodic: "Episodic",
  working: "Working",
  procedural: "Procedural",
  meta: "Meta",
  causal: "Causal",
  goal: "Goal",
  emotional: "Emotional",
  "all-files": "All Files",
  folder: "Folder",
  thoughts: "Thoughts",
  clipboard: "Clipboard",
  spotlight: "Spotlight",
  email: "Email",
  status: "Status",
};

const MEMORY_TYPES: BrowseCategory[] = [
  "semantic", "episodic", "working", "procedural", "meta", "causal", "goal", "emotional",
];

function parentCategory(cat: BrowseCategory): { label: string; category: BrowseCategory } | null {
  if (MEMORY_TYPES.includes(cat)) {
    return { label: "Memories", category: "all-memories" };
  }
  if (cat === "folder") {
    return { label: "Files", category: "all-files" };
  }
  return null;
}

export default function Toolbar() {
  const {
    browseCategory,
    browseFilter,
    browseItems,
    browseLoading,
    navIndex,
    navHistory,
    showPreview,
    navigateBack,
    navigateForward,
    navigateTo,
    togglePreview,
  } = useAppStore();

  const canGoBack = navIndex > 0;
  const canGoForward = navIndex < navHistory.length - 1;
  const parent = parentCategory(browseCategory);

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-brain-border bg-brain-surface/30 text-xs">
      {/* Back / Forward */}
      <button
        onClick={navigateBack}
        disabled={!canGoBack}
        className={`p-1 rounded transition-colors ${
          canGoBack ? "text-brain-text/70 hover:bg-brain-surface hover:text-white" : "text-brain-text/20"
        }`}
        title="Back (Cmd+[)"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>
      <button
        onClick={navigateForward}
        disabled={!canGoForward}
        className={`p-1 rounded transition-colors ${
          canGoForward ? "text-brain-text/70 hover:bg-brain-surface hover:text-white" : "text-brain-text/20"
        }`}
        title="Forward (Cmd+])"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {/* Breadcrumb */}
      <div className="flex items-center gap-1 flex-1 min-w-0 text-brain-text/60">
        {parent && (
          <>
            <button
              onClick={() => navigateTo(parent.category)}
              className="hover:text-brain-accent transition-colors truncate"
            >
              {parent.label}
            </button>
            <span className="text-brain-text/30">›</span>
          </>
        )}
        <span className="text-white truncate">
          {browseFilter
            ? browseFilter.split("/").pop() || CATEGORY_LABELS[browseCategory]
            : CATEGORY_LABELS[browseCategory]}
        </span>
      </div>

      {/* Item count */}
      {browseItems.length > 0 && (
        <span className="text-brain-text/30">
          {browseLoading ? "..." : `${browseItems.length} items`}
        </span>
      )}

      {/* Preview toggle */}
      <button
        onClick={togglePreview}
        className={`p-1 rounded transition-colors ${
          showPreview ? "text-brain-accent bg-brain-accent/10" : "text-brain-text/40 hover:text-brain-text/70"
        }`}
        title="Toggle preview pane"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
        </svg>
      </button>
    </div>
  );
}
