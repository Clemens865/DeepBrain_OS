import { useEffect, useState } from "react";
import { isTauri, getAuthToken } from "./services/backend";
import SearchBar from "./components/SearchBar";
import ResultsList from "./components/ResultsList";
import QuickActions from "./components/QuickActions";
import MemoryFeed from "./components/MemoryFeed";
import Settings from "./components/Settings";
import Onboarding from "./components/Onboarding";
import Toolbar from "./components/Toolbar";
import BrowseView from "./components/BrowseView";
import ChatView from "./components/ChatView";
import BrowserLogin from "./components/BrowserLogin";
import { useAppStore } from "./store/appStore";

type View = "search" | "settings";

function App() {
  const [view, setView] = useState<View>("search");
  const [expanded, setExpanded] = useState(false);
  const [authenticated, setAuthenticated] = useState(isTauri() || !!getAuthToken());
  const {
    query, results, isSearching, loadStatus, loadSettings, settings, mode,
    enterBrowseMode, exitBrowseMode, enterChatMode, exitChatMode,
    pinned, setPinned, expandedView, setExpandedView,
  } = useAppStore();

  useEffect(() => {
    loadStatus();
    loadSettings();

    // Tauri event listeners (IPC events from tray menu, etc.)
    if (!isTauri()) return;

    const unlisteners: Promise<() => void>[] = [];

    (async () => {
      const { listen } = await import("@tauri-apps/api/event");

      // Listen for navigation events from tray
      unlisteners.push(
        listen<string>("navigate", (event) => {
          if (event.payload === "settings") {
            setView("settings");
            setExpanded(true);
          }
        })
      );

      // Listen for pin-changed events from tray menu
      unlisteners.push(
        listen<boolean>("pin-changed", (event) => {
          useAppStore.setState({ pinned: event.payload });
          if (!event.payload && useAppStore.getState().expandedView) {
            useAppStore.getState().setExpandedView(false);
          }
        })
      );

      // Listen for overlay show/hide — focus the appropriate input
      unlisteners.push(
        listen("overlay-shown", () => {
          const chatInput = document.querySelector<HTMLInputElement>("[data-chat-input]");
          if (chatInput) {
            chatInput.focus();
          } else {
            const searchInput = document.querySelector<HTMLInputElement>("[data-search-input]");
            searchInput?.focus();
          }
        })
      );
    })();

    return () => {
      unlisteners.forEach((p) => p.then((fn) => fn()));
    };
  }, [loadStatus, loadSettings]);

  // Expand when we have results or query
  useEffect(() => {
    if (query.length > 0 || results.memories.length > 0) {
      setExpanded(true);
    }
  }, [query, results]);

  // Resize window for browse mode and expanded view (Tauri only)
  useEffect(() => {
    if (!isTauri()) return;

    (async () => {
      const { getCurrentWindow, LogicalSize } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      if (expandedView) {
        win.setSize(new LogicalSize(1200, 800));
        win.setAlwaysOnTop(false);
        win.center();
      } else if (mode === "browse") {
        win.setSize(new LogicalSize(960, 600));
        win.setAlwaysOnTop(true);
        win.center();
      } else if (mode !== "chat") {
        win.setSize(new LogicalSize(680, 480));
        win.setAlwaysOnTop(true);
        win.center();
      }
    })();
  }, [mode, expandedView]);

  // Global keyboard shortcut: Cmd+B for browse toggle
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.metaKey && e.key === "b") {
        e.preventDefault();
        if (mode === "browse") {
          exitBrowseMode();
        } else {
          enterBrowseMode();
          setExpanded(true);
        }
      }
      // Cmd+P for pin toggle
      if (e.metaKey && !e.shiftKey && e.key === "p") {
        e.preventDefault();
        const store = useAppStore.getState();
        const newPinned = !store.pinned;
        store.setPinned(newPinned);
        if (!newPinned && store.expandedView) {
          store.setExpandedView(false);
        }
      }
      // Cmd+Shift+F for expand toggle
      if (e.metaKey && e.shiftKey && e.key === "f") {
        e.preventDefault();
        const store = useAppStore.getState();
        store.setExpandedView(!store.expandedView);
      }
      // Cmd+J for chat toggle
      if (e.metaKey && e.key === "j") {
        e.preventDefault();
        if (mode === "chat") {
          exitChatMode();
        } else {
          enterChatMode();
          setExpanded(true);
        }
      }
      // Cmd+[ / Cmd+] for back/forward
      if (mode === "browse" && e.metaKey && e.key === "[") {
        e.preventDefault();
        useAppStore.getState().navigateBack();
      }
      if (mode === "browse" && e.metaKey && e.key === "]") {
        e.preventDefault();
        useAppStore.getState().navigateForward();
      }
      // Esc in browse mode → exit to search
      if (mode === "browse" && e.key === "Escape") {
        e.preventDefault();
        exitBrowseMode();
      }
      // Esc in chat mode → exit to search
      if (mode === "chat" && e.key === "Escape" && !(e.target instanceof HTMLInputElement)) {
        e.preventDefault();
        exitChatMode();
      }
      // Cmd+Backspace in browse → delete selected
      if (mode === "browse" && e.metaKey && e.key === "Backspace") {
        e.preventDefault();
        const { selectedItemId, selectedItemDetail, deleteBrowseItem } = useAppStore.getState();
        if (selectedItemId && selectedItemDetail?.kind === "memory") {
          deleteBrowseItem(selectedItemId);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [mode, enterBrowseMode, exitBrowseMode, enterChatMode, exitChatMode]);

  // Browser mode: use full viewport; Tauri mode: use fixed overlay sizes
  const isDesktop = isTauri();

  // Browser mode: require authentication
  if (!authenticated) {
    return <BrowserLogin onAuthenticated={() => setAuthenticated(true)} />;
  }

  // Show onboarding on first launch
  if (settings && !settings.onboarded) {
    return (
      <div className={`w-full ${isDesktop ? "h-[480px] rounded-2xl border border-brain-border" : "h-screen"} bg-brain-bg/95 backdrop-blur-xl overflow-hidden animate-fade-in`}>
        <Onboarding />
      </div>
    );
  }

  if (view === "settings") {
    return (
      <div className={`w-full h-screen bg-brain-bg/95 backdrop-blur-xl ${isDesktop ? "rounded-2xl border border-brain-border" : ""} overflow-hidden animate-fade-in`}>
        <Settings onBack={() => setView("search")} />
      </div>
    );
  }

  // Chat mode
  if (mode === "chat") {
    return (
      <div className={`w-full ${isDesktop ? "h-[480px] rounded-2xl border border-brain-border" : "h-screen"} bg-brain-bg/95 backdrop-blur-xl overflow-hidden animate-fade-in flex flex-col`}>
        {isDesktop && <div data-tauri-drag-region className="absolute top-0 left-0 right-0 h-3 cursor-move z-10" />}
        <ChatView />
      </div>
    );
  }

  // Browse mode
  if (mode === "browse") {
    return (
      <div className={`w-full h-screen bg-brain-bg/95 backdrop-blur-xl ${isDesktop ? "rounded-2xl border border-brain-border" : ""} overflow-hidden animate-fade-in flex flex-col`}>
        {isDesktop && <div data-tauri-drag-region className="absolute top-0 left-0 right-0 h-3 cursor-move z-10" />}
        <SearchBar />
        <Toolbar />
        <div className="flex-1 min-h-0">
          <BrowseView />
        </div>
      </div>
    );
  }

  // Search / Remember mode (original layout)
  return (
    <div
      className={`w-full bg-brain-bg/95 backdrop-blur-xl ${isDesktop ? "rounded-2xl border border-brain-border" : ""} overflow-hidden transition-all duration-150 animate-slide-down ${
        isDesktop ? (expanded ? "h-[480px]" : "h-16") : "h-screen"
      }`}
    >
      {isDesktop && <div data-tauri-drag-region className="absolute top-0 left-0 right-0 h-3 cursor-move" />}

      {/* Search bar */}
      <SearchBar />

      {(expanded || !isDesktop) && (
        <div className={`flex flex-col ${isDesktop ? "h-[calc(100%-64px)]" : "h-[calc(100vh-64px)]"} animate-fade-in`}>
          {/* Tab area */}
          {isSearching || results.memories.length > 0 ? (
            <ResultsList />
          ) : (
            <div className="flex-1 flex flex-col">
              <MemoryFeed />
              <QuickActions
                onSettings={() => {
                  setView("settings");
                  setExpanded(true);
                }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
