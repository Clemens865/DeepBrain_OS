import { useEffect } from "react";
import { useAppStore } from "../store/appStore";
import Sidebar from "./Sidebar";
import ContentList from "./ContentList";
import PreviewPane from "./PreviewPane";
import MemoryFeed from "./MemoryFeed";
import DashboardView from "./DashboardView";
import KnowledgeGraphView from "./KnowledgeGraphView";
import KnowledgeExportView from "./KnowledgeExportView";

export default function BrowseView() {
  const { browseCategory, showPreview, loadBrowseItems, loadStatus } = useAppStore();

  // Load items when entering browse or when category changes
  useEffect(() => {
    if (browseCategory !== "home" && browseCategory !== "status" && browseCategory !== "knowledge-graph" && browseCategory !== "knowledge-export") {
      loadBrowseItems();
    }
    if (browseCategory === "home") {
      loadStatus();
    }
  }, [browseCategory, loadBrowseItems, loadStatus]);

  return (
    <div className="flex h-full">
      <Sidebar />

      {/* Main content area */}
      {browseCategory === "home" ? (
        <div className="flex-1 min-w-0 overflow-y-auto">
          <MemoryFeed />
        </div>
      ) : browseCategory === "status" ? (
        <div className="flex-1 min-w-0">
          <DashboardView />
        </div>
      ) : browseCategory === "knowledge-graph" ? (
        <div className="flex-1 min-w-0">
          <KnowledgeGraphView />
        </div>
      ) : browseCategory === "knowledge-export" ? (
        <div className="flex-1 min-w-0">
          <KnowledgeExportView />
        </div>
      ) : (
        <>
          <ContentList />
          {showPreview && <PreviewPane />}
        </>
      )}
    </div>
  );
}
