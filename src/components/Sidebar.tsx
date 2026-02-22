import { useState } from "react";
import { useAppStore, type BrowseCategory } from "../store/appStore";

interface SidebarSection {
  label: string;
  icon: React.ReactNode;
  category: BrowseCategory;
  children?: { label: string; category: BrowseCategory }[];
}

const SECTIONS: SidebarSection[] = [
  {
    label: "Home",
    icon: <HomeIcon />,
    category: "home",
  },
  {
    label: "Memories",
    icon: <BrainIcon />,
    category: "all-memories",
    children: [
      { label: "All", category: "all-memories" },
      { label: "Semantic", category: "semantic" },
      { label: "Episodic", category: "episodic" },
      { label: "Working", category: "working" },
      { label: "Procedural", category: "procedural" },
      { label: "Meta", category: "meta" },
      { label: "Causal", category: "causal" },
      { label: "Goal", category: "goal" },
      { label: "Emotional", category: "emotional" },
    ],
  },
  {
    label: "Files",
    icon: <FolderIcon />,
    category: "all-files",
  },
  {
    label: "Thoughts",
    icon: <LightbulbIcon />,
    category: "thoughts",
  },
  {
    label: "Email",
    icon: <EmailIcon />,
    category: "email",
  },
  {
    label: "Clipboard",
    icon: <ClipboardIcon />,
    category: "clipboard",
  },
  {
    label: "Knowledge Graph",
    icon: <GraphIcon />,
    category: "knowledge-graph",
  },
  {
    label: "Dashboard",
    icon: <ChartIcon />,
    category: "status",
  },
];

export default function Sidebar() {
  const { browseCategory, navigateTo, status } = useAppStore();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ Memories: true });

  const toggle = (label: string) => {
    setExpanded((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  const counts: Record<string, number | undefined> = {
    "all-memories": status?.memory_count,
    "all-files": status?.indexed_files,
    email: status?.indexed_emails,
    thoughts: status?.thought_count,
  };

  return (
    <div className="w-40 flex-shrink-0 border-r border-brain-border overflow-y-auto py-2 bg-brain-bg/50">
      {SECTIONS.map((section) => (
        <div key={section.label}>
          {/* Section header */}
          <button
            onClick={() => {
              if (section.children) {
                toggle(section.label);
              }
              navigateTo(section.category);
            }}
            className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs transition-colors ${
              browseCategory === section.category && !section.children
                ? "bg-brain-accent/15 text-brain-accent"
                : "text-brain-text/70 hover:bg-brain-surface hover:text-brain-text"
            }`}
          >
            {section.children && (
              <span className="text-brain-text/30 text-[10px] w-3">
                {expanded[section.label] ? "▾" : "▸"}
              </span>
            )}
            <span className="w-4 h-4 flex-shrink-0">{section.icon}</span>
            <span className="flex-1 text-left truncate">{section.label}</span>
            {counts[section.category] !== undefined && (
              <span className="text-brain-text/30 text-[10px]">
                {counts[section.category]}
              </span>
            )}
          </button>

          {/* Children */}
          {section.children && expanded[section.label] && (
            <div className="ml-5">
              {section.children.map((child) => (
                <button
                  key={child.category}
                  onClick={() => navigateTo(child.category)}
                  className={`w-full flex items-center gap-2 px-3 py-1 text-[11px] transition-colors ${
                    browseCategory === child.category
                      ? "bg-brain-accent/15 text-brain-accent"
                      : "text-brain-text/50 hover:bg-brain-surface hover:text-brain-text/70"
                  }`}
                >
                  <span className="truncate">{child.label}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ---- Icons ----

function HomeIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
    </svg>
  );
}

function LightbulbIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  );
}

function ClipboardIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
    </svg>
  );
}

function EmailIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  );
}

function GraphIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <circle cx="5" cy="12" r="2" strokeWidth={1.5} />
      <circle cx="19" cy="6" r="2" strokeWidth={1.5} />
      <circle cx="19" cy="18" r="2" strokeWidth={1.5} />
      <path strokeLinecap="round" strokeWidth={1.5} d="M7 11l10-4M7 13l10 4" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  );
}
