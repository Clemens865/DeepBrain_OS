import { useState } from "react";
import { useAppStore } from "../store/appStore";

type Tab = "response" | "memories" | "files" | "email" | "system";

export default function ResultsList() {
  const [activeTab, setActiveTab] = useState<Tab>("response");
  const { results, isSearching } = useAppStore();
  const { memories, files, spotlight, emails, semanticEmails, thinkResult } = results;

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Tabs */}
      <div className="flex gap-1 px-4 pt-2 pb-1 border-b border-brain-border">
        <TabButton
          active={activeTab === "response"}
          onClick={() => setActiveTab("response")}
          label="Response"
          count={thinkResult ? 1 : 0}
        />
        <TabButton
          active={activeTab === "memories"}
          onClick={() => setActiveTab("memories")}
          label="Memories"
          count={memories.length}
        />
        <TabButton
          active={activeTab === "files"}
          onClick={() => setActiveTab("files")}
          label="Files"
          count={files.length}
        />
        <TabButton
          active={activeTab === "email"}
          onClick={() => setActiveTab("email")}
          label="Email"
          count={semanticEmails.length + emails.length}
        />
        <TabButton
          active={activeTab === "system"}
          onClick={() => setActiveTab("system")}
          label="System"
          count={spotlight.length}
        />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isSearching && (
          <div className="text-brain-text/50 text-sm text-center py-8">
            Searching...
          </div>
        )}

        {!isSearching && activeTab === "response" && thinkResult && (
          <div className="space-y-3 animate-fade-in">
            <div className="bg-brain-surface rounded-xl p-4 border border-brain-border">
              <p className="text-white text-sm leading-relaxed">
                {thinkResult.response}
              </p>
              <div className="flex items-center gap-3 mt-3 text-xs text-brain-text/50">
                <span>Confidence: {(thinkResult.confidence * 100).toFixed(0)}%</span>
                <span>{thinkResult.memory_count} memories used</span>
                {thinkResult.ai_enhanced && (
                  <span className="text-brain-accent">AI Enhanced</span>
                )}
              </div>
            </div>
          </div>
        )}

        {!isSearching && activeTab === "memories" && (
          <div className="space-y-2 animate-fade-in">
            {memories.length === 0 ? (
              <div className="text-brain-text/50 text-sm text-center py-8">
                No matching memories found
              </div>
            ) : (
              memories.map((memory) => (
                <div
                  key={memory.id}
                  className="bg-brain-surface rounded-lg p-3 border border-brain-border hover:border-brain-accent/30 transition-colors cursor-default"
                >
                  <p className="text-white text-sm leading-relaxed line-clamp-2">
                    {memory.content}
                  </p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-brain-text/50">
                    <TypeBadge type={memory.memory_type} />
                    <span>{(memory.similarity * 100).toFixed(0)}% match</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {!isSearching && activeTab === "files" && (
          <div className="space-y-2 animate-fade-in">
            {files.length === 0 ? (
              <div className="text-brain-text/50 text-sm text-center py-8">
                No matching files found
              </div>
            ) : (
              files.map((file, i) => (
                <div
                  key={`${file.path}-${i}`}
                  className="bg-brain-surface rounded-lg p-3 border border-brain-border hover:border-brain-accent/30 transition-colors cursor-default"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <svg className="w-3.5 h-3.5 text-brain-accent shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    <span className="text-white text-sm font-medium truncate">{file.name}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-brain-accent/20 text-brain-accent font-medium">
                      .{file.file_type}
                    </span>
                  </div>
                  <p className="text-brain-text/70 text-xs leading-relaxed line-clamp-2 ml-5.5">
                    {file.chunk}
                  </p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-brain-text/50 ml-5.5">
                    <span className="truncate max-w-[300px]">{file.path}</span>
                    <span>{(file.similarity * 100).toFixed(0)}% match</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {!isSearching && activeTab === "email" && (
          <div className="space-y-2 animate-fade-in">
            {semanticEmails.length === 0 && emails.length === 0 ? (
              <div className="text-brain-text/50 text-sm text-center py-8">
                No matching emails found
              </div>
            ) : (
              <>
                {/* Semantic results (from indexed/embedded emails) */}
                {semanticEmails.length > 0 && (
                  <>
                    <div className="text-[10px] uppercase tracking-wider text-brain-text/30 px-1 pt-1">
                      Indexed emails ({semanticEmails.length})
                    </div>
                    {semanticEmails.map((email, i) => (
                      <div
                        key={`sem-${i}`}
                        className="bg-brain-surface rounded-lg p-3 border border-brain-border hover:border-blue-500/30 transition-colors cursor-default"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <svg className="w-3.5 h-3.5 text-blue-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          </svg>
                          <span className="text-white text-sm font-medium truncate">{email.subject}</span>
                          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 font-medium">
                            {(email.similarity * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex items-center gap-3 mt-1 text-xs text-brain-text/50 ml-5.5">
                          <span className="truncate">{email.sender}</span>
                          <span className="text-brain-text/30">{email.mailbox}</span>
                        </div>
                        {email.chunk && (
                          <p className="text-brain-text/40 text-xs leading-relaxed line-clamp-2 ml-5.5 mt-1">
                            {email.chunk}
                          </p>
                        )}
                        <div className="text-brain-text/30 text-[10px] ml-5.5 mt-1">
                          {email.date}
                        </div>
                      </div>
                    ))}
                  </>
                )}

                {/* Live mail results (from AppleScript search) */}
                {emails.length > 0 && (
                  <>
                    <div className="text-[10px] uppercase tracking-wider text-brain-text/30 px-1 pt-2">
                      Live search ({emails.length})
                    </div>
                    {emails.map((email, i) => (
                      <div
                        key={`live-${email.message_id}-${i}`}
                        className="bg-brain-surface rounded-lg p-3 border border-brain-border hover:border-blue-500/30 transition-colors cursor-default"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <svg className="w-3.5 h-3.5 text-blue-400/60 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          </svg>
                          <span className="text-white text-sm font-medium truncate">{email.subject}</span>
                        </div>
                        <div className="flex items-center gap-3 mt-1 text-xs text-brain-text/50 ml-5.5">
                          <span className="truncate">{email.sender}</span>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400/60 font-medium">
                            {email.account}
                          </span>
                        </div>
                        {email.preview && (
                          <p className="text-brain-text/40 text-xs leading-relaxed line-clamp-2 ml-5.5 mt-1">
                            {email.preview}
                          </p>
                        )}
                        <div className="text-brain-text/30 text-[10px] ml-5.5 mt-1">
                          {email.date}
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </>
            )}
          </div>
        )}

        {!isSearching && activeTab === "system" && (
          <div className="space-y-2 animate-fade-in">
            {spotlight.length === 0 ? (
              <div className="text-brain-text/50 text-sm text-center py-8">
                No system results found
              </div>
            ) : (
              spotlight.map((item, i) => (
                <div
                  key={`${item.path}-${i}`}
                  className="bg-brain-surface rounded-lg p-3 border border-brain-border hover:border-brain-accent/30 transition-colors cursor-default"
                  onClick={() => navigator.clipboard.writeText(item.path)}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <svg className="w-3.5 h-3.5 text-brain-text/40 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <span className="text-white text-sm font-medium truncate">{item.name}</span>
                    <SystemKindBadge kind={item.kind} />
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-brain-text/50 ml-5.5">
                    <span className="truncate">{item.path}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
        active
          ? "bg-brain-accent/20 text-brain-accent"
          : "text-brain-text/50 hover:text-brain-text hover:bg-brain-surface"
      }`}
    >
      {label}
      {count > 0 && (
        <span className="ml-1.5 text-[10px] opacity-60">{count}</span>
      )}
    </button>
  );
}

function SystemKindBadge({ kind }: { kind: string }) {
  const colors: Record<string, string> = {
    email: "bg-blue-500/20 text-blue-400",
    document: "bg-green-500/20 text-green-400",
    pdf: "bg-red-500/20 text-red-400",
    code: "bg-brain-accent/20 text-brain-accent",
    image: "bg-pink-500/20 text-pink-400",
    spreadsheet: "bg-emerald-500/20 text-emerald-400",
    presentation: "bg-orange-500/20 text-orange-400",
    audio: "bg-violet-500/20 text-violet-400",
    video: "bg-cyan-500/20 text-cyan-400",
    file: "bg-gray-500/20 text-gray-400",
  };

  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${colors[kind] || colors.file}`}>
      {kind}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    Semantic: "bg-blue-500/20 text-blue-400",
    Episodic: "bg-green-500/20 text-green-400",
    Procedural: "bg-yellow-500/20 text-yellow-400",
    Working: "bg-purple-500/20 text-purple-400",
    Meta: "bg-pink-500/20 text-pink-400",
    Causal: "bg-orange-500/20 text-orange-400",
    Goal: "bg-cyan-500/20 text-cyan-400",
    Emotional: "bg-red-500/20 text-red-400",
  };

  return (
    <span
      className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
        colors[type] || "bg-gray-500/20 text-gray-400"
      }`}
    >
      {type}
    </span>
  );
}
