import { useRef, useEffect, useState } from "react";
import { useAppStore } from "../store/appStore";

export default function ChatView() {
  const { chatMessages, chatLoading, sendChatMessage, clearChat, exitChatMode } = useAppStore();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatLoading]);

  useEffect(() => {
    // Focus chat input after mount with a short delay to ensure the DOM is ready
    const timer = setTimeout(() => inputRef.current?.focus(), 50);
    return () => clearTimeout(timer);
  }, []);

  const handleSend = async () => {
    if (!input.trim() || chatLoading) return;
    const msg = input.trim();
    setInput("");
    await sendChatMessage(msg);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-brain-border">
        <button
          onClick={exitChatMode}
          className="p-1 rounded text-brain-text/50 hover:text-white hover:bg-brain-surface transition-colors"
          title="Back to search (Esc)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <span className="text-sm text-white font-medium flex-1">Chat</span>
        {chatMessages.length > 0 && (
          <button
            onClick={clearChat}
            className="text-[10px] px-2 py-0.5 rounded text-brain-text/40 hover:text-brain-text/70 hover:bg-brain-surface transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {chatMessages.length === 0 && !chatLoading && (
          <div className="text-brain-text/30 text-sm text-center py-12">
            <div className="mb-2">
              <svg className="w-8 h-8 mx-auto text-brain-text/20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            Ask DeepBrain anything.
            <br />
            <span className="text-brain-text/20">Uses your memories, files, and emails for context.</span>
          </div>
        )}

        {chatMessages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-brain-accent/20 text-white rounded-br-sm"
                  : "bg-brain-surface border border-brain-border text-white rounded-bl-sm"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.role === "assistant" && (msg.confidence != null || msg.ai_enhanced) && (
                <div className="flex items-center gap-3 mt-2 text-[10px] text-brain-text/40">
                  {msg.confidence != null && (
                    <span>{(msg.confidence * 100).toFixed(0)}% confidence</span>
                  )}
                  {msg.memory_count != null && msg.memory_count > 0 && (
                    <span>{msg.memory_count} memories</span>
                  )}
                  {msg.ai_enhanced && (
                    <span className="text-brain-accent">AI</span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {chatLoading && (
          <div className="flex justify-start">
            <div className="bg-brain-surface border border-brain-border rounded-xl rounded-bl-sm px-4 py-2.5 text-sm text-brain-text/50">
              <span className="inline-flex gap-1">
                <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
                <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
                <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
              </span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-brain-border p-3 flex items-center gap-2">
        <input
          ref={inputRef}
          data-chat-input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Stop propagation so global handlers in App.tsx don't interfere
            e.stopPropagation();
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
            if (e.key === "Escape") {
              if (input) {
                setInput("");
              } else {
                exitChatMode();
              }
            }
          }}
          placeholder="Type a message..."
          className="flex-1 bg-brain-surface rounded-lg px-3 py-2 text-sm text-white placeholder-brain-text/40 outline-none border border-brain-border focus:border-brain-accent/50 transition-colors"
          disabled={chatLoading}
          autoFocus
        />
        <button
          onClick={handleSend}
          disabled={chatLoading || !input.trim()}
          className="p-2 rounded-lg bg-brain-accent/20 text-brain-accent hover:bg-brain-accent/30 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Send (Enter)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  );
}
