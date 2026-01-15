"use client";

import { useEffect, useState } from "react";
import ChatShell, { type LLMProvider } from "@/components/chat/ChatShell";
import Sidebar from "@/components/layout/Sidebar";
import { type ChatSession, type Msg } from "@/components/chat/types";
import { cn } from "@/lib/utils/cn";
import {
  loadSessions,
  saveSessions,
  createNewSession,
} from "@/lib/storage/ChatSessionStorage";

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

export default function HomePage() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [llmProvider, setLlmProvider] = useState<LLMProvider>("gemini");

  // Load sessions on mount
  useEffect(() => {
    const loaded = loadSessions();
    if (loaded.length > 0) {
      setSessions(loaded);
      setActiveId(loaded[0].id);
    } else {
      // Create initial session if none exist
      const newSession = createNewSession();
      setSessions([newSession]);
      setActiveId(newSession.id);
    }
  }, []);

  // Save sessions whenever they change
  useEffect(() => {
    if (sessions.length > 0) {
      saveSessions(sessions);
    }
  }, [sessions]);

  const activeSession = sessions.find((s) => s.id === activeId);

  const handleCreateSession = () => {
    const newSession = createNewSession();
    setSessions((prev) => [newSession, ...prev]);
    setActiveId(newSession.id);
  };

  const handleDeleteSession = (id: string) => {
    const newSessions = sessions.filter((s) => s.id !== id);
    setSessions(newSessions);

    // If active session is deleted, switch to another
    if (activeId === id) {
      setActiveId(newSessions[0]?.id || null);
    }

    // If all deleted, create new one
    if (newSessions.length === 0) {
      const fresh = createNewSession();
      setSessions([fresh]);
      setActiveId(fresh.id);
    }
  };

  const handleSendMsg = async (text: string, attachments: string[]) => {
    if (!activeId) return;

    const userMsg: Msg = {
      id: uid(),
      role: "user",
      text,
      attachments,
      createdAt: Date.now(),
    };

    // Optimistic update: Add user message immediately
    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeId
          ? { ...s, messages: [...s.messages, userMsg] }
          : s
      )
    );

    try {
      // Call the API with provider
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          attachments,
          provider: llmProvider,
          history: activeSession?.messages || [],
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response");
      }

      const data = await response.json();

      const botMsg: Msg = {
        id: uid(),
        role: "assistant",
        text: data.text || "Sorry, I didn't get a response.",
        createdAt: Date.now(),
      };

      // Add bot response
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeId
            ? { ...s, messages: [...s.messages, botMsg] }
            : s
        )
      );
    } catch (error) {
      console.error("Chat error:", error);
      // Optional: Add an error message to the chat
      const errorMsg: Msg = {
        id: uid(),
        role: "assistant", // or 'system' if supported
        text: "Sorry, something went wrong. Please try again.",
        createdAt: Date.now(),
      };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeId
            ? { ...s, messages: [...s.messages, errorMsg] }
            : s
        )
      );
    }
  };

  const handleRenameSession = (id: string, newTitle: string) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, title: newTitle } : s));
  };

  /* Sidebar Toggle State */
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <main
      className="
        h-screen p-4 md:p-6 flex flex-col
        bg-transparent
      "
    >
      <div className="mx-auto w-full max-w-[1400px] flex-1 h-full min-h-0 relative group/layout">
        <div
          className={cn(
            "grid h-full gap-4 transition-all duration-300 ease-in-out",
            isSidebarOpen ? "md:grid-cols-[320px_1fr]" : "md:grid-cols-[0px_1fr]"
          )}
        >
          {/* SIDEBAR (Chats + Files) */}
          <div className={cn(
            "h-full min-h-0 overflow-hidden transition-all duration-300",
            isSidebarOpen ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4"
          )}>
            <Sidebar
              sessions={sessions}
              activeSessionId={activeId}
              onSelectSession={setActiveId}
              onCreateSession={handleCreateSession}
              onDeleteSession={handleDeleteSession}
              onRenameSession={handleRenameSession}
              onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
            />
          </div>

          {/* CHAT AREA */}
          <div className="h-full min-h-0 relative flex flex-col">
            {activeSession ? (
              <ChatShell
                messages={activeSession.messages}
                onSendMsg={handleSendMsg}
                onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
                isSidebarOpen={isSidebarOpen}
                llmProvider={llmProvider}
                onProviderChange={setLlmProvider}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-zinc-500">
                Select or create a chat
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
