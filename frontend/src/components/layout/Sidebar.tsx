"use client";

import { useState } from "react";
import FilePanel from "@/components/upload/FilePanel";
import ChatList from "@/components/chat/ChatList";
import { cn } from "@/lib/utils/cn";
import { FolderOpen, MessageSquareText, PanelLeftClose } from "lucide-react";
import type { ChatSession } from "@/components/chat/types";

type View = "chats" | "files";

export default function Sidebar({
    sessions,
    activeSessionId,
    onSelectSession,
    onCreateSession,
    onDeleteSession,
    onRenameSession,
    onToggleSidebar,
}: {
    sessions: ChatSession[];
    activeSessionId: string | null;
    onSelectSession: (id: string) => void;
    onCreateSession: () => void;
    onDeleteSession: (id: string) => void;
    onRenameSession: (id: string, newTitle: string) => void;
    onToggleSidebar?: () => void;
}) {
    const [view, setView] = useState<View>("chats");

    return (
        <aside
            className="
            flex h-full flex-col rounded-3xl
            bg-white border border-zinc-200/80
            shadow-2xl shadow-purple-500/10
            overflow-hidden relative group
            "
        >
            {/* Header Tabs */}
            <div className="flex border-b border-zinc-100 p-1 relative pr-10">
                {onToggleSidebar && (
                    <button
                        onClick={onToggleSidebar}
                        className="
                            absolute top-1/2 -translate-y-1/2 right-2 z-10
                            p-1.5 rounded-lg text-zinc-400 
                            hover:text-zinc-600 hover:bg-zinc-100 
                            transition-colors
                        "
                        title="Close Sidebar"
                    >
                        <PanelLeftClose className="w-5 h-5" />
                    </button>
                )}
                <button
                    onClick={() => setView("chats")}
                    className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl text-sm font-semibold transition-all",
                        view === "chats"
                            ? "bg-gradient-to-r from-violet-50 to-cyan-50 text-violet-700"
                            : "text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50"
                    )}
                >
                    <MessageSquareText className={cn("w-4 h-4", view === "chats" && "text-violet-600")} />
                    Chats
                </button>
                <button
                    onClick={() => setView("files")}
                    className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl text-sm font-semibold transition-all",
                        view === "files"
                            ? "bg-gradient-to-r from-violet-50 to-cyan-50 text-violet-700"
                            : "text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50"
                    )}
                >
                    <FolderOpen className={cn("w-4 h-4", view === "files" && "text-violet-600")} />
                    Files
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 p-4 overflow-hidden relative">
                {view === "chats" ? (
                    <ChatList
                        sessions={sessions}
                        activeId={activeSessionId}
                        onSelect={onSelectSession}
                        onCreate={onCreateSession}
                        onDelete={(id, e) => {
                            e.stopPropagation();
                            onDeleteSession(id);
                        }}
                        onRename={onRenameSession}
                    />
                ) : (
                    <div className="h-full">
                        {/* 
                           FilePanel was designed as a card. 
                           Since we are inside a Sidebar card already, 
                           we might want to strip its outer container styling or refactor it.
                           For now, we'll render it directly, but we should probably refactor FilePanel 
                           to be 'headless' or clear its styles if passed a prop.
                           
                           To avoid heavy refactoring now, we can wrap it or just accept nested cards.
                           However, FilePanel has its own 'bg-white shadow' etc.
                           Let's trust the stacking for now and maybe refactor later.
                        */}
                        <FilePanel />
                    </div>
                )}
            </div>
        </aside>
    );
}
