"use client";

import { PanelLeft } from "lucide-react";

export default function ChatHeader({
    onToggleSidebar,
    isSidebarOpen,
}: {
    onToggleSidebar?: () => void;
    isSidebarOpen?: boolean;
}) {
    return (
        <div className="flex items-center gap-3 border-b border-zinc-100 bg-white px-5 py-4">
            {onToggleSidebar && !isSidebarOpen && (
                <button
                    onClick={onToggleSidebar}
                    className="p-2 -ml-2 rounded-lg text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 transition-colors"
                    title={isSidebarOpen ? "Close Sidebar" : "Open Sidebar"}
                >
                    <PanelLeft className="w-5 h-5" />
                </button>
            )}
            <div>
                <div className="text-xs font-bold tracking-wider text-purple-600 uppercase mb-0.5">Q&A Assistant</div>
                <div className="text-lg font-bold text-zinc-900 flex items-center gap-2">
                    Chat
                </div>
            </div>
        </div>
    );
}
