"use client";

import { useMemo, useState } from "react";
import ChatHeader from "./ChatHeader";
import MessageList from "./MessageList";
import ChatComposer from "./ChatComposer";
import type { Msg } from "./types";

function uid() {
    return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

export default function ChatShell({
    messages,
    onSendMsg,
    onToggleSidebar,
    isSidebarOpen,
}: {
    messages: Msg[];
    onSendMsg: (text: string, attachments: string[]) => Promise<void>;
    onToggleSidebar?: () => void;
    isSidebarOpen?: boolean;
}) {
    // We can keep 'thinking' state local for UI effect, 
    // or lift it if we want global loading state. 
    // For now, local is fine as it acts per shell view.
    const [thinking, setThinking] = useState(false);

    const handleSend = async (text: string, attachments: string[]) => {
        setThinking(true);
        try {
            await onSendMsg(text, attachments);
        } finally {
            setThinking(false);
        }
    };

    const footer = useMemo(
        () => (
            <div className="px-5 pb-4 text-xs font-medium text-zinc-500 flex items-center gap-2">
                <span className="bg-zinc-100 px-1.5 py-0.5 rounded text-zinc-600 border border-zinc-200">Tip</span>
                <span>Enter to send • Shift+Enter for new line</span>
            </div>
        ),
        []
    );

    return (
        <div
            className="
            flex flex-col h-full
            rounded-3xl border border-zinc-200/80 bg-white
            shadow-2xl shadow-purple-500/10
            overflow-hidden
        "
        >
            <ChatHeader onToggleSidebar={onToggleSidebar} isSidebarOpen={isSidebarOpen} />
            <div className="flex-1 overflow-hidden flex flex-col bg-zinc-50/50">
                {/* 
                    MessageList usually handles its own scroll. 
                    Ensure it takes available height.
                 */}
                <div className="flex-1 overflow-y-auto">
                    <MessageList messages={messages} />
                </div>
            </div>
            <ChatComposer onSend={handleSend} disabled={thinking} />
            {footer}
        </div>
    );
}
