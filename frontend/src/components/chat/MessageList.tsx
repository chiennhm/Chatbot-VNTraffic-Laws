"use client";

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import type { Msg } from "./types";

function ThinkingIndicator() {
    return (
        <div className="flex items-start gap-3 px-1">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white text-sm font-bold shadow-lg">
                AI
            </div>
            <div className="bg-white border border-zinc-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                <div className="flex items-center gap-2 text-zinc-500">
                    <div className="flex gap-1">
                        <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                        <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                    <span className="text-sm font-medium">Đang suy nghĩ...</span>
                </div>
            </div>
        </div>
    );
}

export default function MessageList({ messages, isThinking = false }: { messages: Msg[]; isThinking?: boolean }) {
    const endRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages.length, isThinking]);

    return (
        <div className="px-4 py-4 space-y-3">
            {messages.map((m) => (
                <MessageBubble key={m.id} role={m.role} text={m.text} attachments={m.attachments} />
            ))}
            {isThinking && <ThinkingIndicator />}
            <div ref={endRef} />
        </div>
    );
}
