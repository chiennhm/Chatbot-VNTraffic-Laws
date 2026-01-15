"use client";

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import type { Msg } from "./types";

function WelcomeMessage() {
    return (
        <div className="flex flex-col items-center justify-center h-full py-12 px-6 text-center">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white text-2xl font-bold shadow-xl mb-6">
                🚗
            </div>
            <h2 className="text-xl font-bold text-zinc-800 mb-2">
                Hỗ trợ học thi lý thuyết lái xe
            </h2>
            <p className="text-zinc-500 text-sm max-w-md mb-6">
                Giải đáp câu hỏi thi lý thuyết lái xe A1, A2, B1, B2 dựa trên văn bản pháp luật
            </p>

            <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-4 max-w-lg text-left">
                <h3 className="font-semibold text-zinc-700 mb-3 flex items-center gap-2">
                    <span>�</span> Bạn có thể hỏi về:
                </h3>
                <ul className="space-y-2 text-sm text-zinc-600">
                    <li className="flex items-start gap-2">
                        <span className="text-purple-500 mt-0.5">•</span>
                        <span>Câu hỏi thi lý thuyết (gửi hình ảnh câu hỏi hoặc gõ nội dung)</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <span className="text-purple-500 mt-0.5">•</span>
                        <span>Biển báo, sa hình, quy tắc nhường đường</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <span className="text-purple-500 mt-0.5">•</span>
                        <span>Mức phạt vi phạm giao thông theo Nghị định 168/2024</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <span className="text-purple-500 mt-0.5">•</span>
                        <span>Quy định trong Luật Trật tự ATGT 2024, Luật Đường bộ 2024</span>
                    </li>
                </ul>
            </div>

            <p className="text-xs text-zinc-400 mt-4">
                💡 Gửi hình ảnh câu hỏi thi để được giải đáp chi tiết
            </p>
        </div>
    );
}

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

    // Show welcome message when no messages
    if (messages.length === 0 && !isThinking) {
        return <WelcomeMessage />;
    }

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
