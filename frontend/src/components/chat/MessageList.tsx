"use client";

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import type { Msg } from "./types";

export default function MessageList({ messages }: { messages: Msg[] }) {
    const endRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages.length]);

    return (
        <div className="px-4 py-4 space-y-3">
            {messages.map((m) => (
                <MessageBubble key={m.id} role={m.role} text={m.text} attachments={m.attachments} />
            ))}
            <div ref={endRef} />
        </div>
    );
}
