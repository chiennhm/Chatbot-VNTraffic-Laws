import type { ChatSession, Msg } from "@/components/chat/types";

const KEY = "chat_sessions_v1";

export function loadSessions(): ChatSession[] {
    if (typeof window === "undefined") return [];
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        // Ensure valid structure if migrating (optional safety)
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

export function saveSessions(sessions: ChatSession[]) {
    localStorage.setItem(KEY, JSON.stringify(sessions));
}

export function createNewSession(): ChatSession {
    return {
        id: Math.random().toString(36).slice(2) + Date.now().toString(36),
        title: "New Chat",
        createdAt: Date.now(),
        messages: [],
    };
}
