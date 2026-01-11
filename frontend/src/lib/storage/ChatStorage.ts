import type { Msg } from "@/components/chat/types";

const KEY = "chat_messages_v1";

export function loadMessages(): Msg[] {
    if (typeof window === "undefined") return [];
    try {
        const raw = localStorage.getItem(KEY);
        return raw ? (JSON.parse(raw) as Msg[]) : [];
    } catch {
        return [];
    }
}

export function saveMessages(msgs: Msg[]) {
    localStorage.setItem(KEY, JSON.stringify(msgs));
}
