"use client";

import { Plus, MessageSquare, Trash2, Pencil, Check, X } from "lucide-react";
import type { ChatSession } from "./types";
import { cn } from "@/lib/utils/cn";
import { useState } from "react";

export default function ChatList({
    sessions,
    activeId,
    onSelect,
    onCreate,
    onDelete,
    onRename,
}: {
    sessions: ChatSession[];
    activeId: string | null;
    onSelect: (id: string) => void;
    onCreate: () => void;
    onDelete: (id: string, e: React.MouseEvent) => void;
    onRename: (id: string, newTitle: string) => void;
}) {
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editValue, setEditValue] = useState("");

    const startEditing = (s: ChatSession, e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingId(s.id);
        setEditValue(s.title);
    };

    const saveEdit = (e?: React.FormEvent) => {
        e?.preventDefault();
        if (editingId && editValue.trim()) {
            onRename(editingId, editValue.trim());
        }
        setEditingId(null);
    };

    const cancelEdit = (e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingId(null);
    };

    return (
        <div className="flex flex-col h-full">
            <button
                onClick={onCreate}
                className="
                    flex items-center gap-2 rounded-2xl px-5 py-3.5 mb-4 mx-2
                    bg-gradient-to-r from-violet-600 to-cyan-500
                    text-white font-bold text-sm shadow-lg shadow-purple-500/25
                    hover:shadow-purple-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all
                "
            >
                <Plus className="w-5 h-5" />
                <span>New Question</span>
            </button>

            <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 px-2">
                {sessions.length === 0 && (
                    <div className="text-center text-sm text-zinc-400 mt-8">
                        No conversations yet.
                    </div>
                )}
                {sessions.map((s) => (
                    <div
                        key={s.id}
                        onClick={() => onSelect(s.id)}
                        className={cn(
                            "group flex items-center justify-between gap-3 rounded-2xl p-3 cursor-pointer transition-all border",
                            activeId === s.id
                                ? "bg-white border-purple-100 shadow-sm shadow-purple-100"
                                : "bg-transparent border-transparent hover:bg-white hover:shadow-sm"
                        )}
                    >
                        <div className="flex items-center gap-3 min-w-0 flex-1">
                            <div className={cn(
                                "flex items-center justify-center w-8 h-8 rounded-full",
                                activeId === s.id ? "bg-purple-100" : "bg-zinc-100 group-hover:bg-white"
                            )}>
                                <MessageSquare className={cn(
                                    "w-4 h-4",
                                    activeId === s.id ? "text-purple-600" : "text-zinc-400"
                                )} />
                            </div>

                            {editingId === s.id ? (
                                <form onSubmit={saveEdit} className="flex-1 flex items-center gap-1" onClick={e => e.stopPropagation()}>
                                    <input
                                        autoFocus
                                        value={editValue}
                                        onChange={(e) => setEditValue(e.target.value)}
                                        onBlur={() => saveEdit()}
                                        onKeyDown={(e) => {
                                            if (e.key === "Escape") setEditingId(null);
                                        }}
                                        className="w-full bg-transparent text-sm font-semibold text-zinc-800 outline-none border-b-2 border-purple-500 px-0 py-0.5"
                                    />
                                </form>
                            ) : (
                                <div className="truncate text-sm font-medium text-zinc-600 group-hover:text-zinc-900 transition-colors">
                                    {s.title || "Untitled Chat"}
                                </div>
                            )}
                        </div>

                        {editingId === s.id ? (
                            <button
                                onClick={cancelEdit}
                                className="p-1.5 rounded-full text-zinc-400 hover:text-red-500 hover:bg-red-50 transition-all"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        ) : (
                            <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity gap-1">
                                <button
                                    onClick={(e) => startEditing(s, e)}
                                    className="p-1.5 rounded-full text-zinc-400 hover:bg-purple-50 hover:text-purple-600 transition-all"
                                >
                                    <Pencil className="w-3.5 h-3.5" />
                                </button>
                                <button
                                    onClick={(e) => onDelete(s.id, e)}
                                    className="p-1.5 rounded-full text-zinc-400 hover:bg-red-50 hover:text-red-500 transition-all"
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
