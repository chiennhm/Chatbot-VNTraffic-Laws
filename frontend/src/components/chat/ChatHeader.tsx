"use client";

import { PanelLeft, Sparkles, Cpu } from "lucide-react";

type LLMProvider = "gemini" | "local";

export default function ChatHeader({
    onToggleSidebar,
    isSidebarOpen,
    llmProvider = "gemini",
    onProviderChange,
}: {
    onToggleSidebar?: () => void;
    isSidebarOpen?: boolean;
    llmProvider?: LLMProvider;
    onProviderChange?: (provider: LLMProvider) => void;
}) {
    const isGemini = llmProvider === "gemini";

    return (
        <div className="flex items-center justify-between border-b border-zinc-100 bg-white px-5 py-4">
            <div className="flex items-center gap-3">
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

            {/* LLM Provider Toggle */}
            {onProviderChange && (
                <div className="flex items-center gap-2">
                    <span className="text-xs text-zinc-500 hidden sm:inline">Model:</span>
                    <div className="flex bg-zinc-100 rounded-lg p-1">
                        <button
                            onClick={() => onProviderChange("gemini")}
                            className={`
                                flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all
                                ${isGemini
                                    ? "bg-white text-purple-600 shadow-sm"
                                    : "text-zinc-500 hover:text-zinc-700"
                                }
                            `}
                            title="Gemini API - Fast, cloud-based"
                        >
                            <Sparkles className="w-4 h-4" />
                            <span className="hidden sm:inline">Gemini</span>
                        </button>
                        <button
                            onClick={() => onProviderChange("local")}
                            className={`
                                flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all
                                ${!isGemini
                                    ? "bg-white text-emerald-600 shadow-sm"
                                    : "text-zinc-500 hover:text-zinc-700"
                                }
                            `}
                            title="Local VLM - Privacy-focused, runs on your GPU"
                        >
                            <Cpu className="w-4 h-4" />
                            <span className="hidden sm:inline">Local</span>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
