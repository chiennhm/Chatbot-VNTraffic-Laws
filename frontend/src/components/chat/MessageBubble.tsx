import { cn } from "@/lib/utils/cn";
import type { Role } from "./types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function MessageBubble(props: { role: Role; text: string; attachments?: string[] }) {
    const isUser = props.role === "user";

    return (
        <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
            <div
                className={cn(
                    "max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                    isUser
                        ? "bg-gradient-to-r from-violet-600 to-cyan-500 text-white shadow-md shadow-purple-500/20"
                        : "border border-zinc-100 bg-white text-zinc-800 shadow-sm"
                )}
            >
                {/* Attachments Grid */}
                {props.attachments && props.attachments.length > 0 && (
                    <div className={cn("grid gap-2 mb-2", props.attachments.length > 1 ? "grid-cols-2" : "grid-cols-1")}>
                        {props.attachments.map((att: string, i: number) => {
                            if (att.startsWith("data:video")) {
                                return (
                                    <video
                                        key={i}
                                        src={att}
                                        controls
                                        className="rounded-lg w-full h-auto max-h-[300px] bg-black"
                                    />
                                );
                            }
                            return (
                                <img
                                    key={i}
                                    src={att}
                                    alt="attachment"
                                    className="rounded-lg object-cover w-full h-auto max-h-[300px]"
                                />
                            );
                        })}
                    </div>
                )}
                {isUser ? (
                    <span className="whitespace-pre-wrap">{props.text}</span>
                ) : (
                    <div className="prose prose-sm prose-zinc max-w-none prose-p:my-1 prose-headings:mt-3 prose-headings:mb-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:bg-zinc-100 prose-pre:text-zinc-800 prose-code:bg-zinc-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-zinc-800 prose-code:before:content-none prose-code:after:content-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {props.text}
                        </ReactMarkdown>
                    </div>
                )}
            </div>
        </div>
    );
}
