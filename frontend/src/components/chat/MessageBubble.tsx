import { cn } from "@/lib/utils/cn";
import type { Role } from "./types";

export default function MessageBubble(props: { role: Role; text: string; attachments?: string[] }) {
    const isUser = props.role === "user";

    return (
        <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
            <div
                className={cn(
                    "max-w-[82%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed",
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
                {props.text}
            </div>
        </div>
    );
}
