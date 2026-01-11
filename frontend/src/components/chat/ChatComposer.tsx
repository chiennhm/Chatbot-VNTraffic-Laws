"use client";

import { useState, useRef } from "react";
import { ImagePlus, X } from "lucide-react";

export default function ChatComposer({
    onSend,
    disabled,
}: {
    onSend: (text: string, attachments: string[]) => void;
    disabled?: boolean;
}) {
    const [value, setValue] = useState("");
    const [attachments, setAttachments] = useState<string[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [isDragging, setIsDragging] = useState(false);

    const processFiles = (files: FileList | File[]) => {
        Array.from(files).forEach((file) => {
            if (!file.type.startsWith("image/") && !file.type.startsWith("video/")) return;

            // Limit file size to 5MB to avoid localStorage quota issues
            if (file.size > 5 * 1024 * 1024) {
                alert("File is too large (max 5MB). Storage is local-only.");
                return;
            }

            const reader = new FileReader();
            reader.onload = (ev) => {
                if (ev.target?.result) {
                    setAttachments((prev) => [...prev, ev.target!.result as string]);
                }
            };
            reader.readAsDataURL(file);
        });
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            processFiles(e.target.files);
        }
        // Reset input so same file can be selected again if needed
        if (e.target.value) e.target.value = "";
    };

    const handlePaste = (e: React.ClipboardEvent) => {
        if (e.clipboardData.files.length > 0) {
            processFiles(e.clipboardData.files);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files.length > 0) {
            processFiles(e.dataTransfer.files);
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const removeAttachment = (index: number) => {
        setAttachments((prev) => prev.filter((_, i) => i !== index));
    };

    const submit = () => {
        const text = value.trim();
        if (!text && attachments.length === 0) return;

        onSend(text, attachments);
        setValue("");
        setAttachments([]);
    };

    return (
        <div
            className="relative border-t border-zinc-100 p-3 bg-white"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
        >
            {/* Drag Overlay */}
            {isDragging && (
                <div className="absolute inset-0 z-50 rounded-xl m-2 border-2 border-dashed border-purple-500 bg-purple-50/90 flex items-center justify-center text-purple-700 font-semibold pointer-events-none">
                    Drop images or videos here to upload
                </div>
            )}

            {/* Attachments Previews */}
            {attachments.length > 0 && (
                <div className="flex gap-2 mb-2 overflow-x-auto pb-2">
                    {attachments.map((att, idx) => (
                        <div key={idx} className="relative group flex-shrink-0">
                            {att.startsWith("data:video") ? (
                                <video
                                    src={att}
                                    className="h-16 w-16 object-cover rounded-xl border border-zinc-200 bg-black"
                                />
                            ) : (
                                <img
                                    src={att}
                                    alt="preview"
                                    className="h-16 w-16 object-cover rounded-xl border border-zinc-200"
                                />
                            )}
                            <button
                                onClick={() => removeAttachment(idx)}
                                className="absolute -top-1.5 -right-1.5 bg-red-500 text-white rounded-full p-0.5 shadow-sm opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                                <X className="w-3 h-3" />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            <div className="flex gap-2 items-center">
                <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept="image/*,video/*"
                    multiple
                    onChange={handleFileSelect}
                />

                <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={disabled}
                    className="h-[44px] w-[44px] flex items-center justify-center rounded-2xl text-zinc-400 hover:text-purple-600 hover:bg-purple-50 transition-colors"
                    title="Add Image or Video"
                >
                    <ImagePlus className="w-5 h-5" />
                </button>

                <div className="flex-1 relative min-w-0">
                    <textarea
                        value={value}
                        disabled={disabled}
                        onChange={(e) => setValue(e.target.value)}
                        onPaste={handlePaste}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                submit();
                            }
                        }}
                        placeholder="Type your message..."
                        className="w-full resize-none rounded-2xl border border-zinc-200 bg-zinc-50 pl-4 pr-10 py-3 text-sm text-zinc-900 outline-none placeholder:text-zinc-400 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all max-h-28 min-h-[44px] break-all whitespace-pre-wrap overflow-hidden"
                        rows={1}
                        style={{ height: 'auto', minHeight: '44px' }}
                        onInput={(e) => {
                            const target = e.target as HTMLTextAreaElement;
                            target.style.height = 'auto';
                            const newHeight = Math.min(target.scrollHeight, 112); // max-h-28 is 112px
                            target.style.height = `${newHeight}px`;

                            // Only show scrollbar if content exceeds max height
                            if (target.scrollHeight > 112) {
                                target.style.overflowY = "auto";
                            } else {
                                target.style.overflowY = "hidden";
                            }
                        }}
                    />
                </div>

                <button
                    onClick={submit}
                    disabled={disabled || (!value.trim() && attachments.length === 0)}
                    className="rounded-2xl px-5 py-2.5 text-sm font-bold text-white
                     bg-gradient-to-r from-violet-600 to-cyan-500
                     shadow-lg shadow-purple-500/20
                     hover:shadow-purple-500/40 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-60 transition-all h-[44px] flex items-center justify-center min-w-[80px]"
                >
                    Send
                </button>
            </div>
        </div>
    );
}
