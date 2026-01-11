"use client";

import { useCallback, useState, useRef } from "react";

interface FileDropzoneProps {
    onPick: (picked: File[]) => void;
    busy?: boolean;
}

export default function FileDropzone({ onPick, busy }: FileDropzoneProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleClick = () => {
        if (!busy && inputRef.current) inputRef.current.click();
    };

    const onDrop = useCallback(
        (e: React.DragEvent<HTMLDivElement>) => {
            e.preventDefault();
            setIsDragging(false);
            if (busy) return;

            const files = Array.from(e.dataTransfer.files);
            if (files.length > 0) onPick(files);
        },
        [onPick, busy]
    );

    const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    return (
        <div
            onClick={handleClick}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            role="button"
            tabIndex={0}
            className={[
                "rounded-2xl border-2 border-dashed p-6 text-center text-sm transition-all duration-200",
                "select-none flex flex-col items-center justify-center gap-2",
                busy ? "cursor-not-allowed opacity-60 bg-zinc-50" : "cursor-pointer",
                isDragging
                    ? "border-purple-500 bg-purple-50 text-purple-700 scale-[1.02]"
                    : "border-indigo-200 bg-indigo-50/50 text-indigo-400 hover:border-purple-400 hover:bg-purple-50 hover:text-purple-600",
            ].join(" ")}
        >
            <div>
                {busy ? (
                    "Processing..."
                ) : (
                    <>
                        <span className="font-semibold text-base block mb-1">Upload File</span>
                        <span className="text-xs opacity-70">
                            Drags & Drop or Click to Browse
                        </span>
                    </>
                )}
            </div>

            <input
                ref={inputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                        onPick(Array.from(e.target.files));
                        e.target.value = "";
                    }
                }}
            />
        </div>
    );
}
