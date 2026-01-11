"use client";

import { useEffect, useMemo, useState } from "react";
import FileDropzone from "./FileDropzone";
import FileItem from "./FileItem";
import { loadFiles, saveFiles, type UploadedFile } from "@/lib/storage/fileStorage";
import Separator from "@/components/ui/seperator";

function uid() {
    return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

export default function FilePanel() {
    const [files, setFiles] = useState<UploadedFile[]>([]);
    const [busy, setBusy] = useState(false);
    const count = useMemo(() => files.length, [files]);

    useEffect(() => {
        setFiles(loadFiles());
    }, []);

    useEffect(() => {
        saveFiles(files);
    }, [files]);

    const onPick = async (picked: File[]) => {
        if (!picked.length) return;
        setBusy(true);

        // Frontend only: just store metadata (name/size/type)
        const newOnes: UploadedFile[] = picked.map((f) => ({
            id: uid(),
            name: f.name,
            size: f.size,
            type: f.type || "unknown",
            addedAt: Date.now(),
        }));

        // small delay to feel smooth
        await new Promise((r) => setTimeout(r, 350));

        setFiles((prev) => [...newOnes, ...prev]);
        setBusy(false);
    };

    return (
        <div
            className="flex h-full flex-col gap-3"
        >
            <div className="flex items-center justify-between px-1 mb-2">
                <div>
                    <div className="text-xs font-bold tracking-wider text-purple-600 uppercase">
                        Knowledge Base
                    </div>
                    <div className="text-xl font-bold text-zinc-800">
                        Files
                    </div>
                </div>
                <div className="text-xs font-semibold px-2 py-1 rounded-md bg-purple-50 text-purple-600">
                    {count}
                </div>
            </div>

            <FileDropzone onPick={onPick} busy={busy} />

            <Separator />

            <div className="flex-1 space-y-2 overflow-y-auto pr-1">
                {files.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 p-4 text-center text-sm text-zinc-500 italic">
                        No files uploaded yet.
                    </div>
                ) : (
                    files.map((f) => (
                        <FileItem
                            key={f.id}
                            file={f}
                            onRemove={() => setFiles((prev) => prev.filter((x) => x.id !== f.id))}
                        />
                    ))
                )}
            </div>
        </div>
    );

}
