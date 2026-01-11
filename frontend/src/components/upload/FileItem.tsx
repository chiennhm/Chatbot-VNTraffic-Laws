import type { UploadedFile } from "@/lib/storage/fileStorage";
import { formatBytes } from "@/lib/utils/format";

export default function FileItem({
    file,
    onRemove,
}: {
    file: UploadedFile;
    onRemove: () => void;
}) {
    return (
        <div
            className="
        group flex items-center justify-between gap-3 rounded-2xl px-3 py-2.5
        border border-zinc-100 bg-white
        shadow-sm shadow-purple-500/5
        hover:border-purple-200 hover:bg-purple-50/30 hover:shadow-purple-500/10
        transition-all duration-200
      "
        >
            <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-zinc-800">
                    {file.name}
                </div>
                <div className="mt-0.5 text-xs text-zinc-500">
                    {formatBytes(file.size)} • {new Date(file.addedAt).toLocaleString()}
                </div>
            </div>

            <button
                onClick={onRemove}
                className="
          rounded-xl px-2.5 py-1 text-xs font-semibold
          bg-zinc-50 text-zinc-500
          hover:bg-red-50 hover:text-red-600
          transition-colors
        "
            >
                Xoá
            </button>
        </div>
    );
}
