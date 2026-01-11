export type UploadedFile = {
    id: string;
    name: string;
    size: number;
    type: string;
    addedAt: number;
};

const KEY = "rag_files_v1";

export function loadFiles(): UploadedFile[] {
    if (typeof window === "undefined") return [];
    try {
        const raw = localStorage.getItem(KEY);
        return raw ? (JSON.parse(raw) as UploadedFile[]) : [];
    } catch {
        return [];
    }
}

export function saveFiles(files: UploadedFile[]) {
    localStorage.setItem(KEY, JSON.stringify(files));
}
