"use client";

import { cn } from "@/lib/utils/cn";

export default function IconButton({
    title,
    onClick,
    children,
    className,
}: {
    title: string;
    onClick?: () => void;
    children: React.ReactNode;
    className?: string;
}) {
    return (
        <button
            type="button"
            title={title}
            onClick={onClick}
            className={cn(
                "inline-flex h-9 w-9 items-center justify-center rounded-xl border transition active:scale-[0.98]",
                "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-100",
                "dark:border-white/10 dark:bg-white/5 dark:text-white/80 dark:hover:bg-white/10",
                className
            )}
        >
            {children}
        </button>
    );
}
