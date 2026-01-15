import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: Request) {
    try {
        const body = await request.json();

        // Proxy to Python backend with provider
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: body.text,
                attachments: body.attachments,
                history: body.history,
                provider: body.provider || "gemini",
            }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || "Backend request failed");
        }

        const data = await response.json();

        return NextResponse.json({
            text: data.text,
            query: data.query,
            sources: data.sources,
        });
    } catch (error) {
        console.error("Chat API error:", error);
        return NextResponse.json(
            { error: error instanceof Error ? error.message : "Internal Server Error" },
            { status: 500 }
        );
    }
}
