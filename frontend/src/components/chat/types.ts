export type Role = "user" | "assistant";

export type Msg = {
    id: string;
    role: Role;
    text: string;
    attachments?: string[]; // base64 strings (images or videos)
    createdAt: number;
};

export type ChatSession = {
    id: string;
    title: string;
    createdAt: number;
    messages: Msg[];
};
