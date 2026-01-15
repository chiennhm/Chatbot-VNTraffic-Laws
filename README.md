# Vietnamese Traffic Law Chatbot 🚗

AI-powered chatbot for Vietnamese traffic law questions using **Google Gemini** or **Local VLM (Qwen3-VL-8B)** with RAG (Retrieval-Augmented Generation).

## ✨ Features

- **Dual LLM Support**: Switch between Gemini API (fast, cloud) and Local VLM (private, GPU)
- **Hybrid RAG Search**: Dense + Sparse vectors with Re-ranking
- **Vietnamese Legal Focus**: Trained on Vietnamese traffic law documents
- **Vision Support**: Upload images for traffic sign recognition (Local VLM only)
- **Modern UI**: Next.js frontend with real-time chat

## 📁 Project Structure

```
traffic_law/
├── backend/          # FastAPI server
│   └── main.py       # API endpoints: /chat, /search
├── frontend/         # Next.js web UI
│   └── src/
├── llm/              # LLM integration module
│   ├── llm_rag.py    # Gemini + Local VLM support
│   └── prompts.json  # Prompt templates
└── rag_law/          # RAG system
    ├── rag/          # Search & upload scripts
    └── processing/   # Document processing
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional, for Local VLM)
- Qdrant Cloud account
- Google Gemini API key (for Gemini provider)

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd traffic_law
```

### 2. Environment Setup

```bash
# Create .env file
cp .env.example .env

# Edit with your credentials
nano .env
```

### 3. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### 4. Run Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
# Server starts at http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# UI available at http://localhost:3000
```

## ⚙️ Configuration

### LLM Providers

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `gemini` | Google Gemini API (default) | `GEMINI_API_KEY` |
| `local` | Qwen3-VL-8B on local GPU | CUDA GPU, unsloth |

**Switch provider in `.env`:**
```env
LLM_PROVIDER=gemini  # or "local"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider: "gemini" or "local" |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `USE_VLM` | `true` | Enable VLM for image support |
| `QDRANT_URL` | - | Qdrant Cloud URL |
| `QDRANT_API_KEY` | - | Qdrant API key |

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with LLM + RAG |
| `/search` | POST | Direct RAG search |
| `/health` | GET | Health check |

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Mức phạt không đội mũ bảo hiểm?",
    "provider": "gemini"
  }'
```

**Response:**
```json
{
  "text": "Theo Nghị định 168/2024...",
  "query": "mức phạt không đội mũ bảo hiểm xe máy",
  "sources": [{"document_id": "ND168", "article": "Điều 7"}]
}
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Gemini API error | Check `GEMINI_API_KEY` is valid |
| Local VLM not loading | Ensure CUDA is available, or use `LLM_PROVIDER=gemini` |
| RAG returns empty | Verify Qdrant credentials and collection exists |
| CORS errors | Check `FRONTEND_URL` matches your frontend |

## 📄 License

MIT License
