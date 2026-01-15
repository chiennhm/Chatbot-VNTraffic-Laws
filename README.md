# Vietnamese Traffic Law Chatbot 🚗

AI-powered chatbot for Vietnamese traffic law questions using Vision-Language Model (Qwen3-VL-8B) and RAG (Retrieval-Augmented Generation).

## 📁 Project Structure

```
traffic_law/
├── backend/          # FastAPI server
│   ├── main.py       # API endpoints: /chat, /search
│   └── requirements.txt
├── frontend/         # Next.js web UI
│   ├── src/
│   └── package.json
├── llm/              # Vision-Language Model module
│   ├── llm_rag.py    # Qwen3-VL inference
│   └── prompts.json  # Prompt templates
└── rag_law/          # RAG system
    ├── rag/          # Search & upload scripts
    └── processing/   # Document processing
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional, for VLM)
- Qdrant Cloud account

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd traffic_law
```

### 2. Environment Setup

```bash
# Create .env file
cp rag_law/.env.example .env

# Edit with your credentials
nano .env
```

**Required environment variables:**

```env
# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=VN_traffic_regulation_md_hybrid

# Embedding Models
DENSE_MODEL_ID=Savoxism/vietnamese-legal-embedding-finetuned
SPARSE_MODEL_ID=Qdrant/bm25
RERANK_MODEL_ID=namdp-ptit/ViRanker

# Optional: VLM settings
USE_VLM=true
VLM_MODEL_ID=unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
```

### 3. Install Dependencies

```bash
# Python dependencies (from project root)
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 4. Run Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
#or using: uvicorn main:app --host 0.0.0.0 --port 8000
# Server starts at http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# UI available at http://localhost:3000
```

## ☁️ Cloud Deployment

### Deploy to Railway/Render

1. **Backend:**
   - Connect GitHub repo
   - Set build command: `pip install -r backend/requirements.txt -r rag_law/requirements.txt`
   - Set start command: `python backend/main.py`
   - Add environment variables from `.env`

2. **Frontend:**
   - Set build command: `npm install && npm run build`
   - Set start command: `npm start`
   - Set `BACKEND_URL` to your backend URL

### Deploy to Google Cloud Run

```bash
# Build and push backend
gcloud builds submit --tag gcr.io/PROJECT_ID/traffic-chatbot-backend

# Deploy
gcloud run deploy traffic-chatbot-backend \
  --image gcr.io/PROJECT_ID/traffic-chatbot-backend \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "QDRANT_URL=xxx,QDRANT_API_KEY=xxx"
```

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with VLM + RAG |
| `/search` | POST | Direct RAG search |
| `/health` | GET | Health check |

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Mức phạt không đội mũ bảo hiểm?",
    "attachments": []
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

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_HOST` | `0.0.0.0` | Backend host |
| `BACKEND_PORT` | `8000` | Backend port |
| `FRONTEND_URL` | `http://localhost:3000` | CORS allowed origin |
| `USE_VLM` | `true` | Enable Vision-Language Model |
| `VLM_LOAD_4BIT` | `true` | Load VLM in 4-bit quantization |

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CORS errors | Check `FRONTEND_URL` matches your frontend |
| VLM not loading | Ensure CUDA is available, or set `USE_VLM=false` |
| RAG returns empty | Verify Qdrant credentials and collection exists |
| Slow first request | Models loading on first call, subsequent calls faster |

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request
