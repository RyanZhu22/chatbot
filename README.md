# Internal Chatbot MVP

Node.js + React.js internal chatbot frontend and API gateway.

## Features

- ChatGPT style web UI (React + Vite)
- Node.js chat API (`/api/chat`)
- Real LLM call via OpenAI-compatible API
- Streaming response with pause support
- History messages are sent to backend for multi-turn context
- Assistant responses are rendered as Markdown in the UI
- Local knowledge base retrieval (RAG) with source citations

## Setup

1. Copy env template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

2. Fill `.env`:

- `LLM_PROVIDER`: `openai` or `deepseek` (default `openai`)
- `OPENAI_API_KEY`: required when provider is `openai`
- `OPENAI_BASE_URL`: optional, default `https://api.openai.com/v1`
- `OPENAI_MODEL`: optional, default `gpt-4o-mini`
- `DEEPSEEK_API_KEY`: required when provider is `deepseek`
- `DEEPSEEK_BASE_URL`: optional, default `https://api.deepseek.com`
- `DEEPSEEK_MODEL`: optional, default `deepseek-chat`
- `RAG_ENABLED`: `true`/`false`, default `true`
- `RAG_KNOWLEDGE_DIR`: local knowledge folder, default `knowledge`
- `RAG_TOP_K`: max retrieved chunks, default `4`
- `RAG_MIN_SCORE`: retrieval score threshold, default `0.8`
- `RAG_CHUNK_SIZE`: chunk size for indexing, default `800`
- `RAG_CONTEXT_MAX_CHARS`: max chars injected to model context, default `4000`
- `RAG_BM25_K1`: BM25 parameter, default `1.2`
- `RAG_BM25_B`: BM25 parameter, default `0.75`
- `RAG_CANDIDATE_MULTIPLIER`: candidate pool size factor before rerank, default `8`
- `RAG_MMR_LAMBDA`: diversity vs relevance tradeoff, default `0.78`
- `RAG_NGRAM_SIZE`: char n-gram size for fuzzy matching, default `3`
- `SYSTEM_PROMPT`: optional

3. Install and run:

```bash
npm install
npm run dev
```

Frontend: `http://localhost:5173`  
Backend: `http://localhost:3001`

## DeepSeek Example

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_real_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## API

- `GET /api/health`
- `POST /api/chat`

## Knowledge Base (RAG)

Put your internal `.md` / `.txt` files under `knowledge/` (or `RAG_KNOWLEDGE_DIR`), for example:

- `knowledge/hr_policy.md`
- `knowledge/engineering_guidelines.md`

The backend will index these files and retrieve relevant chunks for each user question.  
Retriever mode is `hybrid-bm25-mmr` (BM25 + phrase match + char n-gram + MMR rerank).  
When a chunk is used, the UI shows source citations under the assistant message.

Request body:

```json
{
  "provider": "openai",
  "message": "hello",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```
