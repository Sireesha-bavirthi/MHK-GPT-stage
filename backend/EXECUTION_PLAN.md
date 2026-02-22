# MHK-GPT v2 — Execution Plan

> **Root issue (Python env):** All packages are installed under **Anaconda Python 3.13**.
> Always use `python` / `uvicorn` from that env, **not** `/opt/homebrew/bin/uvicorn`.

---

## Step 0 — One-time env check

```bash
which python      # must be /opt/homebrew/anaconda3/bin/python
which uvicorn     # must be /opt/homebrew/anaconda3/bin/uvicorn
python --version  # 3.13.x
```

If `uvicorn` points to `/opt/homebrew/bin/uvicorn`, use the full path:
```bash
/opt/homebrew/anaconda3/bin/uvicorn app.main:app --reload
```

---

## Step 1 — Install Redis

```bash
# Install via Docker (easiest — no brew install needed)
docker run -d --name mhk_redis -p 6379:6379 redis:7-alpine

# Verify it's running
docker ps | grep mhk_redis
```

Or keep using the existing brew install if you have it:
```bash
brew install redis && brew services start redis
```

---

## Step 2 — Install Python dependencies

```bash
cd /Users/shalinitata/MHK-GPT/backend

# Use conda Python explicitly
/opt/homebrew/anaconda3/bin/pip install -r requirements.txt
```

Verify key packages are installed:
```bash
/opt/homebrew/anaconda3/bin/python -c "import fastapi, langgraph, slowapi, redis, sendgrid; print('All OK')"
```

---

## Step 3 — Set up environment variables

```bash
cd /Users/shalinitata/MHK-GPT/backend
cp .env.example .env
```

Edit `.env` — fill in at minimum:
```env
OPENAI_API_KEY=sk-your-key-here

# Email for OTP (one of these two)
SMTP_PASSWORD=your-16-char-gmail-app-password   # Gmail: Account → Security → App Passwords
# OR
SENDGRID_API_KEY=SG.your-sendgrid-key

# JobDiva (already pre-filled in .env.example)
JOBDIVA_USERNAME=hr@mhktechinc.com
JOBDIVA_PASSWORD=Houston@77070

# Redis (if using Docker from Step 1)
REDIS_URL=redis://localhost:6379
```

---

## Step 4 — Ingest documents into Qdrant

> Only needed first time, or when documents change.

```bash
cd /Users/shalinitata/MHK-GPT/backend

# Make sure Qdrant is running (local Docker)
docker run -d --name mhk_qdrant -p 6333:6333 qdrant/qdrant:latest

# Ingest
/opt/homebrew/anaconda3/bin/python -m app.services.document_processor.ingest
```

---

## Step 5 — Start the backend

```bash
cd /Users/shalinitata/MHK-GPT/backend

# Must use conda's uvicorn
/opt/homebrew/anaconda3/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected startup output:
```
============================================================
  MHK-GPT v2 — Agentic AI System starting up
============================================================
  LLM model    : gpt-4o-mini
  Redis        : redis://localhost:6379
  LangGraph agent compiled ✓
```

Verify it's live:
```bash
curl http://localhost:8000/
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/analytics/health
```

---

## Step 6 — Start the frontend

```bash
cd /Users/shalinitata/MHK-GPT/frontend

# First time only
npm install

# Start dev server
npm run dev
```

Frontend runs at: **http://localhost:3000**

> **Important:** Make sure the frontend's API base URL points to `http://localhost:8000/api/v1/agent/chat`.
> Check `frontend/src/` for any `API_URL` or `NEXT_PUBLIC_API_URL` env var and set it in `frontend/.env.local`:
> ```env
> NEXT_PUBLIC_API_URL=http://localhost:8000
> ```

---

## Step 7 — Quick smoke tests

```bash
# Test agent chat (job search)
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What job openings are available?", "conversation_id": "test-1"}'

# Test meeting scheduler
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to schedule a meeting", "conversation_id": "meet-1"}'

# Test general QA (RAG)
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "When was MHK Tech founded?", "conversation_id": "qa-1"}'

# Check cost log
curl http://localhost:8000/api/v1/analytics/cost
```

---

## Step 8 — Run unit tests

```bash
cd /Users/shalinitata/MHK-GPT/backend
/opt/homebrew/anaconda3/bin/python -m pytest tests/unit/ -v
# Expected: 28 passed
```

---

## All In One (Docker Compose)

```bash
cd /Users/shalinitata/MHK-GPT/infrastructure/docker
docker-compose up --build
# Starts: backend + Redis + Qdrant together
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `No module named 'slowapi'` | Run with `/opt/homebrew/anaconda3/bin/uvicorn`, not `uvicorn` |
| `No module named 'pydantic'` | Run with `/opt/homebrew/anaconda3/bin/python`, not `python3.13` from Homebrew |
| `Redis connection refused` | Start Redis: `docker run -d -p 6379:6379 redis:7-alpine` |
| `Qdrant connection refused` | Start Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant` |
| `OpenAI API key invalid` | Set `OPENAI_API_KEY` in `backend/.env` |
| `JobDiva auth failed` | Check `JOBDIVA_USERNAME` / `JOBDIVA_PASSWORD` in `.env` |
