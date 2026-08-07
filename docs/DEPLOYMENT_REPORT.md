# Atlas AI Financial Assistant â€” Deployment Report

Date: 2026-08-07 Â· Environment: production (Render, Singapore, free tier)
Commit: `612a1ce` "Fix scheduler misfire death, add interest-scope briefings, spec docs"

## 1. What was deployed

| Area | Detail |
|---|---|
| Repo | `atlas-ai-assistant` (Push to `origin/main`, auto-deployed on Render) |
| Base URL | `https://atlas-bot-peop.onrender.com/` (health `GET /health` â†’ `{"status":"ok","app":"Atlas","version":"0.1.0","env":"prod"}`) |
| Bot | `atlasassistant_ai_bot` â€” no slash commands, no menus |
| Storage | PostgreSQL (Supabase pooler), SQLite locally |

## 2. Two production bugs found & fixed this deployment

1. **Scheduler permanently died on the first stale job**
   - `scheduling/worker.py` logged via `logging.getLogger` but called `logger.warning(..., job_id=...)`; the structlog-configured logger rejected the extra kwargs with `TypeError`, which aborted every `sweep_once()`. As soon as one job fell behind, the scheduler crashed forever and missed every scheduled task.
   - Fix: `logger = get_logger(__name__)` from `app.core.logging`.
   - Regression test: `test_misfired_job_recovers_to_next_boundary` (stale job lands on a real future boundary instead of staying in the past forever).

2. Stale `next_run_at` during sleep = permanent misfires
   - While Render's free tier put the bot to sleep, jobs were "due" in the past (`next_run > now`). Prior code kept the past timestamp and re-fired it forever. Now `compute_next_run(cron, after=now)` guarantees the job always lands on its next real future boundary.

## 3. Live verification results (all against the deployed prod)

| Item | Result |
|---|---|
| Health | 200 `{"status":"ok",...}` |
| Cycle jobs | `price_alerts` (`*/15`), `news_alerts` / `filing_alerts` (`*/30`) woke at 04:30 / 04:45 â€” `job_events` recorded, `next_run_at` advanced to 05:00 |
| Reminder | "â° Reminder: review the audit" â†’ enqueued 04:26:16, **SENT** 04:26:22; job `enabled=false` after fire (`once`-disable works) |
| Outbox | 53 items SENT; outbox worker (retry/backoff/rate-limit, idempotent) healthy |
| E2E (8 workflows) | research â†’ company profile (Microsoft); benchmark quote Meta; watchlist add+list; market news; NVDA price alert; reminder; PDF doc â†’ **5 key points** reply + `documents` PROCESSED; voice â†’ graceful fallback |
| Document | file_id (real PDF); `documents.status = PROCESSED`, `doc_meta.kind=document`, reply 04:40:02 |

## 4. Regression gate at deploy time

- `pytest`: **189 tests, exit 0** (asyncio, UoW, queries)
- `ruff` clean (`check` + `format`)

## 5. Voice / STT â€” solved with a FREE path (no OpenRouter credit needed)

- The OpenRouter STT request (`POST https://openrouter.ai/api/v1/audio/transcriptions`,
  JSON `input_audio` base64) matches OpenRouter's documented contract, but OpenRouter now
  enforces a **minimum $0.50 balance for any audio request** â€” below that every call returns:
  `{"error":{"message":"This request requires at least $0.50 in balance for audio","code":402,...}}`
  (confirmed live 2026-08-07 against our exact payload). Chat/LLM paths are unaffected.
- **Free fix (shipped in this deployment):** a `GroqSTT` provider that calls Groq's
  OpenAI-compatible, free-tier Whisper endpoint (`https://api.groq.com/openai/v1/audio/transcriptions`,
  multipart `file`), which has **no balance gate**. Configure in Render env:
  - `STT_PROVIDER=groq`
  - `GROQ_API_KEY=<free key from https://console.groq.com>`
  - (optional) `GROQ_STT_MODEL=whisper-large-v3-turbo`
- If `STT_PROVIDER` is unset, OpenRouter remains the default (vision keeps using OpenRouter
  whenever its key is present). All four new tests cover the Groq adapter
  (`test_groq_stt_sends_openai_style_multipart`, empty, HTTP error, pipeline drop-in).

## 6. Remaining items

- [ ] Create a free key at https://console.groq.com â†’ set `STT_PROVIDER=groq` +
      `GROQ_API_KEY` on Render â†’ send one voice message â†’ confirm `documents.status=PROCESSED`
      and the transcript text in the reply.
- [ ] Watch the 08:00 UTC daily briefing run (now `scope=both`) to capture the interest-section prose live.
- Optional integrations (Gmail/Calendar/Drive) deliberately deferred; documented in `docs/REQUIREMENT_MATRIX.md`.