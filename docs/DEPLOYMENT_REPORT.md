# Atlas AI Financial Assistant - Deployment Report

Date: 2026-08-08 · Environment: production (Render free tier)
Commits: `612a1ce`, `932ba75`, `398c823` (auto-deployed on push to `origin/main`)

## 1. What was deployed

| Area | Detail |
|---|---|
| Repo | `atlas-ai-assistant` (GitHub `suryaexists2/atlas-ai-financial-assistant`) |
| Base URL | `https://atlas-bot-peop.onrender.com/` (`GET /health` -> `{"status":"ok","app":"Atlas","version":"0.1.0","env":"prod"}`) |
| Bot | `@atlasassistant_ai_bot` - natural text + voice only, no commands/menus |
| Storage | PostgreSQL (Supabase neon, asyncpg) prod; SQLite locally |
| STT | `STT_PROVIDER=groq` + free `GROQ_API_KEY` (Render env, not exposed) |

## 2. Bugs fixed and shipped this milestone

1. **Scheduler permanently died on the first stale job** (`scheduling/worker.py`).
   It logged via `logging.getLogger`, but called `logger.warning(..., job_id=...)`;
   the structlog logger rejects extra kwargs with `TypeError`, aborting every
   `sweep_once()`. Fix: `logger = get_logger(__name__)`. Regression test:
   `test_misfired_job_recovers_to_next_boundary`.

2. **Stale `next_run_at` while Render sleeps = recurring misfires.** Jobs were
   perpetually "due" in the past. Fix: `compute_next_run(cron, after=now)` so a job
   always lands on its next future boundary.

3. **Voice STT failed in prod** because OpenRouter now requires a minimum **$0.50
   account balance for any audio** (HTTP 402 on every request). Fix: added
   `GroqSTT` - a free, drop-in provider using Groq's OpenAI-compatible
   (multipart) transcriptions API with no balance gate. 4 new tests
   (multipart request shape, empty result, HTTP error, pipeline drop-in).

## 3. Live regression (prod)

| Item | Result |
|---|---|
| Health | 200 OK |
| Cycle jobs | `price_alerts`, `news_alerts`, `filing_alerts` ran 04:30 / 04:45; `next_run_at` advanced to 05:00; `job_events` recorded |
| Reminder | "⏰ Reminder: review the audit" enqueued 04:26:16, **SENT** 04:26:22; job auto-disabled (`enabled=false`) after fire |
| E2E (text) | research, quote, watchlist, market news, NVDA alert, reminder, PDF doc - all acked; document PROCESSED + reply SENT |
| **Voice (new)** | **4 real voice notes, all transcribed + replied** (see below) |

### Voice-STT end-to-end proof (05:31-05:32 UTC)

| Voice | Lang | Doc status | Transcript persisted | Agent reply (SENT) |
|---|---|---|---|---|
| `voice_en` | EN | PROCESSED | "What's today's market price and the biggest news for NVIDIA?" | current NVDA price $218.99 (-0.10%), high/low/open |
| `voice_hi` | Hinglish | PROCESSED | "इन्वीडिया का शेयर प्राइस क्या है और आज की टॉप निउस क्या है?" | NVDA $218.99 high/low open reply |
| `voice_hi_pure` | Hindi | PROCESSED | "मुझे Nvidia के latest earnings का summary दो।" | NVDA earnings: est 1.7922 / actual 1.87 (beat) |
| `voice_meta` | EN | PROCESSED | "Give me a quick overview of Meta and the latest earnings." | Meta company overview reply |

Chain: Telegram voice update -> download (fresh bot-owned file_id) -> Groq Whisper
-> transcript persisted as `[voice transcript]...` message + `documents.status =
PROCESSED` -> Atlas agent (LLM) -> outbox reply -> Telegram SENT. English and
Hindi/Hinglish both verified; transcripts correct in Devanagari for the Indic
clips; old 402-failure rows (04:42/04:47) remain only as historical evidence and
now succeed after the Groq switch.

## 4. Regression gate

- `pytest`: **193 tests, exit 0** (asyncio, UoW, queries, provider mocks)
- `ruff` clean (`check` + `format`)

## 5. Voice / STT details

- OpenRouter audio gate confirmed live: HTTP `402` `{"code":402,"message":"This
  request requires at least $0.50 in balance for audio"}`; applies to every
  audio request on the account (any model / chat-with-audio). No workaround on
  that provider; top-up or free alternative required.
- Free path: set `STT_PROVIDER=groq` with a free `GROQ_API_KEY`
  (https://console.groq.com). `GroqSTT` posts multipart to
  `https://api.groq.com/openai/v1/audio/transcriptions`.
- If `STT_PROVIDER` is unset the app keeps OpenRouter as the default; vision
  always uses OpenRouter when its key is present.

## 6. Remaining items

- [x] Voice live-verified (EN + Hindi/Hinglish) 2026-08-07 05:31-05:32 UTC
- [ ] Watch the 08:00 UTC daily-brief run (`scope=both`) for the interest-section prose
- Optional Gmail / Calendar / Drive: deliberately deferred (see matrix note)

## 7. Security note

No API keys or secrets are printed, committed, or logged. Env secret pointers
used for behavior confirmation only. Voice transcripts in DB contain no secret
data.