# Atlas - AI financial intelligence inside Telegram

Atlas is a conversational, proactive financial assistant built for the
**Atlas AI Financial Assistant Hackathon**. It lives inside Telegram, reads
documents and voice notes, answers questions with live market and SEC data,
and quietly handles the morning briefing, alerts, and reminders - all through
natural conversation, not commands.

## What it does

- **Conversational by design** - no slash commands or menus; just chat in
  Telegram.
- **Multi-turn conversation** - keeps recent history per conversation and a
  tool-driven long-term memory (preferences, risk tolerance, goals).
- **Onboarding** - first-time users get a guided setup (role, interests,
  watchlist, briefing time, reminder time) before the agent takes over.
- **Document intelligence** - parses PDF, DOCX, XLSX, CSV, TXT, MD, and JSON;
  transcribes voice notes (Whisper via OpenRouter) and OCRs images (vision
  model). Extracted text is stored so the user can ask follow-up questions.
- **Live market data** - quotes, profiles, market/company news and earnings via
  Finnhub; SEC filings (10-K / 10-Q / 8-K) via a small EDGAR client.
- **Proactive jobs** - a DB-backed scheduler composes a personal morning
  briefing, evaluates price/news/filing alerts with cooldowns, and fires
  one-off reminders - all delivered through a durable Telegram outbox.
- **Google Sheets** - link a public spreadsheet and Atlas can read its rows
  (no OAuth needed; works when the sheet is shared as "anyone with the link").
- **Reliable delivery** - outbox table with exponential backoff, retries,
  rate limiting, and idempotent re-delivery.
- **Operable** - structured JSON logs, correlation IDs, and a `/health` route.

## Layout

```
interfaces/     Telegram (HTTP webhook + processor), responder, HTTP routes
application/    agent core, onboarding, ingestion pipeline, intelligence jobs
domain/         entities, enums, repository contract ports
infrastructure/ DB (SQLAlchemy async), providers (Finnhub, SEC, Google Sheets),
                 Telegram API + outbox worker, durable scheduler
```

Layers talk through ports in `domain/repositories.py` and a `UnitOfWork` that
owns a single async session, so every agent side effect goes through the DB and
is retryable by the workers. Tests run against in-memory SQLite.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"
```

`cp .env.example .env`, then set `TELEGRAM_BOT_TOKEN`, `OPENROUTER_API_KEY`,
`FINNHUB_API_KEY`, and `SEC_USER_AGENT`, and switch `ECHO_MODE=false` to enable
the real agent.

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

Register the webhook or use the long-poll script:

```bash
python scripts/set_webhook.py     # or: python scripts/run_polling.py
```

## Tests and lint

```bash
pytest -q --no-cov     # the suite (184 tests)
ruff check app tests
```

## Environment

Every setting is documented in `.env.example`. Minimal production set:

- `DATABASE_URL` - Postgres via asyncpg in prod; SQLite locally.
- `TELEGRAM_BOT_TOKEN` (+ `TELEGRAM_WEBHOOK_SECRET`, `PUBLIC_BASE_URL`).
- `OPENROUTER_API_KEY` - agent, voice STT, and image OCR.
- `FINNHUB_API_KEY` - quotes, news, earnings.
- `SEC_USER_AGENT` - identify EDGAR requests.

## Deployment

The ASGI app is `app.main:app`. On startup it starts the Telegram update
processor, the outbox sender worker, and the proactive scheduler, and seeds the
global alert-monitoring jobs. Health probe: `/health`.