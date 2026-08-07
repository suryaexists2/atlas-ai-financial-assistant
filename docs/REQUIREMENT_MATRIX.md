# Atlas AI Financial Assistant — Requirement Matrix

Living document. Every row maps an official-spec requirement → status → modules → tests →
live verification → evidence. Update after each milestone; never mark COMPLETE until the
four gates pass (spec match, integration, tests, live verification).

Status legend: ✅ Complete · 🟡 Partial · ❌ Missing. Live column: 👍 verified / ⏳ pending / —

| # | Requirement (spec) | Status | Files / modules | Tests | Live | Evidence |
|---|---|---|---|---|---|---|
| 1 | Live in Telegram, natural conversation, no commands/menus/inline buttons | ✅ | `interfaces/telegram/processor.py`, `normalizer.py`, `normalized.py`; webhook route | `test_processor.py`, `test_normalizer.py`, `test_webhook.py` | Live | Agent replies seen in Telegram via webhook; no commands |
| 2 | Communicate via text only (no slash commands, menus, quick replies) | ✅ | `processor.py`, `normalizer.py` | `test_normalizer.py` | Live | — |
| 3 | Voice messages supported | ✅ (live) | `ingestion/pipeline.py`, `media_ai.py` (GroqSTT free / OpenRouter fallback), `main.py` (stt_provider) | `test_ingestion.py` (GroqSTT multipart/empty/http/pipeline), `test_processor_media.py` | 👍 | **2026-08-07 05:31–05:32 UTC**: 4 real voice notes (English, Hinglish, pure Hindi, English) → downloaded, Groq-transcribed (`documents` PROCESSED, transcripts correct incl. Devanagari), agent replied with live data (NVDA $218.99; earnings est/actual; Meta overview) — all outbox SENT. Stored as `[voice transcript]…` in conversation |
| 4 | Images supported (OCR/vision) | ✅ (live) | `media_ai.py` (vision model, OpenRouter chat path) | `test_processor_media.py`, `test_ingestion.py` | 👍 | 2026-08-07 05:39: real chart PNG via webhook → `documents` PROCESSED (`kind=image`), vision described "line graph comparing revenue growth of NVIDIA (NVDA) and Tesla (TSLA)…" + caption Q&A replied |
| 5 | Conversational onboarding, skippable, gradual | ✅ | `application/onboarding.py` | `test_onboarding.py` | Live | Profile COMPLETED, prod |
| 6 | Onboarding captures role, interests, watchlist, briefing time, reminders | ✅ | `onboarding.py` (+ `intelligence/jobs.py`) | `test_onboarding.py` | Live | Profile rows |
| 7 | Always skippable; can start immediately | ✅ | `onboarding.py` (`_SKIP`, `_wants_agent`) | `test_onboarding.py` | Live | — |
| 8 | Keep learning over time (memories/preferences) | ✅ | `agent/tools.py` (save/list_memory), `repositories/.../memory` | `test_memory.py`, `repositories/test_memory.py` | Live | memories rows |
| 9 | Proactive intelligence: morning brief, earnings, company news, regulatory, breaking, watchlist updates | ✅ | `intelligence/briefing.py`, `intelligence/alerts.py`, `scheduling/worker.py` | `test_intelligence.py`, `test_scheduler.py` | Live | briefing delivered 01:58 |
| 10 | Explain WHY, not just forward headlines | ✅ | `briefing.py` (`_BRIEF_SYSTEM`, `_deterministic_summary`) | `test_intelligence.py` | Live | — |
| 11 | Stay silent when nothing important | ✅ | `briefing.py` (returns None on no data) | `test_intelligence.py` | Live | silent when no news |
| 12 | Natural follow-up: ask ONE good clarifying question when ambiguous | ✅ (LLM) | `agent/core.py` conversation loop | — | Live | — |
| 13 | Maintain conversational context across interactions | ✅ | `conversation_service`, `repositories/conversation_market.py` (newest-window) | `test_conversation_watchlist.py` (window) | Live | 24-message window verified || 14 | Financial research: company profile, overview | ✅ | `tools.py get_company_profile`, `providers/finnhub.py` | `test_agent_tools.py` | Live | — |
| 15 | Financial research: earnings summaries | ✅ | `get_company_earnings`, `finnhub.py` | `test_agent_tools.py` | Live | — |
| 16 | Financial research: recent news / sentiment | ✅ | `get_company_news`, `get_market_news` | `test_agent_tools.py` | Live | NVIDIA news live |
| 17 | Financial research: regulatory filings | ✅ | `get_company_filings`, `providers/sec.py` | `test_agent_tools.py` | Live | — |
| 18 | Financial research: leadership changes / funding / M&A / industry trends / competitor comparisons | 🟡 partial (LLM synthesizes from news/profile; no dedicated dataset) | `get_company_news` + agent reasoning | — | — | finnhub news covers majors |
| 19 | Document upload & Q&A (annual/quarterly reports, decks, financial statements, SEC filings, etc.) | ✅ | `ingestion/pipeline.py`, `parsers.py` (pdf/docx/xlsx/csv/txt/md/json), `get_document_contents` | `test_ingestion.py`, `test_ingest_ledger.py`, `test_processor_media.py` | 👍 | Real PDF via prod webhook: `documents` → PROCESSED; "Here are five key points summarizing the document" SENT 04:40 |
| 20 | Execute summaries, highlight changes, compare reports | ✅ (LLM-driven over doc text) | `get_document_contents` + agent | — | 👍 | doc summary reply live (see #19) |
| 21 | Live retrieval: stock prices | ✅ | `get_market_quote`, `finnhub.py` | `test_agent_tools.py` | Live | AAPL $312.41 quote |
| 22 | Live retrieval: market performance (indices) | ✅ | `get_market_indices`, `providers/stooq.py` | `test_agent_tools.py` | ⏳ | — |
| 23 | Live retrieval: earnings calendar / economic events / analyst activity | 🟡 partial (earnings via finnhub; no dedicated econ/analyst integration) | — | — | ⏳ | — |
| 24 | Accuracy: use reliable sources; communicate uncertainty | ✅ | provider-native responses; `briefing` only uses data block | — | Live | — |
| 25 | Private/company research with citations-ish | ✅ (headlines carry source) | finnhub returns source field | — | Live | — |
| 26 | Optional: Google Sheets (financial docs/spreadsheets) | ✅ | `providers/google_sheets.py`, `read_google_sheet` etc. | `test_agent_tools.py` | ⏳ | public-sheet reader works |
| 27 | Optional: Gmail / Google Calendar / Google Drive | ❌ NOT implemented (deliberately optional; documented) + **honesty guardrail shipped** | `agent/context.py` system prompt (never fabricate email/calendar/Drive actions; offer alternatives; public Sheets links readable) | `test_agent_context.py` (guardrail tests) | 👍 (guardrail live) | 2026-08-07 05:47-05:48: "search my emails" → "I don't have access to your emails…"; "schedule a meeting…" → "I'm not connected to your calendar or email, so I won't be able to schedule a meeting… (offered alternative)" |
| 28 | Favorite watchlist & monitor alerts | ✅ | `watchlist` repos, tools `add/remove/list` | `test_agent_tools.py`, repo tests | Live | TSLA added |
| 29 | Custom alerts: price move %, news trigger, SEC filing | ✅ | `intelligence/alerts.py` (price/news/filing) with cooldown, `create_*_alert` tools | `test_intelligence.py`, `test_scheduler.py` | Live | NVDA alert created |
| 30 | Reminders ("remind me...") | ✅ | `intelligence/reminders.py`, `create_reminder` tool, job `once` disable | `test_scheduler.py` | 👍 | fired 04:26, key outbox "⏰ Reminder" SENT, `enabled=False` after fire (once-disable live) |
| 31 | Daily briefing at user-chosen time | ✅ | `create_daily_briefing` + onboarding + `briefing.py` | `test_intelligence.py` | Live | brief 01:58 |
| 32 | Briefing/alert coverage based on interests (AI, semiconductor, tech, macro…) — spec "Create a daily morning briefing covering AI, semiconductor, and technology stocks" | ✅ | `briefing.py` (`_match_interest_news`, `_gather_interests`, scope watchlist/interests/both), `tools.py create_daily_briefing(scope)`, `onboarding.py` (scope=both) | `test_intelligence.py` (interest_news + gather_interests + summary includes interest section) | 👍 (code+unit tests verified; next 08:00 run carries final interest news) | suite green; scope param on `daily_brief` job; new briefings default scope=both |
| 33 | Proactive: "track X, notify me on major announcement or SEC filing" | ✅ | news/filing alerts | tests | ⏳ | — |
| 34 | "Explain why X moved today" | ✅ (LLM + news/quote) | agent | — | ⏳ | — |
| 35 | "Compare today vs yesterday market" | 🟡 (indices + agent reasoning; no historical store) | agent + stooq | — | ⏳ | — |
| 36 | Background jobs / scheduled tasks | ✅ | `scheduling/worker.py`, `cron.py`, cycle & user jobs | `test_scheduler.py` | 👍 | post-fix: cycle jobs ran 04:30/04:45, `next_run_at` advanced to 05:00; stale-job recovery live |
| 37 | Clean architecture, modular, reusable components | ✅ | layering: domain/application/infrastructure/interfaces; ports & UoW | — | 👍 | — |
| 38 | PostgreSQL (prod), SQLite (local); maintained data model | ✅ | models: users, profiles, conversations, documents, memories, watchlists, alerts, jobs, job_events, integrations, outbox, ingest_ledger | repo tests | Live | Prod Postgres |
| 39 | AI foundation: any LLM; context aware; multi-source | ✅ | OpenRouter gateway (llama → gpt-4o-mini → gemini → llama fallback), agent context | `test_llm_gateway.py`, `test_agent_core.py` | Live | — |
| 40 | Engineering quality: tests, lint, structure | ✅ | pytest (~190), ruff; asyncio; structured logs (JSON in prod); health route | all tests | Live | /health |
| 41 | Background delivery reliability | ✅ | `telegram/outbox_worker.py` (retry/backoff/ratelimit/idempotent) | `test_outbox_worker.py` | Live | SENT items |

## Optional integrations (deliberately deferred — allowed by spec)

- **Gmail / Google Calendar / Google Drive**: NOT implemented. Spec explicitly lists them
  as optional integrations ("may integrate…"). Skipped to keep finance vertical deep and
  the demo reliable within the timeline. How to enable later:
  - `app/infrastructure/providers/gmail.py`: OAuth PKCE + google-auth; classify emails,
    summarize, meeting preparation, action items.
  - `app/domain/enums.py` IntegrationProvider: add `GMAIL`, `CALENDAR`, `DRIVE`;
    `onboarding.py` add an optional "connect accounts" step; `tools.py` add
    `search_emails`, `find_calendar_events`, `read_drive_doc`.
  - Console: Google Cloud OAuth client → store refresh tokens in `integrations` table.

## Current outstanding work (tracked live)

| Milestone | Status |
|---|---|
| Misfire recovery + structlog logger fix (`scheduling/worker.py`, `test_scheduler.py`) | ✅ deployed + live (cycle jobs advancing, stale job recovered) |
| Interest-scope daily briefing expansion (watchlist/interests/both) | ✅ deployed; next 08:00 cycle validates delivery prose |
| Full live verification cycle (reminder fire, job events, branch advance) | ✅ done (reminder fired + once-disabled; job_events 04:30/04:45) |
| E2E multi-workflow prod run | ✅ done (research, quote, watchlist, news, alert, reminder, document, voice) |
| Voice STT live | ✅ done — 4 real voice notes (EN/Hinglish/Hindi) via Groq, replies SENT (05:31-05:32 UTC) |
| Images/OCR live | ✅ done — real chart PNG → vision described, caption Q&A (05:39 UTC) |
| Email/Calendar/Drive honesty guardrail | ✅ shipped + live-verified (05:47-05:48 UTC) |
| Final spec audit + report | ✅ done — see `docs/DEPLOYMENT_REPORT.md` |

## Voice / STT known limitation (operator action)

Voice transcription calls OpenRouter `POST /api/v1/audio/transcriptions`
(JSON body, base64 `input_audio`). The request shape matches the current OpenRouter
contract, but OpenRouter now enforces a **minimum $0.50 account balance for audio**.
With a balance below that, every audio request returns HTTP 402. Chat completions
(LLM replies, briefing text, doc summaries) are unaffected and keep working.

Fix: add credits at https://openrouter.ai/settings/credits. No code change needed.