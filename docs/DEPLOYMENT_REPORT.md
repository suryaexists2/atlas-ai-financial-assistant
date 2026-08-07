# Atlas AI Financial Assistant - Deployment Report

Date: 2026-08-07 · Environment: production (Render free tier)
Commits (live build `1da95a9`): `612a1ce`, `932ba75`, `398c823`, `28dce9b`,
`5506381`, `2f94a67`, `91cbf01`, `cf8f934`, `b014668`, `1da95a9` (auto-deployed on push)

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
| Health | 200 OK (`{"status":"ok","app":"Atlas","version":"0.1.0","env":"prod","build":"<commit>"}`) |
| Cycle jobs | `price_alerts`, `news_alerts`, `filing_alerts` ran 04:30 / 04:45; `next_run_at` advanced to 05:00; `job_events` recorded |
| Reminder | "⏰ Reminder: review the audit" enqueued 04:26:16, **SENT** 04:26:22; job auto-disabled (`enabled=false`) after fire |
| E2E (text) | research, quote, watchlist, market news, NVDA alert, reminder, PDF doc - all acked; document PROCESSED + reply SENT |
| **Voice** | **4 real voice notes, all transcribed + replied** (see below) |
| **Gmail connector** | OAuth PKCE live; prod `search_emails` returned real inbox matches (T3) |
| **Google Calendar** | `create_event` 200 + HTTP DELETE 204 (write verified); bot meeting scheduled end-to-end |
| **Google Drive** | `read_drive_doc` parsed a real 357 KB `inbound…pdf` (3963 chars) and summarized via webhook |

### Google-connector end-to-end proof (14:44-14:53 UTC)

OAuth consent published to Production (external, 100-tester cap); `integration_links`
rows for GMAIL/CALENDAR/DRIVE with access+refresh tokens and requested scopes.

- Meeting scheduling through the bot (fixed flow): "The meeting with
  `suryatextnow@gmail.com` has been scheduled for tomorrow at 10:30 am for 30 minutes
  and is titled 'earnings review'."
- Voice reminder: "Remind me to review the NVIDIA earnings call" → reminder set.
- Excel chat: `earnings_model.xlsx` walkthrough (Revenue 95,000 / +6.7%, Gross 52% / +3pts,
  EPS 1.64 / +8.6%).

### Root cause fixed for the meeting fallback

The Google Calendar create path inside the agent always failed with "Google API
error" while identical direct calls succeeded. Two fixes shipped this milestone:

1. `91cbf01` — When an agent turn exhausts its tool-call budget without a final
   answer, give it one last **tools-free** pass so the user still receives a real
   reply (the meeting reply, not the "temporary hiccup" fallback).
2. `b014668` — the model emitted `schedule_meeting.attendees` in Google's native
   shape (`[{"email": …}]`); the handler stringified the dicts and Google returned
   **400 "Invalid attendee email"**. Attendees are now normalized (dict → email value)
   before the request body is built. Regression test:
   `test_schedule_meeting_normalizes_object_attendees`.
3. `cf8f934` — Calendar API errors now surface status + short body in the tool
   result so diagnostics are no longer opaque.

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
- [x] Google connectors live (Gmail / Calendar / Drive) 2026-08-07 14:44-14:53 UTC
- [x] Meeting scheduling fixed and re-verified (attendee normalization, build `b014668`)
- [ ] Watch the 08:00 UTC daily-brief run (`scope=both`) for the interest-section prose

## 7. Security note

No API keys or secrets are printed, committed, or logged. Env secret pointers
used for behavior confirmation only. Voice transcripts in DB contain no secret
data. **Operator action (after the hackathon event): regenerate the Google OAuth
client secret** — it was shared in plain chat during troubleshooting and mirrored
in local scratch scripts; update the Render env after rotation.