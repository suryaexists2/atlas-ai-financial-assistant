# Atlas AI Financial Assistant — Official Hackathon Specification (Snapshot)

> Read-only working reference. Source:
> https://humanity-founders.notion.site/atlas-ai-financial-assistant-hackathon
> Captured 2026-08-07. This file is the single source of truth for scope decisions.
> If the live page changes materially, refresh this snapshot and re-run the requirement matrix.

## Objective

Build an AI-powered Financial Assistant that helps finance professionals stay informed,
conduct research, prepare for meetings, and make better decisions through natural
conversations inside Telegram.

The assistant should understand context, remember previous conversations, provide
proactive intelligence, and help users quickly find the information that matters most.

### Product principles

- Every interaction should reduce manual work.
- Responses concise and immediately useful.
- Prioritize quality over feature count.
- Surface information that actually matters instead of summarizing everything.
- Feel conversational rather than command-driven.
- Product thinking and UX matter more than dozens of integrations.

### What to avoid

- A chatbot that simply answers questions.
- A news reader with AI summaries.
- Long reports that require excessive scrolling.
- A command-based Telegram bot.
- Menus, slash commands, inline buttons, quick replies, command navigation.

## Onboarding

- Welcome, natural, conversational — no lengthy forms or complicated flows.
- Gradually understand the user by asking a few simple questions.
- Focus: role, interests, watchlist, monitoring preferences, insight types, briefing
  schedule, custom alerts.
- Optional: offer to connect accounts (Gmail, Google Calendar, Google Drive, Google
  Sheets) conversationally; always skippable; connect later supported.
- Optional extra verticals (investing, startups, business, tech, healthcare…) only after
  finance-first experience is well developed. Finance always the primary vertical.
- User must always be able to skip any onboarding question and start using immediately.
- Keep learning through future conversations (preferences, watchlists, workflow,
  notification schedule, connected data, recurring tasks).

## 2. Proactive intelligence

Primary responsibility: proactively deliver meaningful financial intelligence.

- Examples: morning market brief, evening summary, earnings updates, company news,
  regulatory announcements, economic events, breaking news, personalized watchlist updates.
- Explain WHY something matters, not just forward headlines.
- If nothing important, stay silent — quality beats frequency.

## 3. Conversational/experience requirements

- Natural communication, like speaking to an experienced analyst or executive assistant.
- Examples of natural prompts the assistant must handle:
  - "What are the biggest market-moving events I should know about today?"
  - "Compare Microsoft and Google from an investment perspective."
  - "Summarize Apple's latest earnings call in five key points."
  - "Analyze this annual report and highlight the biggest risks."
  - "Compare these two companies based on revenue growth, profitability, and valuation."
  - "Explain why Nvidia's stock moved today."
  - "Track Tesla and notify me whenever there's a major announcement or SEC filing."
  - "Create a daily morning briefing covering AI, semiconductor, and technology stocks."
  - "Schedule a meeting with my team tomorrow to discuss this earnings report."
  - "Remind me one hour before Apple's earnings call."
  - "Summarize this Google Sheet and identify unusual trends."
  - "Search my emails and summarize conversations related to this company."
  - "Find latest news about this acquisition and explain potential market impact."
  - "Create an alert if this stock moves more than 5% in a day."
  - "Compare today's market performance with yesterday's and explain biggest changes."
- When a request is ambiguous, ask ONE good follow-up before assuming.
- Maintain conversational context across interactions; use preferences/memories.
- Users communicate using only: Text, Voice Messages, Images. (No slash commands etc.)

## 4. Financial research

- Research public and private companies; recent developments; help faster decisions.
- Examples: company profiles, business overview, financial performance, earnings
  summaries, recent news, leadership changes, funding activity, M&A, regulatory
  filings, market sentiment, industry trends, competitor comparisons.
- Explain why information matters; provide useful context.

## 5. Document understanding

- Users upload documents and ask questions naturally.
- Examples of docs: annual reports, quarterly reports, earnings presentations, investment
  decks, financial statements, SEC filings, due diligence docs, research reports.
- Capabilities: summarize, explain financial performance, compare reports, extract key
  insights, answer questions from uploads, highlight important changes, executive summaries.
- Conversational, not document-driven flows.

## 6. Live information retrieval

- Retrieve live info when local knowledge is insufficient; synthesize into concise answers.
- Examples: stock prices, company news, earnings releases, economic events, SEC filings,
  market performance, industry updates, analyst activity.
- Accuracy extremely important; use reliable sources; don't present unverified info as
  fact; communicate uncertainty.

## 7. Integrations

- Gmail / Google Calendar potential: email centering, meeting preparation, follow-up
  reminders, action items, company context. (OPTIONAL)
- Google Drive / Google Sheets (financial docs and spreadsheets): analyze spreadsheets,
  explain financials, review KPIs, compare forecasts, detect anomalies, search documents. (OPTIONAL)
- Financial data providers: SEC EDGAR, FinnHub, Financial Modeling Prep, Polygon,
  Alpha Vantage, Yahoo Finance, government APIs. Apply company fundamentals, prices,
  earnings schedules, SEC filings, insider moves, ratios, economic indicators, news.
- More integrations welcome only if they truly improve the experience.

## 8. Personalization

- Learn over time via conversation; examples: companies you follow, preferred industries,
  topics, frequently asked questions, briefing schedule, reading preferences,
  conversation history, research interests, watchlists.

## 9. Engineering

- Practical, useful; enhancements allowed once core works E2E. Creativity encouraged so
  long as core stays intuitive/conversational.
- Tech stack: any modern backend (Node/Express/NestJS, Python/FastAPI). Recommendation:
  Telegram bot dev, natural conversational experience, AI integration, backend API,
  database integration, auth (where required), 3rd-party integrations, background jobs /
  scheduled tasks, clean structure, reusable components, well-organized codebase, Git.
- Database: MongoDB / PostgreSQL / MySQL / SQLite. Support: user profiles, conversation
  history, user preferences, connected integrations, financial documents,
  personalization, assistant memory, app-specific data. Maintainable, scalable design.
- AI foundation: any LLM (OpenAI/Gemini/Claude/Llama/Qwen/Mistral/DeepSeek/Gemma…).
  Model choice not a criterion; quality of conversation, reasoning, reliability,
  personalization, UX matter more. Capabilities: natural conversations, context
  awareness, summarization, company research, financial reasoning, personalized
  responses, document understanding, multi-source synthesis.
- Interface: Telegram primary; natural conversation; text/voice/images only.
- Submission: record a demo; submit a LIVE bot (judges interact with it); source code
  NOT required. Telegram group attached.

## Judging criteria (weighted)

| Criterion | Weight |
|---|---|
| Usefulness, proactivity, overall user value | 30% |
| Product thinking, judgment, thoughtful feature selection | 25% |
| AI experience and conversational quality | 20% |
| Depth of the finance vertical | 15% |
| Engineering quality and implementation | 10% |

## Prizes / success framing

- Winner: most functional, thoughtful, user-friendly assistant → Founding Engineer role.
- Certificates awarded on functional MVP quality.
- Emphasis: building what "lies there" — watches, learns, acts — "I understand how you
  work — let me do it for you."

---
*End of snapshot. Do not modify this file during normal development; update only if the
live specification changes.*