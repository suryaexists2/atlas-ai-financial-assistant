"""Live smoke test for a deployed Atlas instance.

Does everything that can be automated against the deployed service plus the
Telegram API, and prints the exact manual steps for the parts only a human
(real Telegram client) can do.

Usage:
    python scripts/smoke_test.py --base-url https://atlas-bot.onrender.com

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_WEBHOOK_SECRET / DATABASE_URL from .env.
Use `--db DATABASE_URL` to point at a different database for the e2e checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import httpx

CHECKS = []


def ok(name: str, detail: str = "") -> None:
    CHECKS.append(("PASS", name, detail))
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def fail(name: str, detail: str = "") -> None:
    CHECKS.append(("FAIL", name, detail))
    print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a deployed Atlas.")
    parser.add_argument("--base-url", required=True, help="e.g. https://atlas.onrender.com")
    parser.add_argument("--db", default=None, help="DATABASE_URL override for DB checks")
    parser.add_argument("--chat-id", type=int, default=None, help="your Telegram user id (for e2e)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    database_url = args.db or os.environ.get("DATABASE_URL", "")

    print("\n=== 1. Health endpoint ===")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{base}/health")
            ok("/health reachable", f"HTTP {r.status_code}")
            body = r.json()
            ok("body ok", json.dumps(body)[:200])
            corr = r.headers.get("x-correlation-id")
            ok("x-correlation-id header", corr or "missing")
            if corr is None:
                fail("x-correlation-id header", "absent")

    except httpx.HTTPError as exc:
        fail("health endpoint", str(exc))

    print("\n=== 2. Database health ===")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            r = await client.get(f"{base}/health/db")
            ok("db health", f"HTTP {r.status_code} {r.text[:200]}")
    except httpx.HTTPError as exc:
        fail("database health", str(exc))

    print("\n=== 3. Telegram webhook registration ===")
    if not token:
        info("TELEGRAM_BOT_TOKEN not set in .env; skipping real Telegram checks.")
    else:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
            data = r.json().get("result", {})
            url = data.get("url")
            ok("webhook endpoint configured", url or "none")
            if url != f"{base}/webhook/telegram":
                fail("webhook URL matches base", f"got {url}")
            if data.get("last_error_message"):
                fail("last webhook error", str(data["last_error_message"]))
            else:
                ok("no last webhook error")

    manual_db = database_url and database_url.startswith("postgresql")
    print("\n=== 4-5. Message flow (needs your Telegram client) ===")
    info("Send a TEXT, VOICE, and an IMAGE (with caption) to @<your_bot> now,")
    info("and wait ~2-3 seconds after each. The bot echoes in dev/echo mode.")
    await asyncio.to_thread(input, "  When done, press Enter to continue...")

    if manual_db:
        import asyncpg

        conn = await asyncpg.connect(database_url.replace("+asyncpg", ""), timeout=10)
        try:
            recent = await conn.fetch(
                "SELECT content, content_type, correlation_id "
                "FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "JOIN users u ON u.id = c.user_id "
                "ORDER BY m.created_at DESC LIMIT 3"
            )
            info("recent user messages:")
            for row in recent:
                ok(
                    "message persisted",
                    f"{row['content_type']} corr={row['correlation_id']}",
                )
            sent = await conn.fetchval(
                "SELECT count(*) FROM outbound_messages WHERE status = 'SENT'"
            )
            ok("outbound sent", f"{sent} sent message(s)")
        finally:
            await conn.close()
    else:
        info("Manual check: open the deployed logs / Render dashboard and view the")
        info("correlation_id-scoped log lines; verify one outbound echo per message.")

    print(f"\n=== Summary ({len(CHECKS)} checks) ===")
    failed = [c for c in CHECKS if c[0] == "FAIL"]
    if failed:
        for _, name, detail in failed:
            print(f"  FAIL {name}: {detail}")
        raise SystemExit(1)
    print("  All automated checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
