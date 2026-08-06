import asyncio

import httpx

TOKEN = "8237034978:AAG3AHAx94xQrpLLa26V_cKM2orn5fkcK_I"
BASE = "https://atlas-bot-peop.onrender.com"


async def main() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE}/health")
        print("health:", r.status_code, r.text[:120])

        info = await client.get(
            f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo"
        )
        data = info.json().get("result", {})
        print("webhook url:", data.get("url"))
        print("pending:", data.get("pending_update_count"))
        print("last_error:", data.get("last_error_message"))
        print("last_ip:", data.get("ip_address"))


asyncio.run(main())