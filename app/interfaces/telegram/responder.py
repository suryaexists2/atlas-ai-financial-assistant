"""Default dev responder. M3 replaces this with the Agent Core."""

from app.interfaces.telegram.normalized import NormalizedMessage


async def dev_echo_reply(message: NormalizedMessage) -> str:
    if message.is_media:
        short_id = message.media_file_id[:12]
        return (
            f"Got it — received your {message.media_type} (id: {short_id}…). "
            "I'll be able to analyze this soon."
        )
    return f"Got it — I heard: {message.combined_text}"
