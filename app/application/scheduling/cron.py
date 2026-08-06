"""Cron expression helpers for the DB-backed scheduler."""

from __future__ import annotations

import datetime as dt
import re
import zoneinfo
from typing import Any

from croniter import croniter

UTC = zoneinfo.ZoneInfo("UTC")


def is_valid_cron(expr: str) -> bool:
    """True when `expr` is a parseable 5-field cron expression."""
    try:
        croniter(expr, dt.datetime.now(UTC))
        return True
    except (ValueError, KeyError, TypeError):
        return False


def compute_next_run(expr: str, after: dt.datetime | None = None) -> dt.datetime | None:
    """Next fire time strictly after `after` (default: now), in UTC.

    Returns None when the expression is unparseable.
    """
    base = (after or dt.datetime.now(UTC)).astimezone(UTC)
    try:
        iterator = croniter(expr, base)
    except (ValueError, KeyError, TypeError):
        return None
    return iterator.get_next(dt.datetime).astimezone(UTC)


def cron_from_local_time(time: str, tz_name: str | None = None) -> str:
    """Turns "HH:MM" (user-local) into a UTC cron "minute hour * * *".

    Unknown/invalid timezones fall back to interpreting the time as UTC.
    """
    hour, sep, minute = time.partition(":")
    if not sep:
        return f"0 {int(hour or 0)} * * *"
    local_dt = dt.datetime.combine(
        dt.date(2026, 1, 1), dt.time(int(hour), int(minute))
    )
    tz: Any = UTC
    if tz_name:
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except (zoneinfo.ZoneInfoNotFoundError, TypeError):
            tz = UTC
    utc_dt = local_dt.replace(tzinfo=tz).astimezone(UTC)
    return f"{utc_dt.minute} {utc_dt.hour} * * *"


def extract_clock_time(text: str | None) -> str | None:
    """Pull an "HH:MM" (with optional am/pm) out of free text, or None."""
    if not text:
        return None
    match = re.search(r"(?i)(\d{1,2}):(\d{2})\s*(am|pm)?(?=\D|$)", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    period = (match.group(3) or "").lower()
    if period == "pm" and hour < 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


__all__ = ["UTC", "compute_next_run", "cron_from_local_time", "extract_clock_time", "is_valid_cron"]
