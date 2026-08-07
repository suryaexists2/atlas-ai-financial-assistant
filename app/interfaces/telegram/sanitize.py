"""Final reply sanitizer for Telegram delivery.

The last gate before a reply is persisted/enqueued. Guarantees the user never
sees raw LLM/tool plumbing: leaked tool names, function-call syntax, JSON,
debug metadata, model/gateway error strings, or malformed markdown.

This is defense-in-depth: the system prompt already tells the model to write
clean prose; this layer catches whatever still slips through.
"""

from __future__ import annotations

import re

from app.application.agent.tools import DEFAULT_TOOLS

# Wrapped tool refs, both plain and with inline args:
#   (get_market_quote)
#   (get_market_quote(symbol="TSLA"))
#   (get_company_profile(symbol="AAPL"))
_WRAPPED_TOOL_PATTERN = re.compile(
    r"\(\s*(?:" + "|".join(re.escape(t.name) for t in DEFAULT_TOOLS) + r")(?:\s*\([^)]*\))?\s*\)",
    re.IGNORECASE,
)
# Bare function-call syntax with arguments:
#   get_market_quote(symbol="TSLA")
_BARE_TOOL_CALL_PATTERN = re.compile(
    r"(?<![\w.])(?:" + "|".join(re.escape(t.name) for t in DEFAULT_TOOLS) + r")\s*\([^)]*\)",
    re.IGNORECASE,
)
# JSON fragments that look like serialized function calls:
#   {"name": "get_market_quote", "arguments": {"symbol": "TSLA"}}
# Tolerates one level of nested braces (the arguments object).
_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}")


def _drop_json_tool_call(match: re.Match) -> str:
    fragment = match.group(0)
    has_name = re.search(r'"(?:name|tool_call)"\s*:', fragment)
    if not has_name:
        return fragment
    has_arguments = re.search(r'"(?:arguments|tool_calls)"\s*:', fragment)
    mentions_tool = any(f'"{t.name}"' in fragment for t in DEFAULT_TOOLS)
    if has_arguments or mentions_tool:
        return ""
    return fragment
# Debug/gateway noise the model may echo verbatim.
_DEBUG_NOISE_PATTERN = re.compile(
    r"\[(?:took|llm|usage|status)[^\]]{0,80}\]|(?:LLM provider error|LLM request timed out|"
    r"HTTP\s+\d{3}|finish_reason|status_code|tool_choice)[^\n]{0,60}",
    re.IGNORECASE,
)
# Tool-call labels like "Tool call: get_market_quote".
_TOOL_LABEL_PATTERN = re.compile(
    r"(?:tool(?:_call)?|function(?:_call)?)\s*[:=]\s*",
    re.IGNORECASE,
)
# Fenced code blocks (often carry leaked JSON).
_CODE_FENCE_PATTERN = re.compile(r"```[a-z]*\s*\n.*?\n```\s*", re.IGNORECASE | re.DOTALL)
# Markdown ATX headings (Telegram does not render "#").
_HEADING_PATTERN = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


def sanitize_reply(text: str) -> str:
    """Removes tool/plumbing remnants and normalizes formatting for Telegram."""
    if not text:
        return ""
    cleaned = _CODE_FENCE_PATTERN.sub("", text)
    cleaned = _WRAPPED_TOOL_PATTERN.sub("", cleaned)
    cleaned = _BARE_TOOL_CALL_PATTERN.sub("", cleaned)
    cleaned = _JSON_OBJECT_PATTERN.sub(_drop_json_tool_call, cleaned)
    cleaned = _TOOL_LABEL_PATTERN.sub("", cleaned)
    cleaned = _DEBUG_NOISE_PATTERN.sub("", cleaned)
    cleaned = _HEADING_PATTERN.sub("", cleaned)
    cleaned = _MULTI_NEWLINE_PATTERN.sub("\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


__all__ = ["sanitize_reply"]
