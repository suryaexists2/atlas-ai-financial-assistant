"""Reply sanitizer regression tests (Telegram UX contract).

Guarantees raw LLM/tool plumbing can never reach an outbound message:
tool names, function-call syntax, JSON, debug metadata, gateway errors,
and malformed markdown are stripped; legitimate content (quotes, bullets,
links, Hindi/Hinglish) is preserved.
"""

from app.interfaces.telegram.sanitize import sanitize_reply


def assert_clean(text: str) -> str:
    result = sanitize_reply(text)
    for token in (
        "get_market_quote",
        "get_market_news",
        "get_company_profile",
        "get_company_filings",
        "get_market_indices",
        "save_memory",
        "tool_calls",
        "arguments",
        "HTTP ",
        "status_code",
        "LLM provider",
    ):
        assert token not in result
    return result


def test_quote_tool_call_never_reaches_outbound():
    raw = "Tesla price is $312.41 (get_market_quote(symbol=\"TSLA\"))"
    assert_clean(raw)
    assert sanitize_reply(raw) == "Tesla price is $312.41"


def test_bare_function_call_syntax_removed():
    raw = "Nvidia's move today: get_market_quote(symbol=\"NVDA\") returned $218.99"
    assert sanitize_reply(raw) == "Nvidia's move today: returned $218.99"


def test_wrapped_tool_ref_removed():
    raw = "(get_market_news) Here is what happened in markets today."
    assert sanitize_reply(raw) == "Here is what happened in markets today."


def test_json_function_call_fragment_removed():
    raw = (
        "Result: {\"name\": \"get_market_quote\", \"arguments\": "
        "{\"symbol\": \"TSLA\"}} Tesla is at $312.41"
    )
    assert_clean(raw)
    assert sanitize_reply(raw) == "Result: Tesla is at $312.41"


def test_multi_tool_response_cleaned():
    raw = (
        "AAPL quote (get_market_quote(symbol=\"AAPL\")): $234. News: "
        "(get_company_news) five headlines. Filings: get_company_filings(symbol=\"AAPL\")."
    )
    cleaned = assert_clean(raw)
    assert cleaned == "AAPL quote : $234. News: five headlines. Filings: ."


def test_tool_error_response_never_leaks_gateway_details():
    raw = (
        "I couldn't retrieve that right now. LLM provider error 402: "
        "insufficient credits (get_market_quote(symbol=\"TSLA\"))"
    )
    cleaned = assert_clean(raw)
    assert cleaned == "I couldn't retrieve that right now."
    assert "402" not in cleaned


def test_long_response_formatting_normalized():
    raw = (
        "## Market Summary\n\n"
        "```json\n{\"name\": \"get_market_news\"}\n```\n"
        "- NVDA +1.2%\n- TSLA -0.4%\n\n\n\n"
        "### Takeaway\nPrices may be delayed."
    )
    cleaned = sanitize_reply(raw)
    assert "```" not in cleaned
    assert "get_market_news" not in cleaned
    assert "##" not in cleaned
    assert "###" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Market Summary" in cleaned
    assert "- NVDA +1.2%" in cleaned
    assert "Prices may be delayed." in cleaned


def test_hindi_hinglish_formatting_preserved():
    raw = "NVDA ka price dekho (get_market_quote(symbol=\"NVDA\"))"
    assert sanitize_reply(raw) == "NVDA ka price dekho"


def test_legitimate_parens_and_links_preserved():
    raw = "Watchlist: (TSLA) and (NVDA). Source: https://www.reuters.com/technology"
    assert sanitize_reply(raw) == raw


def test_heading_marks_removed_but_bold_kept():
    raw = "## Quick view\n**Tesla (TSLA)** is trading around **$312.41**."
    cleaned = sanitize_reply(raw)
    assert cleaned == "Quick view\n**Tesla (TSLA)** is trading around **$312.41**."


def test_empty_input_returns_empty():
    assert sanitize_reply("") == ""
    assert sanitize_reply("   ") == ""
