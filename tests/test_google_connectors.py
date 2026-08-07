"""Google connector tests: OAuth client, Gmail/Calendar/Drive APIs, tools,
refresh/retry semantics, the OAuth routes, and button payload plumbing."""

import asyncio
import base64
import datetime as dt
import json

import httpx
import pytest

from app.application.agent.tools import ToolContext, default_registry
from app.application.ingestion.types import MediaIngestionResult, ParsedDocument
from app.domain.enums import DocumentKind, IntegrationProvider
from app.infrastructure.providers.google_calendar import CalendarClient
from app.infrastructure.providers.google_drive import DriveClient
from app.infrastructure.providers.google_gmail import GmailClient
from app.infrastructure.providers.google_oauth import (
    GoogleOAuthClient,
    GoogleTokenExpiredError,
    TokenBundle,
)

SCOPES = ["email", "gmail.readonly", "calendar.events", "drive.readonly"]


def oauth_client(http=None) -> GoogleOAuthClient:
    return GoogleOAuthClient(
        "client-id", "client-secret", "https://atlas.test/oauth/google/callback", SCOPES, http=http
    )


# --- GoogleOAuthClient --------------------------------------------------------


def test_authorization_url_has_pkce_and_offline():
    client = oauth_client()
    url = client.authorization_url("state123", "verifier1234567890")
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "code_challenge=" in url and "code_challenge_method=S256" in url
    assert "access_type=offline" in url and "prompt=consent" in url
    assert "state=state123" in url
    assert "gmail.readonly" in url and "drive.readonly" in url


@pytest.mark.asyncio
async def test_exchange_code_parses_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/token"
        # httpx puts urlencoded body in content; verify grant type via body bytes
        body = request.content.decode()
        assert "grant_type=authorization_code" in body
        assert "code_verifier" in body
        return httpx.Response(
            200,
            json={
                "access_token": "acc1",
                "refresh_token": "ref1",
                "expires_in": 3600,
                "scope": "email gmail.readonly",
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bundle = await oauth_client(http).exchange_code("code", "verifier")
    assert bundle.access_token == "acc1"
    assert bundle.refresh_token == "ref1"
    assert bundle.scope == ["email", "gmail.readonly"]


@pytest.mark.asyncio
async def test_refresh_token_flow():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "grant_type=refresh_token" in request.content.decode()
        return httpx.Response(
            200,
            json={"access_token": "acc2", "expires_in": 3500, "scope": "email"},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bundle = await oauth_client(http).refresh_access_token("ref1")
    assert bundle.access_token == "acc2"
    assert bundle.refresh_token == "ref1"


@pytest.mark.asyncio
async def test_invalid_grant_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error":"invalid_grant"}')

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    from app.infrastructure.providers.google_oauth import GoogleOAuthError

    with pytest.raises(GoogleOAuthError) as exc:
        await oauth_client(http).exchange_code("c", "v")
    assert exc.value.kind == "invalid_grant"


@pytest.mark.asyncio
async def test_revoke_tolerates_invalid_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/revoke"
        return httpx.Response(400, text='{"error":"invalid_token"}')

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await oauth_client(http).revoke("ref1")  # must not raise


@pytest.mark.asyncio
async def test_userinfo_401_raises_token_expired():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(GoogleTokenExpiredError):
        await oauth_client(http).userinfo("acc")


# --- Gmail / Calendar / Drive clients -----------------------------------------


@pytest.mark.asyncio
async def test_gmail_search_and_message():
    body = base64.urlsafe_b64encode(b"Tesla earnings beat expectations.").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "gmail.googleapis.com"
        if request.url.path == "/gmail/v1/users/me/messages":
            return httpx.Response(
                200, json={"messages": [{"id": "m1", "threadId": "t1", "snippet": "s"}]}
            )
        if request.url.path == "/gmail/v1/users/me/messages/m1":
            return httpx.Response(
                200,
                json={
                    "id": "m1",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "boss@corp.com"},
                            {"name": "Subject", "value": "Q3 review"},
                            {"name": "Date", "value": "Mon, 3 Aug 2026"},
                        ],
                        "mimeType": "text/plain",
                        "body": {"data": body},
                    },
                },
            )
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gmail = GmailClient("tok", http=http)
    results = await gmail.search("tesla")
    assert results[0]["id"] == "m1"
    message = await gmail.get_message("m1")
    assert message["from"] == "boss@corp.com"
    assert message["subject"] == "Q3 review"
    assert "beat expectations" in message["body_excerpt"]


@pytest.mark.asyncio
async def test_gmail_401_raises_token_expired():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gmail = GmailClient("tok", http=http)
    with pytest.raises(GoogleTokenExpiredError):
        await gmail.search("tesla")


@pytest.mark.asyncio
async def test_calendar_list_and_create():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.googleapis.com"
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "ev0",
                            "summary": "Earnings call",
                            "start": {"dateTime": "2026-08-08T10:00:00Z"},
                        }
                    ]
                },
            )
        body = json.loads(request.content.decode())
        assert body["summary"] == "Team sync"
        assert body["attendees"] == [{"email": "a@b.com"}]
        return httpx.Response(
            200,
            json={
                "id": "ev1",
                "summary": "Team sync",
                "start": {"dateTime": body["start"]["dateTime"]},
                "end": {"dateTime": body["end"]["dateTime"]},
                "htmlLink": "https://calendar.google.com/event?eid=ev1",
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    calendar = CalendarClient("tok", http=http)
    events = await calendar.list_events(days=3)
    assert events[0]["summary"] == "Earnings call"
    start = dt.datetime.now(dt.UTC) + dt.timedelta(hours=2)
    event = await calendar.create_event(
        summary="Team sync",
        start=start,
        end=start + dt.timedelta(minutes=60),
        attendees=["a@b.com"],
    )
    assert event["id"] == "ev1"


@pytest.mark.asyncio
async def test_drive_search_download_and_export():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.googleapis.com"
        if request.url.path == "/drive/v3/files":
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "id": "f1",
                            "name": "nvidia.pdf",
                            "mimeType": "application/pdf",
                            "size": "1024",
                        }
                    ]
                },
            )
        if request.url.path == "/drive/v3/files/f1":
            return httpx.Response(200, content=b"%PDF-1.4 fake")
        if request.url.path == "/drive/v3/files/doc1/export":
            return httpx.Response(200, content=b"plain exported text")
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    drive = DriveClient("tok", http=http)
    files = await drive.search("nvidia")
    assert files[0]["name"] == "nvidia.pdf"
    assert await drive.download("f1", mime_type="application/pdf") == b"%PDF-1.4 fake"
    exported = await drive.download(
        "doc1", mime_type="application/vnd.google-apps.document", filename="Doc"
    )
    assert exported == b"plain exported text"


# --- Tools --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_google_unconfigured(uow, demo_user):
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"])
    registry = default_registry()
    result = json.loads(await registry.execute(ctx, "connect_google", {}))
    assert "not configured" in result["error"]
    assert ctx.oauth_connect_url is None


@pytest.mark.asyncio
async def test_connect_google_creates_flow_and_button_url(uow, demo_user):
    fake = oauth_client()
    ctx = ToolContext(
        uow=uow,
        user_id=demo_user["user_id"],
        google_oauth=fake,
        public_base_url="https://atlas.test/",
        chat_id=8478080533,
    )
    registry = default_registry()
    result = json.loads(await registry.execute(ctx, "connect_google", {}))
    assert "Tap the button below" in result["message"]
    assert ctx.oauth_connect_url is not None
    assert ctx.oauth_connect_url.startswith("https://atlas.test/oauth/google/start?state=")
    async with uow:
        flow = await uow.oauth_flows.get_by_state(ctx.oauth_connect_url.split("state=")[1])
    assert flow is not None
    assert flow.user_id == demo_user["user_id"]
    assert flow.chat_id == 8478080533
    assert flow.code_verifier


@pytest.mark.asyncio
async def test_connect_google_already_connected(uow, demo_user):
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"], provider=IntegrationProvider.GMAIL, access_token="t"
        )
        await uow.commit()
    ctx = ToolContext(
        uow=uow,
        user_id=demo_user["user_id"],
        google_oauth=oauth_client(),
        public_base_url="https://atlas.test/",
        chat_id=1,
    )
    result = json.loads(await default_registry().execute(ctx, "connect_google", {}))
    assert "already connected" in result["message"]
    assert ctx.oauth_connect_url is None


@pytest.mark.asyncio
async def test_disconnect_google_revokes_and_removes(uow, demo_user):
    revoked = []

    class FakeClient:
        configured = True

        async def revoke(self, token):
            revoked.append(token)

    async with uow:
        for provider in (
            IntegrationProvider.GMAIL,
            IntegrationProvider.CALENDAR,
            IntegrationProvider.DRIVE,
        ):
            await uow.integrations.upsert(
                demo_user["user_id"],
                provider=provider,
                access_token="a",
                refresh_token=f"ref-{provider.value}",
            )
        await uow.commit()

    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_oauth=FakeClient())
    result = json.loads(await default_registry().execute(ctx, "disconnect_google", {}))
    assert "gmail, calendar, drive" in result["message"]
    assert sorted(revoked) == ["ref-calendar", "ref-drive", "ref-gmail"]
    async with uow:
        for provider in (
            IntegrationProvider.GMAIL,
            IntegrationProvider.CALENDAR,
            IntegrationProvider.DRIVE,
        ):
            assert await uow.integrations.get_by_provider(demo_user["user_id"], provider) is None


@pytest.mark.asyncio
async def test_search_emails_not_connected(uow, demo_user):
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client())
    result = json.loads(await default_registry().execute(ctx, "search_emails", {"query": "tesla"}))
    assert "not connected" in result["error"]


def _gmail_http():
    body = base64.urlsafe_b64encode(b"discussed the merger").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gmail/v1/users/me/messages":
            return httpx.Response(
                200, json={"messages": [{"id": "m1", "threadId": "t1", "snippet": "s"}]}
            )
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "payload": {
                    "headers": [{"name": "Subject", "value": "Tesla update"}],
                    "mimeType": "text/plain",
                    "body": {"data": body},
                },
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_search_emails_connected(uow, demo_user):
    http = _gmail_http()
    client = oauth_client(http=http)
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"],
            provider=IntegrationProvider.GMAIL,
            access_token="tok",
            scopes=["gmail.readonly"],
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        )
        await uow.commit()
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_oauth=client, google_http=http)
    result = json.loads(await default_registry().execute(ctx, "search_emails", {"query": "tesla"}))
    assert result["emails"][0]["subject"] == "Tesla update"
    assert "discussed the merger" in result["emails"][0]["body_excerpt"]


@pytest.mark.asyncio
async def test_expired_token_refreshes_and_retries_once(uow, demo_user):
    token_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            token_calls["n"] += 1
            return httpx.Response(
                200,
                json={"access_token": "fresh", "expires_in": 3600, "scope": "gmail.readonly"},
            )
        return httpx.Response(200, json={"messages": []})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"],
            provider=IntegrationProvider.GMAIL,
            access_token="stale",
            refresh_token="ref1",
            scopes=["gmail.readonly"],
            expires_at=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5),
        )
        await uow.commit()

    ctx = ToolContext(
        uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client(http), google_http=http
    )
    result = json.loads(await default_registry().execute(ctx, "search_emails", {"query": "x"}))
    assert result == {"query": "x", "emails": []}
    assert token_calls["n"] == 1
    async with uow:
        link = await uow.integrations.get_by_provider(
            demo_user["user_id"], IntegrationProvider.GMAIL
        )
    assert link.access_token == "fresh"


@pytest.mark.asyncio
async def test_concurrent_calls_refresh_once(uow, demo_user):
    token_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            token_calls["n"] += 1
            return httpx.Response(
                200,
                json={"access_token": "fresh", "expires_in": 3600, "scope": "gmail.readonly"},
            )
        return httpx.Response(200, json={"messages": []})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"],
            provider=IntegrationProvider.GMAIL,
            access_token="stale",
            refresh_token="ref1",
            scopes=["gmail.readonly"],
            expires_at=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5),
        )
        await uow.commit()

    ctx = ToolContext(
        uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client(http), google_http=http
    )
    registry = default_registry()
    await asyncio.gather(
        registry.execute(ctx, "search_emails", {"query": "x"}),
        registry.execute(ctx, "search_emails", {"query": "y"}),
    )
    assert token_calls["n"] == 1  # lock serialized the refresh


@pytest.mark.asyncio
async def test_api_401_after_valid_token_refreshes_and_retries(uow, demo_user):
    token_calls = {"n": 0}
    api_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            token_calls["n"] += 1
            return httpx.Response(
                200,
                json={"access_token": "fresh", "expires_in": 3600, "scope": "gmail.readonly"},
            )
        api_calls["n"] += 1
        if api_calls["n"] == 1:
            return httpx.Response(401)
        return httpx.Response(200, json={"messages": []})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"],
            provider=IntegrationProvider.GMAIL,
            access_token="stale",
            refresh_token="ref1",
            scopes=["gmail.readonly"],
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        )
        await uow.commit()

    ctx = ToolContext(
        uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client(http), google_http=http
    )
    result = json.loads(await default_registry().execute(ctx, "search_emails", {"query": "x"}))
    assert result == {"query": "x", "emails": []}
    assert api_calls["n"] == 2
    assert token_calls["n"] == 1  # retried exactly once


@pytest.mark.asyncio
async def test_schedule_meeting_creates_event(uow, demo_user):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert "Team sync" in body["summary"]
        return httpx.Response(
            200,
            json={
                "id": "ev1",
                "summary": body["summary"],
                "start": {"dateTime": body["start"]["dateTime"]},
                "end": {"dateTime": body["end"]["dateTime"]},
                "htmlLink": "https://calendar.google.com/event?eid=ev1",
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"],
            provider=IntegrationProvider.CALENDAR,
            access_token="tok",
            scopes=["calendar.events"],
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        )
        await uow.commit()
    ctx = ToolContext(
        uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client(http), google_http=http
    )
    result = json.loads(
        await default_registry().execute(
            ctx,
            "schedule_meeting",
            {"summary": "Team sync", "when": "tomorrow 10:30"},
        )
    )
    assert result["event_id"] == "ev1"
    assert result["link"].startswith("https://calendar.google.com")


@pytest.mark.asyncio
async def test_read_drive_doc_uses_pipeline(uow, demo_user):
    class FakePipeline:
        async def process(self, *, file_id, mime_type, filename, data=None):
            assert data is not None and data.raw == b"%PDF-1.4 fake"
            return MediaIngestionResult(
                document=ParsedDocument(
                    kind=DocumentKind.PDF,
                    text="NVIDIA 2025 annual revenue grew 114%.",
                    filename=filename,
                    mime_type=mime_type,
                    byte_size=len(data.raw),
                    chunk_count=1,
                    truncated=False,
                ),
                content="NVIDIA 2025 annual revenue grew 114%.",
            )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/drive/v3/files":
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "id": "f1",
                            "name": "nvidia.pdf",
                            "mimeType": "application/pdf",
                            "size": "12",
                        }
                    ]
                },
            )
        return httpx.Response(200, content=b"%PDF-1.4 fake")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with uow:
        await uow.integrations.upsert(
            demo_user["user_id"],
            provider=IntegrationProvider.DRIVE,
            access_token="tok",
            scopes=["drive.readonly"],
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        )
        await uow.commit()
    ctx = ToolContext(
        uow=uow,
        user_id=demo_user["user_id"],
        google_oauth=oauth_client(http),
        google_http=http,
        media_pipeline=FakePipeline(),
    )
    result = json.loads(
        await default_registry().execute(ctx, "read_drive_doc", {"query": "nvidia"})
    )
    assert result["chosen"]["name"] == "nvidia.pdf"
    assert "revenue grew 114%" in result["content"]


@pytest.mark.asyncio
async def test_read_drive_doc_not_connected(uow, demo_user):
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client())
    result = json.loads(await default_registry().execute(ctx, "read_drive_doc", {"query": "x"}))
    assert "not connected" in result["error"]


@pytest.mark.asyncio
async def test_schedule_meeting_not_connected(uow, demo_user):
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_oauth=oauth_client())
    result = json.loads(
        await default_registry().execute(
            ctx, "schedule_meeting", {"summary": "x", "when": "tomorrow 09:00"}
        )
    )
    assert "not connected" in result["error"]


# --- Sender button plumbing ----------------------------------------------------


@pytest.mark.asyncio
async def test_sender_passes_reply_markup():
    from app.infrastructure.telegram.rate_limit import RateLimiter
    from app.infrastructure.telegram.sender import TelegramSender

    class FakeApi:
        def __init__(self):
            self.sent = []

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
            return {}

    api = FakeApi()
    sender = TelegramSender(
        api, RateLimiter(global_per_sec=1000.0, per_chat_per_sec=1000.0, burst=100)
    )
    markup = {"inline_keyboard": [[{"text": "Connect Google", "url": "https://atlas.test/start"}]]}
    ok = await sender.send(
        chat_id=1, payload={"type": "text", "text": "Tap below", "reply_markup": markup}
    )
    assert ok
    assert api.sent[0]["reply_markup"] == markup


@pytest.mark.asyncio
async def test_sender_rejects_bad_reply_markup():
    from app.infrastructure.telegram.rate_limit import RateLimiter
    from app.infrastructure.telegram.sender import TelegramSender

    sender = TelegramSender(
        None, RateLimiter(global_per_sec=1000.0, per_chat_per_sec=1000.0, burst=100)
    )  # type: ignore[arg-type] - rejected before api use
    ok = await sender.send(chat_id=1, payload={"type": "text", "text": "x", "reply_markup": "nope"})
    assert not ok


# --- OAuth routes --------------------------------------------------------------


@pytest.fixture
def fake_oauth_client():
    class Fake:
        configured = True

        def __init__(self):
            self.exchanged: list[str] = []
            self.revoked: str | None = None

        def authorization_url(self, state, verifier):
            return f"https://accounts.google.com/o/oauth2/v2/auth?state={state}"

        async def exchange_code(self, code, verifier):
            self.exchanged.append(code)
            return TokenBundle(
                access_token="acc",
                refresh_token="ref",
                expires_in=3600,
                scope=SCOPES,
            )

        async def revoke(self, token):
            self.revoked = token

        def lock_for(self, user_id):
            return asyncio.Lock()

    return Fake()


@pytest.fixture
def oauth_client_fixture(session_factory, fake_oauth_client, monkeypatch):
    from app.core.config import Settings
    from app.interfaces.api.routes import oauth
    from app.main import create_app

    settings = Settings(
        app_env="test",
        database_url="sqlite+aiosqlite://",
        google_oauth_client_id="cid",
        google_oauth_client_secret="csec",
        public_base_url="http://test",
    )
    app = create_app(settings)
    app.state.session_factory = session_factory
    monkeypatch.setattr(oauth, "_oauth_client", lambda settings: fake_oauth_client)
    return app, fake_oauth_client


@pytest.mark.asyncio
async def test_oauth_start_redirects_and_callback_stores_tokens(
    session_factory, oauth_client_fixture, monkeypatch
):
    import uuid as _uuid

    from httpx import ASGITransport, AsyncClient

    app, fake = oauth_client_fixture
    user_id = _uuid.uuid4()
    async with session_factory() as session:
        from app.domain.entities import User

        session.add(User(id=user_id, telegram_id=900001, username="oauthuser", timezone="UTC"))
        await session.commit()

    async with session_factory() as session:
        from app.infrastructure.db.repositories.memory_jobs_outbox import SqlOAuthFlowRepository

        repo = SqlOAuthFlowRepository(session)
        flow = await repo.create(
            state="st" + "x" * 40,
            user_id=user_id,
            chat_id=424242,
            code_verifier="verifier" + "y" * 40,
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(minutes=10),
        )
        await session.commit()
        state = flow.state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        start = await ac.get("/oauth/google/start", params={"state": state})
        assert start.status_code == 307
        assert start.headers["location"].startswith("https://accounts.google.com")

        cb = await ac.get("/oauth/google/callback", params={"state": state, "code": "authcode"})
        assert cb.status_code == 200
        assert "Connected to Atlas" in cb.text

        # State is one-time: a second callback with the same state must fail.
        replay = await ac.get(
            "/oauth/google/callback", params={"state": state, "code": "authcode2"}
        )
        assert replay.status_code == 400

        bogus = await ac.get(
            "/oauth/google/callback", params={"state": "zz" + "q" * 40, "code": "c"}
        )
        assert bogus.status_code == 400

    async with session_factory() as session:
        from sqlalchemy import select

        from app.domain.entities import IntegrationLink, OutboundMessage

        links = (await session.execute(select(IntegrationLink))).scalars().all()
        providers = {link.provider.value for link in links}
        assert providers == {"gmail", "calendar", "drive"}
        assert all(link.access_token == "acc" for link in links)
        assert all(link.refresh_token == "ref" for link in links)

        outbox = (await session.execute(select(OutboundMessage))).scalars().all()
        assert any(m.chat_id == 424242 for m in outbox)
        assert any("Google connected" in m.payload.get("text", "") for m in outbox)


@pytest.mark.asyncio
async def test_oauth_start_rejects_unknown_state(session_factory, oauth_client_fixture):
    from httpx import ASGITransport, AsyncClient

    app, _ = oauth_client_fixture
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/oauth/google/start", params={"state": "nope" + "z" * 40})
    assert resp.status_code == 400
