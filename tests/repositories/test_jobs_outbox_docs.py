"""Job, alert, outbox, document, integration repository tests (M1)."""

import datetime as dt

import pytest

from app.domain.enums import (
    AlertKind,
    DocumentStatus,
    IntegrationProvider,
    OutboundStatus,
)


@pytest.mark.asyncio
async def test_job_crud_and_enable_listing(uow, demo_user):
    user_id = demo_user["user_id"]
    job_id = None

    async with uow:
        job = await uow.jobs.create(
            job_type="morning_brief",
            cron_expr="0 8 * * *",
            user_id=user_id,
            params={"markets": "us"},
        )
        await uow.commit()
        job_id = job.id

    async with uow:
        jobs = await uow.jobs.list_enabled()
        assert len(jobs) == 1 and jobs[0].id == job_id
        fetched = await uow.jobs.get_by_id(job_id)
        assert fetched.params == {"markets": "us"}

    async with uow:
        job = await uow.jobs.get_by_id(job_id)
        job.enabled = False
        await uow.commit()

    async with uow:
        assert len(await uow.jobs.list_enabled()) == 0

    async with uow:
        job = await uow.jobs.get_by_id(job_id)
        await uow.jobs.delete(job)
        await uow.commit()
        assert await uow.jobs.get_by_id(job_id) is None


@pytest.mark.asyncio
async def test_record_run_is_idempotent(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        job = await uow.jobs.create(
            job_type="watchlist_alert_check",
            cron_expr="*/5 * * * *",
            user_id=user_id,
        )
        await uow.commit()
        job_id = job.id

    fired_at = dt.datetime(2026, 8, 6, 8, 0, tzinfo=dt.UTC)
    async with uow:
        first = await uow.jobs.record_run(job_id, run_key="k1", scheduled_at=fired_at)
        second = await uow.jobs.record_run(job_id, run_key="k1", scheduled_at=fired_at)
        await uow.commit()
        assert first is True
        assert second is False  # duplicate run rejected


@pytest.mark.asyncio
async def test_lock_for_run_respects_disabled(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        job = await uow.jobs.create(
            job_type="morning_brief", cron_expr="0 8 * * *", user_id=user_id
        )
        job.enabled = False
        await uow.commit()
        job_id = job.id

    async with uow:
        locked = await uow.jobs.lock_for_run(job_id)
        assert locked is None


@pytest.mark.asyncio
async def test_update_run_state(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        job = await uow.jobs.create(
            job_type="morning_brief", cron_expr="0 8 * * *", user_id=user_id
        )
        await uow.commit()
        job_id = job.id

    async with uow:
        job = await uow.jobs.get_by_id(job_id)
        last = dt.datetime(2026, 8, 6, 8, 0, tzinfo=dt.UTC)
        nxt = dt.datetime(2026, 8, 7, 8, 0, tzinfo=dt.UTC)
        await uow.jobs.update_run_state(job, last_run_at=last, next_run_at=nxt)
        await uow.commit()

    async with uow:
        job = await uow.jobs.get_by_id(job_id)
        # SQLite returns naive datetimes; normalize before comparing
        assert job.last_run_at.replace(tzinfo=dt.UTC) == last
        assert job.next_run_at.replace(tzinfo=dt.UTC) == nxt


@pytest.mark.asyncio
async def test_alert_create_list_enabled(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        alert = await uow.alerts.create(
            user_id,
            kind=AlertKind.PRICE,
            symbol="NVDA",
            condition={"op": "gte", "threshold_pct": 5},
        )
        await uow.commit()
        alert_id = alert.id

    async with uow:
        fetched = await uow.alerts.list_enabled(user_id)
        assert len(fetched) == 1
        assert fetched[0].id == alert_id
        assert fetched[0].condition["threshold_pct"] == 5

    async with uow:
        fetched = await uow.alerts.get_by_id(alert_id)
        await uow.alerts.delete(fetched)
        await uow.commit()
        assert await uow.alerts.get_by_id(alert_id) is None


@pytest.mark.asyncio
async def test_outbox_enqueue_claim_fail_sent(uow, demo_user):
    async with uow:
        low = await uow.outbox.enqueue(chat_id=42, payload={"text": "brief"}, priority=0)
        high = await uow.outbox.enqueue(chat_id=42, payload={"text": "alert"}, priority=10)
        await uow.commit()

    async with uow:
        claimed = await uow.outbox.claim_due(limit=10)
        assert [m.payload["text"] for m in claimed] == ["alert", "brief"]  # high priority first
        high_c = next(m for m in claimed if m.id == high.id)
        low_c = next(m for m in claimed if m.id == low.id)
        await uow.outbox.mark_sent(high_c)
        retry_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)
        await uow.outbox.mark_failed(low_c, error="timeout", next_retry_at=retry_at)
        await uow.commit()

    async with uow:
        remaining = await uow.outbox.claim_due(limit=10)
        assert all(m.id != high.id for m in remaining)  # sent, not claimed
        assert len(remaining) == 0 or all(m.id != low.id for m in remaining)  # not due yet

    async with uow:
        from sqlalchemy import select

        from app.domain.entities import OutboundMessage

        result = await uow.session.execute(
            select(OutboundMessage).where(OutboundMessage.id == low.id)
        )
        failed = result.scalar_one()
        assert failed.status == OutboundStatus.PENDING
        assert failed.last_error == "timeout"
        assert failed.attempt == 1


@pytest.mark.asyncio
async def test_outbox_terminal_failure_when_retry_exhausted(uow, demo_user):
    async with uow:
        msg = await uow.outbox.enqueue(priority=0, chat_id=42, payload={"text": "x"})
        await uow.commit()
        msg_id = msg.id

    async with uow:
        from sqlalchemy import select

        from app.domain.entities import OutboundMessage

        result = await uow.session.execute(
            select(OutboundMessage).where(OutboundMessage.id == msg_id)
        )
        m = result.scalar_one()
        await uow.outbox.mark_failed(m, error="give up", next_retry_at=None)
        await uow.commit()

    async with uow:
        from sqlalchemy import select

        from app.domain.entities import OutboundMessage

        result = await uow.session.execute(
            select(OutboundMessage).where(OutboundMessage.id == msg_id)
        )
        assert result.scalar_one().status == OutboundStatus.FAILED


@pytest.mark.asyncio
async def test_document_lifecycle(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        doc = await uow.documents.create(
            user_id,
            filename="annual_report.pdf",
            mime_type="application/pdf",
            size_bytes=12345,
        )
        await uow.commit()
        doc_id = doc.id
        assert doc.status == DocumentStatus.PENDING

    async with uow:
        doc = await uow.documents.get_by_id(doc_id)
        await uow.documents.update_status(
            doc, DocumentStatus.PROCESSED, chunk_count=42, tokens=9999
        )
        await uow.commit()

    async with uow:
        doc = await uow.documents.get_by_id(doc_id)
        assert doc.status == DocumentStatus.PROCESSED
        assert doc.doc_meta["chunk_count"] == 42

    async with uow:
        docs = await uow.documents.list_for_user(user_id)
        assert len(docs) == 1


@pytest.mark.asyncio
async def test_integration_upsert_and_get(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        await uow.integrations.upsert(
            user_id,
            provider=IntegrationProvider.GMAIL,
            access_token="tok",
            scopes=["gmail.readonly"],
        )
        await uow.commit()

    async with uow:
        link = await uow.integrations.get_by_provider(user_id, IntegrationProvider.GMAIL)
        assert link is not None and link.access_token == "tok"

        await uow.integrations.upsert(
            user_id,
            provider=IntegrationProvider.GMAIL,
            access_token="tok2",
            scopes=["gmail.readonly", "gmail.send"],
        )
        await uow.commit()

    async with uow:
        link = await uow.integrations.get_by_provider(user_id, IntegrationProvider.GMAIL)
        assert link.access_token == "tok2"
        assert "gmail.send" in link.scopes
