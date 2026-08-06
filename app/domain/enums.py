"""Domain enums. Stored as VARCHAR in both SQLite and Postgres."""

from enum import StrEnum


class OnboardingStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    DOCUMENT = "document"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentKind(StrEnum):
    """Kind of a media/document payload detected before parsing."""

    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    VOICE = "voice"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class AlertKind(StrEnum):
    PRICE = "price"
    NEWS = "news"
    FILING = "filing"
    EARNINGS = "earnings"


class JobStatus(StrEnum):
    PENDING = "pending"
    EXECUTED = "executed"
    SKIPPED = "skipped"


class OutboundStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class IntegrationProvider(StrEnum):
    GMAIL = "gmail"
    CALENDAR = "calendar"
    DRIVE = "drive"
    SHEETS = "sheets"
