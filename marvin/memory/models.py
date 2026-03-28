import datetime
import uuid

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime.datetime


class ConversationSummary(BaseModel):
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    summary_text: str
    message_count: int
    messages: list[Message]  # originals preserved — no data loss
    start_index: int
    end_index: int
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
