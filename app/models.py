"""
Data models — the contract for input transcripts and output evaluations.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Enums ──

class Speaker(str, Enum):
    OPERATOR = "Operator"
    CUSTOMER = "Customer"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CriterionID(str, Enum):
    KR2_1 = "KR2.1"
    KR2_2 = "KR2.2"
    KR2_3 = "KR2.3"
    KR2_4 = "KR2.4"
    KR2_5 = "KR2.5"


# ── Input Models ──

class Segment(BaseModel):
    """One turn in a conversation (operator or customer speaks)."""

    speaker: Speaker
    text: str = ""
    start_time: float = Field(..., alias="start", ge=0)
    end_time: float = Field(..., alias="end", ge=0)

    model_config = {"populate_by_name": True}

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 0.0)

    @property
    def is_empty(self) -> bool:
        """Text is empty or meaningless (e.g. '...' or '.')"""
        return len(self.text) == 0 or self.text in ("...", "…", ".", "-")


class CallTranscript(BaseModel):
    """Full call transcript — the main input to the pipeline."""

    call_id: str
    segments: list[Segment] = Field(min_length=1)

    @property
    def operator_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.speaker == Speaker.OPERATOR]

    @property
    def customer_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.speaker == Speaker.CUSTOMER]

    @property
    def operator_text(self) -> str:
        """All operator speech joined into one string."""
        return " ".join(s.text for s in self.operator_segments if not s.is_empty)

    @property
    def customer_text(self) -> str:
        return " ".join(s.text for s in self.customer_segments if not s.is_empty)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments if not s.is_empty)

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end_time - self.segments[0].start_time


# ── Internal Models (passed between pipeline stages) ──

class RuleResult(BaseModel):
    """Output from rule-based check for one criterion."""

    score: Optional[int] = Field(None, ge=0, le=3)
    confidence: Confidence = Confidence.LOW
    needs_llm: bool = True
    reasoning: str = ""
    signals: list[str] = Field(default_factory=list)  # what rules detected


class LLMResult(BaseModel):
    """Output from LLM evaluation for one criterion."""

    score: int = Field(..., ge=0, le=3)
    reasoning: str = ""


# ── Output Models (final result) ──

class CriterionScore(BaseModel):
    """Final score for one criterion — goes into output JSON."""

    score: int = Field(..., ge=0, le=3)
    reasoning: str
    probability: Confidence  # how confident we are in this score


class CallEvaluation(BaseModel):
    """Complete evaluation output for one call."""

    call_id: str
    scores: dict[str, CriterionScore]
    pii_detected: bool = False
    warnings: list[str] = Field(default_factory=list)
