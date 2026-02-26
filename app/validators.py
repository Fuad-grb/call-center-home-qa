"""
Input validation and sanitization.
Handles edge cases: empty segments, broken JSON, too-short audio, etc.
"""

from __future__ import annotations
from typing import Any
from pydantic import ValidationError
from app.logger import get_logger
from app.models import CallTranscript
from config.settings import settings

logger = get_logger(__name__)


class ValidationResult:
    """Holds validation outcome with warnings."""

    def __init__(self, transcript: CallTranscript | None, warnings: list[str], is_valid: bool):
        self.transcript = transcript
        self.warnings = warnings
        self.is_valid = is_valid


def validate_input(raw_data: dict[str, Any]) -> ValidationResult:
    """
    Validate raw JSON input.

    Returns ValidationResult:
    - is_valid=False → fatal error, cannot process
    - is_valid=True + warnings → processable but has issues
    """
    warnings: list[str] = []

    # --- Basic structure checks ---
    if not isinstance(raw_data, dict):
        logger.error("Input is not a dictionary")
        return ValidationResult(None, ["Input is not a JSON object"], False)

    if "call_id" not in raw_data:
        logger.error("Missing call_id")
        return ValidationResult(None, ["Missing call_id"], False)

    if "segments" not in raw_data or not isinstance(raw_data.get("segments"), list):
        logger.error("Missing or invalid segments")
        return ValidationResult(None, ["Missing or invalid segments array"], False)

    if len(raw_data["segments"]) == 0:
        logger.error("Empty segments array")
        return ValidationResult(None, ["Segments array is empty"], False)

    # --- Normalize field names ---
    # Eval dataset uses "start"/"end", but task description shows "start_time"/"end_time"
    # We handle both formats here
    normalized_segments = []
    for i, seg in enumerate(raw_data["segments"]):
        if not isinstance(seg, dict):
            warnings.append(f"Segment {i} is not an object, skipped")
            continue

        normalized = dict(seg)
        if "start_time" in normalized and "start" not in normalized:
            normalized["start"] = normalized.pop("start_time")
        if "end_time" in normalized and "end" not in normalized:
            normalized["end"] = normalized.pop("end_time")

        normalized_segments.append(normalized)

    raw_data = {**raw_data, "segments": normalized_segments}

    # --- Parse with Pydantic ---
    try:
        transcript = CallTranscript(**raw_data)
    except ValidationError as e:
        logger.error("Validation failed", extra={"errors": str(e)})
        return ValidationResult(None, [f"Schema validation error: {e}"], False)

    # --- Post-parse warnings (non-fatal) ---

    empty_count = sum(1 for s in transcript.segments if s.is_empty)
    if empty_count > 0:
        warnings.append(f"{empty_count} empty/meaningless segment(s) detected")

    short_count = sum(
        1 for s in transcript.segments
        if s.duration < settings.min_segment_duration and not s.is_empty
    )
    if short_count > 0:
        warnings.append(f"{short_count} segment(s) shorter than {settings.min_segment_duration}s")

    if not transcript.operator_segments:
        warnings.append("No operator segments found — scoring may be unreliable")

    if not transcript.customer_segments:
        warnings.append("No customer segments found")

    if transcript.total_duration < 1.0:
        warnings.append(f"Very short call ({transcript.total_duration:.1f}s)")

    logger.info("Input validated", extra={
        "call_id": transcript.call_id,
        "segments": len(transcript.segments),
        "warnings": len(warnings),
    })

    return ValidationResult(transcript, warnings, True)
