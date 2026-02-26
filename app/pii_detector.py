"""
PII detection and masking via regex.

Detects:
- FIN codes (7 alphanumeric, Azerbaijani ID)
- Card numbers (16 digits, with/without spaces)
- Phone numbers (Azerbaijani: +994, 050/055/070 etc.)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PIIMatch:
    pii_type: str   # "FIN", "CARD", "PHONE"
    value: str
    start: int
    end: int


@dataclass
class PIIResult:
    has_pii: bool = False
    matches: list[PIIMatch] = field(default_factory=list)
    masked_text: str = ""


# ── Regex Patterns ──

# Card: 16 digits, optionally grouped by 4
_CARD_PATTERN = re.compile(
    r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b'
)

# Azerbaijani phone numbers
_PHONE_PATTERN = re.compile(
    r'(\+994[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})'
    r'|(\b0[1-9]\d[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b)'
    r'|(\(0[1-9]\d\)[\s\-]?\d{3}[\s\-]?\d{2,4}\b)'
)

# FIN: exactly 7 uppercase letters+digits (must have BOTH letters and digits)
_FIN_PATTERN = re.compile(r'\b([A-Z0-9]{7})\b')

# Words that look like FIN but aren't
_FIN_WHITELIST = {"KONTAKT", "SAMSUNG", "IPHONE", "ANDROID", "WINDOWS", "PREMIUM", "STANDART"}


def _is_likely_fin(candidate: str, context: str) -> bool:
    """Check if 7-char match is actually a FIN code, not a product name."""
    if candidate.upper() in _FIN_WHITELIST:
        return False
    # FIN must have BOTH letters and digits
    has_letter = any(c.isalpha() for c in candidate)
    has_digit = any(c.isdigit() for c in candidate)
    return has_letter and has_digit


def detect_pii(text: str) -> PIIResult:
    """Scan text for PII. Returns matches and masked version."""
    if not text or not text.strip():
        return PIIResult(has_pii=False, masked_text=text)

    matches: list[PIIMatch] = []

    # Cards first (most specific — 16 digits)
    for m in _CARD_PATTERN.finditer(text):
        matches.append(PIIMatch("CARD", m.group(), m.start(), m.end()))

    # Phones
    for m in _PHONE_PATTERN.finditer(text):
        value = m.group(1) or m.group(2) or m.group(3)
        if value:
            matches.append(PIIMatch("PHONE", value, m.start(), m.end()))

    # FIN codes (with disambiguation)
    for m in _FIN_PATTERN.finditer(text):
        candidate = m.group(1)
        ctx_start = max(0, m.start() - 50)
        ctx_end = min(len(text), m.end() + 50)
        context = text[ctx_start:ctx_end]

        if _is_likely_fin(candidate, context):
            # Don't double-count if overlaps with card/phone
            overlap = any(ex.start <= m.start() < ex.end for ex in matches)
            if not overlap:
                matches.append(PIIMatch("FIN", candidate, m.start(), m.end()))

    # Build masked text (replace from end to preserve indices)
    masked = text
    for match in sorted(matches, key=lambda x: x.start, reverse=True):
        masked = masked[:match.start] + "****" + masked[match.end:]

    has_pii = len(matches) > 0
    if has_pii:
        logger.info("PII detected", extra={
            "types": [m.pii_type for m in matches],
            "count": len(matches),
        })

    return PIIResult(has_pii=has_pii, matches=matches, masked_text=masked)
