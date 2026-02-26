"""
Rule-based evaluation engine.

Each criterion gets deterministic Python checks (keywords, patterns, timing).
Rules handle what's deterministic. LLM handles semantics.
"""

from __future__ import annotations
from difflib import SequenceMatcher
from app.logger import get_logger
from app.models import CallTranscript, Confidence, RuleResult, Segment, Speaker

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# KR2.5 — Professional behavior (most rule-friendly)
# ═══════════════════════════════════════════════════════════

_INTERNAL_LEAK_PHRASES = [
    "micro donub", "mikro donub", "sistem donub", "sistem işləmir",
    "internet zəifdir", "outlook işləmir", "server düşüb",
    "proqram xətası", "texniki nasazlıq bizdə", "crm işləmir",
    "bizdə problem var", "şirkət böhran", "şirkətdə problem",
    "daxili problem", "daxili sistem",
]

_PERSONAL_OPINION_PHRASES = [
    "şəxsən mən", "məncə belə", "mən olsaydım",
    "mən sizin yerinizə", "özüm etməzdim",
    "düzünü desəm", "açığını deyim",
]

_SUSPICIOUS_PHRASES = [
    "bilmirəm", "əmin deyiləm", "deyə bilmərəm",
    "gözləyin bir az", "sistem yavaşdır",
]


def evaluate_kr25(transcript: CallTranscript) -> RuleResult:
    """KR2.5: Professional behavior — keyword detection."""
    operator_text = transcript.operator_text.lower()
    signals: list[str] = []

    # Internal info leak → definitive score 0
    for phrase in _INTERNAL_LEAK_PHRASES:
        if phrase in operator_text:
            signals.append(f"Daxili məlumat sızması: '{phrase}'")

    if signals:
        return RuleResult(
            score=0, confidence=Confidence.HIGH, needs_llm=False,
            reasoning="Operator daxili prosedur/problemləri müştəriyə bildirib",
            signals=signals,
        )

    # Personal opinion → definitive score 0
    for phrase in _PERSONAL_OPINION_PHRASES:
        if phrase in operator_text:
            signals.append(f"Şəxsi fikir: '{phrase}'")

    if signals:
        return RuleResult(
            score=0, confidence=Confidence.HIGH, needs_llm=False,
            reasoning="Operator şəxsi fikir bildirib",
            signals=signals,
        )

    # Suspicious but not definitive → LLM needed
    for phrase in _SUSPICIOUS_PHRASES:
        if phrase in operator_text:
            signals.append(f"Şübhəli ifadə: '{phrase}'")

    if signals:
        return RuleResult(
            score=None, confidence=Confidence.LOW, needs_llm=True,
            reasoning="Şübhəli ifadələr aşkarlandı, semantik yoxlama lazımdır",
            signals=signals,
        )

    # Clean — probably fine, but LLM should confirm
    return RuleResult(
        score=3, confidence=Confidence.MEDIUM, needs_llm=True,
        reasoning="Açıq pozuntu aşkarlanmadı",
        signals=[],
    )


# ═══════════════════════════════════════════════════════════
# KR2.2 — Requirement formation (partially rule-based)
# ═══════════════════════════════════════════════════════════

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_questions(segments: list[Segment], speaker: Speaker) -> list[str]:
    questions = []
    for seg in segments:
        if seg.speaker != speaker or seg.is_empty:
            continue
        for sentence in seg.text.replace("!", ".").split("."):
            sentence = sentence.strip()
            if sentence.endswith("?"):
                questions.append(sentence)
    return questions


def evaluate_kr22(transcript: CallTranscript) -> RuleResult:
    """KR2.2: Requirement formation — check for repeated/unanswered questions."""
    signals: list[str] = []

    operator_questions = _extract_questions(transcript.segments, Speaker.OPERATOR)

    # Check operator asked duplicate questions
    seen: list[str] = []
    repeat_count = 0
    for q in operator_questions:
        for prev in seen:
            if _similarity(q, prev) > 0.7:
                repeat_count += 1
                signals.append(f"Təkrarlanan sual: '{q[:60]}'")
                break
        seen.append(q)

    # Check if customer questions got answered
    unanswered = 0
    for i, seg in enumerate(transcript.segments):
        if seg.speaker == Speaker.CUSTOMER and "?" in seg.text:
            next_op = None
            for j in range(i + 1, len(transcript.segments)):
                if transcript.segments[j].speaker == Speaker.OPERATOR:
                    next_op = transcript.segments[j]
                    break
            if next_op is None or next_op.is_empty:
                unanswered += 1
                signals.append(f"Cavabsız sual: '{seg.text[:60]}'")

    if repeat_count >= 3 or unanswered >= 2:
        return RuleResult(
            score=0, confidence=Confidence.MEDIUM, needs_llm=True,
            reasoning="Çoxlu təkrar və/ya cavabsız suallar",
            signals=signals,
        )

    if repeat_count >= 1 or unanswered >= 1:
        return RuleResult(
            score=None, confidence=Confidence.LOW, needs_llm=True,
            reasoning="Bəzi təkrarlar/cavabsız suallar aşkarlandı",
            signals=signals,
        )

    return RuleResult(
        score=None, confidence=Confidence.LOW, needs_llm=True,
        reasoning="Aşkar problem tapılmadı, semantik yoxlama lazımdır",
        signals=signals,
    )


# ═══════════════════════════════════════════════════════════
# KR2.4 — Routing and Registration (keyword-based)
# ═══════════════════════════════════════════════════════════

_ROUTING_KEYWORDS = [
    "mağaza", "filial", "şöbə", "texnik", "usta",
    "qaynar xətt", "hüquq", "rəhbər", "müdir",
    "zəng edəcəyik", "geri dönəcəyik", "yönləndirirəm",
    "qeydə alıram", "qeyd etdim", "qeydiyyat",
    "sifarişiniz", "müraciətiniz", "nömrəniz",
]


def evaluate_kr24(transcript: CallTranscript) -> RuleResult:
    """KR2.4: Routing/registration — keyword detection."""
    operator_text = transcript.operator_text.lower()
    signals: list[str] = []

    found = [kw for kw in _ROUTING_KEYWORDS if kw in operator_text]

    if found:
        signals.append(f"Yönləndirmə/qeydiyyat sözləri: {found}")
        return RuleResult(
            score=None, confidence=Confidence.LOW, needs_llm=True,
            reasoning="Yönləndirmə/qeydiyyat sözləri tapıldı, kontekst yoxlaması lazımdır",
            signals=signals,
        )

    return RuleResult(
        score=None, confidence=Confidence.LOW, needs_llm=True,
        reasoning="Yönləndirmə/qeydiyyat sözləri tapılmadı",
        signals=signals,
    )


# ═══════════════════════════════════════════════════════════
# KR2.1 — Active help (mostly LLM, rules check timing)
# ═══════════════════════════════════════════════════════════

_LONG_PAUSE_THRESHOLD = 30.0  # seconds


def evaluate_kr21(transcript: CallTranscript) -> RuleResult:
    """KR2.1: Active help — check response gaps and dismissive phrases."""
    signals: list[str] = []

    # Long pauses between customer and operator
    long_pauses = 0
    for i in range(len(transcript.segments) - 1):
        curr = transcript.segments[i]
        nxt = transcript.segments[i + 1]
        if curr.speaker == Speaker.CUSTOMER and nxt.speaker == Speaker.OPERATOR:
            gap = nxt.start_time - curr.end_time
            if gap > _LONG_PAUSE_THRESHOLD:
                long_pauses += 1
                signals.append(f"Uzun fasilə: {gap:.0f}s")

    # Very short operator responses (minimal effort)
    op_segs = transcript.operator_segments
    if op_segs:
        avg_len = sum(len(s.text) for s in op_segs) / len(op_segs)
        if avg_len < 15:
            signals.append(f"Çox qısa cavablar (orta: {avg_len:.0f} simvol)")

    # Dismissive phrases
    dismissive = ["bilmirəm", "mümkün deyil", "edə bilmərəm", "bizə aid deyil"]
    op_lower = transcript.operator_text.lower()
    for phrase in dismissive:
        if phrase in op_lower:
            signals.append(f"Rədd edici: '{phrase}'")

    if long_pauses >= 2:
        return RuleResult(
            score=1, confidence=Confidence.LOW, needs_llm=True,
            reasoning="Uzun fasilələr və passiv davranış əlamətləri",
            signals=signals,
        )

    # Almost always needs LLM — "active help" is semantic
    return RuleResult(
        score=None, confidence=Confidence.LOW, needs_llm=True,
        reasoning="Fəal yardım semantik qiymətləndirmə tələb edir",
        signals=signals,
    )


# ═══════════════════════════════════════════════════════════
# KR2.3 — Product/service knowledge (almost fully LLM)
# ═══════════════════════════════════════════════════════════

def evaluate_kr23(transcript: CallTranscript) -> RuleResult:
    """KR2.3: Product knowledge — only basic pattern checks."""
    signals: list[str] = []
    operator_text = transcript.operator_text.lower()

    # Does operator mention specific numbers (prices, speeds, etc)?
    if any(char.isdigit() for char in operator_text):
        signals.append("Operator konkret rəqəmsal məlumat verib")

    # Uncertainty phrases
    uncertainty = ["dəqiq bilmirəm", "yəqin ki", "ola bilər", "zəng edib soruşun"]
    for phrase in uncertainty:
        if phrase in operator_text:
            signals.append(f"Qeyri-müəyyənlik: '{phrase}'")

    return RuleResult(
        score=None, confidence=Confidence.LOW, needs_llm=True,
        reasoning="Məhsul bilikləri semantik qiymətləndirmə tələb edir",
        signals=signals,
    )


# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════

_EVALUATORS = {
    "KR2.1": evaluate_kr21,
    "KR2.2": evaluate_kr22,
    "KR2.3": evaluate_kr23,
    "KR2.4": evaluate_kr24,
    "KR2.5": evaluate_kr25,
}


def evaluate_rules(criterion_id: str, transcript: CallTranscript) -> RuleResult:
    """Run rule-based evaluation for a given criterion."""
    evaluator = _EVALUATORS.get(criterion_id)
    if evaluator is None:
        logger.warning("No rule evaluator", extra={"criterion": criterion_id})
        return RuleResult(score=None, confidence=Confidence.LOW, needs_llm=True,
                          reasoning=f"No rules for {criterion_id}")

    result = evaluator(transcript)
    logger.info("Rule done", extra={
        "criterion": criterion_id,
        "score": result.score,
        "confidence": result.confidence.value,
        "needs_llm": result.needs_llm,
    })
    return result
