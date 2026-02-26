"""
Hybrid evaluation pipeline orchestrator.

Flow per call:
1. Validate input
2. Detect PII
3. For each of 5 criteria:
   a. Rule-based check
   b. If rule confidence HIGH → skip LLM
   c. Else → call LLM
   d. Combine → final score + probability
4. Return structured output
"""

from __future__ import annotations
from typing import Any

from app.logger import get_logger
from app.models import (
    CallEvaluation, CallTranscript, Confidence,
    CriterionID, CriterionScore, RuleResult,
)
from app.pii_detector import detect_pii
from app.rule_engine import evaluate_rules
from app.llm_evaluator import evaluate_with_llm
from app.validators import validate_input

logger = get_logger(__name__)

_ALL_CRITERIA = [c.value for c in CriterionID]


def _determine_probability(
    rule_result: RuleResult,
    llm_used: bool,
    llm_score: int | None,
) -> Confidence:
    """
    How confident are we in the final score?
    - Rule HIGH + no LLM → HIGH
    - Rule + LLM agree → HIGH
    - LLM only → MEDIUM
    - Rule + LLM disagree → LOW
    """
    if not llm_used:
        return rule_result.confidence

    if rule_result.score is not None and llm_score is not None:
        if rule_result.score == llm_score:
            return Confidence.HIGH
        else:
            return Confidence.LOW

    return Confidence.MEDIUM


def _combine_scores(
    rule_result: RuleResult,
    llm_score: int | None,
    llm_reasoning: str,
    llm_used: bool,
) -> tuple[int, str]:
    """
    Merge rule and LLM into final score + reasoning.
    LLM takes priority when used (it understands semantics).
    """
    if not llm_used and rule_result.score is not None:
        return rule_result.score, rule_result.reasoning

    if llm_score is not None:
        reasoning = llm_reasoning
        if rule_result.signals:
            reasoning += f" [Rule: {', '.join(rule_result.signals[:3])}]"
        return llm_score, reasoning

    # Fallback — shouldn't normally happen
    logger.warning("No score from rule or LLM, defaulting to 2")
    return 2, "Qiymətləndirmə üçün kifayət qədər məlumat yoxdur"


def evaluate_call(raw_input: dict[str, Any]) -> CallEvaluation:
    """
    Main entry point: evaluate one call transcript.
    """
    # ── Step 1: Validate ──
    validation = validate_input(raw_input)
    if not validation.is_valid:
        logger.error("Invalid input", extra={"warnings": validation.warnings})
        return CallEvaluation(
            call_id=raw_input.get("call_id", "UNKNOWN"),
            scores={},
            warnings=validation.warnings,
        )

    transcript = validation.transcript
    warnings = list(validation.warnings)

    # ── Step 2: PII Detection ──
    pii_detected = False
    for seg in transcript.segments:
        pii_result = detect_pii(seg.text)
        if pii_result.has_pii:
            pii_detected = True
            warnings.append(f"PII in segment [{seg.start_time}-{seg.end_time}]")

    # ── Step 3: Evaluate each criterion ──
    scores: dict[str, CriterionScore] = {}

    for criterion_id in _ALL_CRITERIA:
        logger.info("Evaluating", extra={
            "call_id": transcript.call_id,
            "criterion": criterion_id,
        })

        # 3a. Rule-based
        rule_result = evaluate_rules(criterion_id, transcript)

        # 3b. Skip LLM?
        llm_used = False
        llm_score = None
        llm_reasoning = ""

        skip_llm = (
            rule_result.confidence == Confidence.HIGH
            and not rule_result.needs_llm
            and rule_result.score is not None
        )

        if skip_llm:
            logger.info("Skipping LLM — rule HIGH", extra={
                "criterion": criterion_id,
                "rule_score": rule_result.score,
            })
        else:
            # 3c. Call LLM
            try:
                llm_result = evaluate_with_llm(criterion_id, transcript, rule_result)
                llm_used = True
                llm_score = llm_result.score
                llm_reasoning = llm_result.reasoning
            except Exception as e:
                logger.error("LLM failed", extra={
                    "criterion": criterion_id,
                    "error": str(e),
                })
                warnings.append(f"LLM failed for {criterion_id}: {e}")
                if rule_result.score is None:
                    llm_score = 2
                    llm_reasoning = "LLM xətası — standart qiymət"

        # 3d. Combine
        final_score, final_reasoning = _combine_scores(
            rule_result, llm_score, llm_reasoning, llm_used
        )
        probability = _determine_probability(rule_result, llm_used, llm_score)

        scores[criterion_id] = CriterionScore(
            score=final_score,
            reasoning=final_reasoning,
            probability=probability,
        )

    return CallEvaluation(
        call_id=transcript.call_id,
        scores=scores,
        pii_detected=pii_detected,
        warnings=warnings,
    )
