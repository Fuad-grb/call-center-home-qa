"""
LLM evaluation via Groq API.

Handles: prompt construction, API calls with retry, response parsing.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path

import yaml
from groq import Groq, APIError, RateLimitError

from app.logger import get_logger
from app.models import CallTranscript, LLMResult, RuleResult
from config.settings import settings

logger = get_logger(__name__)

# ── Load prompt templates ──

_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "criteria.yaml"
_PROMPTS: dict | None = None


def _get_prompts() -> dict:
    global _PROMPTS
    if _PROMPTS is None:
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
            _PROMPTS = yaml.safe_load(f)
    return _PROMPTS


# ── Groq client (singleton) ──

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env file.")
        _CLIENT = Groq(api_key=settings.groq_api_key)
    return _CLIENT


# ── Format transcript for LLM ──

def _format_transcript(transcript: CallTranscript) -> str:
    lines = []
    for seg in transcript.segments:
        if seg.is_empty:
            lines.append(f"[{seg.start_time:.1f}-{seg.end_time:.1f}] {seg.speaker.value}: [boş/anlaşılmaz]")
        else:
            lines.append(f"[{seg.start_time:.1f}-{seg.end_time:.1f}] {seg.speaker.value}: {seg.text}")
    return "\n".join(lines)


# ── Build prompt ──

def _build_prompt(criterion_id: str, transcript: CallTranscript, rule_result: RuleResult) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    prompts = _get_prompts()
    criterion = prompts["criteria"].get(criterion_id)
    if criterion is None:
        raise ValueError(f"Unknown criterion: {criterion_id}")

    scoring = criterion["scoring"]

    if rule_result.signals:
        rule_signals = "Rule-based analiz aşağıdakıları aşkarladı:\n" + "\n".join(
            f"  - {s}" for s in rule_result.signals
        )
    else:
        rule_signals = "Rule-based analiz heç bir aşkar problem tapmadı."

    user_prompt = prompts["eval_prompt_template"].format(
        criterion_name=criterion["name"],
        criterion_description=criterion["description"],
        score_3=scoring[3],
        score_2=scoring[2],
        score_1=scoring[1],
        score_0=scoring[0],
        rule_signals=rule_signals,
        transcript=_format_transcript(transcript),
    )

    return prompts["system_prompt"], user_prompt


# ── Parse LLM response ──

def _parse_llm_response(raw_text: str) -> LLMResult:
    """Extract JSON from LLM response, handle markdown wrappers."""
    text = raw_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(f"Cannot parse LLM response: {raw_text[:200]}")
        else:
            raise ValueError(f"No JSON in LLM response: {raw_text[:200]}")

    score = data.get("score")
    if score is None or not isinstance(score, (int, float)):
        raise ValueError(f"Invalid 'score' in LLM response: {data}")

    score = int(score)
    score = max(0, min(3, score))  # clamp to 0-3

    reasoning = str(data.get("reasoning", ""))

    return LLMResult(score=score, reasoning=reasoning)


# ── API call with retry ──

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()

    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            wait = 2 ** attempt
            logger.warning("Rate limited", extra={"attempt": attempt, "wait": wait})
            time.sleep(wait)

        except APIError as e:
            logger.error("Groq API error", extra={"attempt": attempt, "error": str(e)})
            if attempt == settings.llm_max_retries:
                raise
            time.sleep(1)

    raise RuntimeError(f"LLM failed after {settings.llm_max_retries} retries")


# ── Public API ──

def evaluate_with_llm(
    criterion_id: str,
    transcript: CallTranscript,
    rule_result: RuleResult,
) -> LLMResult:
    """Evaluate one criterion using LLM."""
    system_prompt, user_prompt = _build_prompt(criterion_id, transcript, rule_result)

    logger.info("Calling LLM", extra={
        "criterion": criterion_id,
        "call_id": transcript.call_id,
    })

    raw = _call_llm(system_prompt, user_prompt)
    result = _parse_llm_response(raw)

    logger.info("LLM done", extra={
        "criterion": criterion_id,
        "score": result.score,
    })

    return result
