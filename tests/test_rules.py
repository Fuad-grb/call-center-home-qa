"""Tests for rule engine."""
import pytest
from app.models import CallTranscript, Confidence
from app.rule_engine import evaluate_rules


def _transcript(operator_texts, customer_texts=None):
    segs = []
    t = 0.0
    for ct in (customer_texts or ["Salam"]):
        segs.append({"speaker": "Customer", "text": ct, "start": t, "end": t + 2})
        t += 2.5
    for ot in operator_texts:
        segs.append({"speaker": "Operator", "text": ot, "start": t, "end": t + 3})
        t += 3.5
    return CallTranscript(call_id="T", segments=segs)


class TestKR25:
    def test_clean(self):
        r = evaluate_rules("KR2.5", _transcript(["Sizə necə kömək edim?"]))
        assert r.score == 3
        assert r.confidence == Confidence.MEDIUM

    def test_internal_leak(self):
        r = evaluate_rules("KR2.5", _transcript(["Micro donub, gözləyin."]))
        assert r.score == 0
        assert r.confidence == Confidence.HIGH
        assert r.needs_llm is False

    def test_personal_opinion(self):
        r = evaluate_rules("KR2.5", _transcript(["Şəxsən mən etməzdim."]))
        assert r.score == 0
        assert r.confidence == Confidence.HIGH

    def test_suspicious_needs_llm(self):
        r = evaluate_rules("KR2.5", _transcript(["Bilmirəm, gözləyin."]))
        assert r.needs_llm is True


class TestKR21:
    def test_needs_llm(self):
        r = evaluate_rules("KR2.1", _transcript(["Kömək edim."]))
        assert r.needs_llm is True
