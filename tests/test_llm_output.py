"""Tests for LLM response parsing (no API calls)."""
import pytest
from app.llm_evaluator import _parse_llm_response


class TestParsing:
    def test_clean_json(self):
        r = _parse_llm_response('{"score": 3, "reasoning": "Yaxşı"}')
        assert r.score == 3

    def test_code_block(self):
        r = _parse_llm_response('```json\n{"score": 2, "reasoning": "Ok"}\n```')
        assert r.score == 2

    def test_extra_text(self):
        r = _parse_llm_response('Here:\n{"score": 1, "reasoning": "Zəif"}\nDone.')
        assert r.score == 1

    def test_clamp_high(self):
        r = _parse_llm_response('{"score": 5, "reasoning": "t"}')
        assert r.score == 3

    def test_clamp_low(self):
        r = _parse_llm_response('{"score": -1, "reasoning": "t"}')
        assert r.score == 0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_llm_response("not json")

    def test_missing_score_raises(self):
        with pytest.raises(ValueError):
            _parse_llm_response('{"reasoning": "no score"}')
