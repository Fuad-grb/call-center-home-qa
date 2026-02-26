"""Tests for input validation."""
import pytest
from app.validators import validate_input


class TestValidateInput:
    def test_valid_input(self):
        data = {
            "call_id": "TEST",
            "segments": [
                {"speaker": "Operator", "text": "Salam", "start": 0.0, "end": 2.0},
                {"speaker": "Customer", "text": "Salam", "start": 2.5, "end": 4.0},
            ],
        }
        result = validate_input(data)
        assert result.is_valid is True
        assert result.transcript.call_id == "TEST"

    def test_missing_call_id(self):
        result = validate_input({"segments": [{"speaker": "Operator", "text": "Hi", "start": 0, "end": 1}]})
        assert result.is_valid is False

    def test_empty_segments(self):
        result = validate_input({"call_id": "T", "segments": []})
        assert result.is_valid is False

    def test_not_a_dict(self):
        result = validate_input("string")
        assert result.is_valid is False

    def test_empty_text_warning(self):
        data = {
            "call_id": "T",
            "segments": [
                {"speaker": "Operator", "text": "", "start": 0.0, "end": 1.0},
                {"speaker": "Customer", "text": "Salam", "start": 1.5, "end": 3.0},
            ],
        }
        result = validate_input(data)
        assert result.is_valid is True
        assert any("empty" in w.lower() for w in result.warnings)

    def test_start_time_alias(self):
        data = {
            "call_id": "T",
            "segments": [{"speaker": "Operator", "text": "Salam", "start_time": 0.0, "end_time": 2.0}],
        }
        result = validate_input(data)
        assert result.is_valid is True

    def test_no_operator_warning(self):
        data = {
            "call_id": "T",
            "segments": [{"speaker": "Customer", "text": "Salam", "start": 0.0, "end": 2.0}],
        }
        result = validate_input(data)
        assert any("operator" in w.lower() for w in result.warnings)
