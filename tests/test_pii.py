"""Tests for PII detection."""
import pytest
from app.pii_detector import detect_pii


class TestPII:
    def test_no_pii(self):
        assert detect_pii("Salam, necəsiniz?").has_pii is False

    def test_phone(self):
        r = detect_pii("Nömrəm +994501234567")
        assert r.has_pii is True
        assert "****" in r.masked_text

    def test_card(self):
        r = detect_pii("Kart: 4169 1234 5678 9010")
        assert r.has_pii is True

    def test_fin(self):
        r = detect_pii("FIN kodum 5TXKM4R")
        assert r.has_pii is True

    def test_price_not_pii(self):
        assert detect_pii("Qiymət 2500 AZN").has_pii is False

    def test_empty(self):
        assert detect_pii("").has_pii is False

    def test_multiple(self):
        r = detect_pii("FIN: A1B2C3D, Tel: +994501234567")
        assert len(r.matches) >= 2
