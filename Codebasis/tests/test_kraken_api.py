"""Tests fuer die Kraken-API-Huelle: Retries, Backoff, Signierung, Fehlerbehandlung.

Alle Antworten sind gemockt - es wird nie die echte API kontaktiert.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
import requests

from hurstbot.config import KrakenApiConfig
from hurstbot.kraken_api import KrakenAPI, KrakenAPIError, KrakenVerbindungsFehler

CONFIG = KrakenApiConfig(max_wiederholungen=3, backoff_basis_sekunden=0.0, anfragen_pro_sekunde=0.0)


def _antwort(json_daten: dict, status: int = 200) -> MagicMock:
    antwort = MagicMock()
    antwort.json.return_value = json_daten
    antwort.status_code = status
    antwort.raise_for_status.return_value = None
    return antwort


class TestOeffentlicheEndpunkte:
    def test_ticker_liefert_result(self):
        session = MagicMock()
        session.get.return_value = _antwort({"error": [], "result": {"XXBTZEUR": {"c": ["50000.0", "0.1"]}}})
        api = KrakenAPI(CONFIG, session=session)

        ergebnis = api.ticker(["XXBTZEUR"])

        assert ergebnis["XXBTZEUR"]["c"][0] == "50000.0"

    def test_mehrere_paare_werden_komma_separiert(self):
        session = MagicMock()
        session.get.return_value = _antwort({"error": [], "result": {}})
        api = KrakenAPI(CONFIG, session=session)

        api.ticker(["XXBTZEUR", "XDGEUR"])

        _, kwargs = session.get.call_args
        assert kwargs["params"]["pair"] == "XXBTZEUR,XDGEUR"

    def test_ohlc_uebergibt_since(self):
        session = MagicMock()
        session.get.return_value = _antwort({"error": [], "result": {"XXBTZEUR": [], "last": 123}})
        api = KrakenAPI(CONFIG, session=session)

        api.ohlc("XXBTZEUR", 5, since=1700000000)

        _, kwargs = session.get.call_args
        assert kwargs["params"]["since"] == 1700000000
        assert kwargs["params"]["interval"] == 5


class TestFehlerbehandlung:
    def test_fachlicher_fehler_wird_nicht_wiederholt(self):
        session = MagicMock()
        session.get.return_value = _antwort({"error": ["EQuery:Unknown asset pair"], "result": {}})
        api = KrakenAPI(CONFIG, session=session)

        with pytest.raises(KrakenAPIError, match="Unknown asset pair"):
            api.ticker(["QUATSCHEUR"])

        assert session.get.call_count == 1  # kein Retry bei dauerhaftem Fehler

    @patch("hurstbot.kraken_api.time.sleep")
    def test_rate_limit_wird_wiederholt(self, _sleep):
        session = MagicMock()
        session.get.side_effect = [
            _antwort({"error": ["EAPI:Rate limit exceeded"], "result": {}}),
            _antwort({"error": [], "result": {"XXBTZEUR": {"c": ["1.0", "1"]}}}),
        ]
        api = KrakenAPI(CONFIG, session=session)

        ergebnis = api.ticker(["XXBTZEUR"])

        assert session.get.call_count == 2
        assert "XXBTZEUR" in ergebnis

    @patch("hurstbot.kraken_api.time.sleep")
    def test_netzwerkfehler_wird_wiederholt_dann_aufgegeben(self, _sleep):
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("Netz weg")
        api = KrakenAPI(CONFIG, session=session)

        with pytest.raises(KrakenVerbindungsFehler):
            api.ticker(["XXBTZEUR"])

        assert session.get.call_count == CONFIG.max_wiederholungen

    @patch("hurstbot.kraken_api.time.sleep")
    def test_erholung_nach_voruebergehendem_netzwerkfehler(self, _sleep):
        session = MagicMock()
        session.get.side_effect = [
            requests.Timeout("zu langsam"),
            _antwort({"error": [], "result": {"XXBTZEUR": {"c": ["1.0", "1"]}}}),
        ]
        api = KrakenAPI(CONFIG, session=session)

        ergebnis = api.ticker(["XXBTZEUR"])

        assert "XXBTZEUR" in ergebnis


class TestPrivateEndpunkte:
    def test_ohne_schluessel_wird_abgelehnt(self):
        api = KrakenAPI(CONFIG, session=MagicMock())

        with pytest.raises(KrakenAPIError, match="KRAKEN_API_KEY"):
            api.balance()

    def test_signierung_setzt_erwartete_header(self):
        session = MagicMock()
        session.post.return_value = _antwort({"error": [], "result": {"ZEUR": "100.0"}})
        secret = base64.b64encode(b"test-secret-bytes").decode()
        api = KrakenAPI(CONFIG, api_key="test-key", api_secret=secret, session=session)

        api.balance()

        _, kwargs = session.post.call_args
        assert kwargs["headers"]["API-Key"] == "test-key"
        assert kwargs["headers"]["API-Sign"]  # Signatur wurde berechnet
        assert "nonce" in kwargs["data"]

    def test_add_order_mit_validate_setzt_flag(self):
        session = MagicMock()
        session.post.return_value = _antwort({"error": [], "result": {"descr": {}}})
        secret = base64.b64encode(b"test-secret-bytes").decode()
        api = KrakenAPI(CONFIG, api_key="k", api_secret=secret, session=session)

        api.add_order("XXBTZEUR", "buy", "limit", 0.001, preis=50000.0, validate=True)

        _, kwargs = session.post.call_args
        assert kwargs["data"]["validate"] == "true"
        assert kwargs["data"]["pair"] == "XXBTZEUR"
        assert kwargs["data"]["type"] == "buy"

    def test_add_order_ohne_validate_setzt_kein_flag(self):
        session = MagicMock()
        session.post.return_value = _antwort({"error": [], "result": {"txid": ["ABC"]}})
        secret = base64.b64encode(b"test-secret-bytes").decode()
        api = KrakenAPI(CONFIG, api_key="k", api_secret=secret, session=session)

        api.add_order("XXBTZEUR", "buy", "limit", 0.001, preis=50000.0, validate=False)

        _, kwargs = session.post.call_args
        assert "validate" not in kwargs["data"]
