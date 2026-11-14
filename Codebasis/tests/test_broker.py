"""Tests fuer die Broker-Schnittstellen - insbesondere die Live-Sicherheitsschranken."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from hurstbot.broker import (
    KrakenLiveBroker,
    KrakenValidateBroker,
    LiveModusNichtErlaubt,
    PaarPraezision,
    PaperBroker,
)
from hurstbot.config import GebuehrenConfig

GEBUEHREN = GebuehrenConfig(taker_gebuehr_prozent=0.26, geschaetzter_spread_prozent=0.1)
PRAEZISION = PaarPraezision(preis_dezimalstellen=2, mengen_dezimalstellen=8, mindest_ordervolumen=0.0001)


@pytest.fixture(autouse=True)
def _ohne_live_confirm_env(monkeypatch):
    monkeypatch.delenv("KRAKEN_LIVE_CONFIRM", raising=False)


class TestPaperBroker:
    def test_kauf_liefert_plausible_menge_und_gebuehr(self):
        broker = PaperBroker(GEBUEHREN)
        ausfuehrung = broker.kaufe("XBTEUR", 100.0, 50000.0, PRAEZISION)
        assert ausfuehrung.erfolgreich is True
        assert ausfuehrung.fill_preis > 50000.0  # Kauf-Fill liegt ueber Referenz (Spread)
        assert ausfuehrung.gebuehr_eur > 0
        assert ausfuehrung.menge > 0

    def test_verkauf_fill_liegt_unter_referenz(self):
        broker = PaperBroker(GEBUEHREN)
        ausfuehrung = broker.verkaufe("XBTEUR", 0.01, 50000.0, PRAEZISION)
        assert ausfuehrung.fill_preis < 50000.0

    def test_unter_mindestordervolumen_schlaegt_fehl(self):
        broker = PaperBroker(GEBUEHREN)
        praezision = PaarPraezision(preis_dezimalstellen=2, mengen_dezimalstellen=8, mindest_ordervolumen=1.0)
        ausfuehrung = broker.kaufe("XBTEUR", 10.0, 50000.0, praezision)
        assert ausfuehrung.erfolgreich is False


class TestKrakenValidateBroker:
    def test_validate_ruft_api_mit_validate_true_auf(self):
        fake_api = MagicMock()
        fake_api.add_order.return_value = {}
        broker = KrakenValidateBroker(fake_api, GEBUEHREN)
        ausfuehrung = broker.kaufe("XBTEUR", 100.0, 50000.0, PRAEZISION)

        assert ausfuehrung.erfolgreich is True
        fake_api.add_order.assert_called_once()
        _, kwargs = fake_api.add_order.call_args
        assert kwargs.get("validate") is True


class TestKrakenLiveBrokerSicherheit:
    """Live-Modus darf nur mit ALLEN DREI Freigaben moeglich sein."""

    def test_verweigert_ohne_jegliche_freigabe(self):
        with pytest.raises(LiveModusNichtErlaubt):
            KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=False, nutzer_bestaetigung="")

    def test_verweigert_ohne_live_flag(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        with pytest.raises(LiveModusNichtErlaubt):
            KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=False, nutzer_bestaetigung="JA")

    def test_verweigert_ohne_env_var(self):
        with pytest.raises(LiveModusNichtErlaubt):
            KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=True, nutzer_bestaetigung="JA")

    def test_verweigert_ohne_nutzerbestaetigung(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        with pytest.raises(LiveModusNichtErlaubt):
            KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=True, nutzer_bestaetigung="")

    def test_verweigert_bei_falscher_nutzerbestaetigung(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        with pytest.raises(LiveModusNichtErlaubt):
            KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=True, nutzer_bestaetigung="vielleicht")

    def test_erlaubt_mit_allen_drei_freigaben(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        broker = KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=True, nutzer_bestaetigung="ja")
        assert isinstance(broker, KrakenLiveBroker)

    def test_env_var_falscher_wert_verweigert(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "ja_klar")  # nicht exakt "JA"
        with pytest.raises(LiveModusNichtErlaubt):
            KrakenLiveBroker(MagicMock(), GEBUEHREN, live_flag_gesetzt=True, nutzer_bestaetigung="JA")
