"""Tests fuer das Laden und Cachen der OHLC-Daten (gegen gemockte Kraken-Antworten)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from hurstbot.config import MultiSkalenConfig
from hurstbot.data import _parse_ohlc_antwort, cache_pfad, hole_multiskalen_ohlc, hole_ohlc
from hurstbot.kraken_api import KrakenVerbindungsFehler


class TestParseOhlcAntwort:
    def test_wandelt_strings_in_zahlen_und_zeitstempel(self, ohlc_antwort):
        df = _parse_ohlc_antwort(ohlc_antwort(anzahl=10))

        assert len(df) == 10
        assert df["close"].dtype == float
        assert str(df["zeit"].dtype).startswith("datetime64")
        assert df["zeit"].is_monotonic_increasing

    def test_leere_antwort_liefert_leeres_dataframe(self):
        df = _parse_ohlc_antwort({"XXBTZEUR": [], "last": 0})
        assert df.empty

    def test_doppelte_zeitstempel_werden_entfernt(self):
        kerze = [1700000000, "100", "101", "99", "100", "100", "1", 5]
        df = _parse_ohlc_antwort({"XXBTZEUR": [kerze, kerze], "last": 1700000000})
        assert len(df) == 1


class TestHoleOhlc:
    def test_laedt_und_schreibt_cache(self, tmp_path, ohlc_antwort):
        api = MagicMock()
        api.ohlc.return_value = ohlc_antwort(anzahl=50)

        df = hole_ohlc(api, "XXBTZEUR", 5, tmp_path)

        assert len(df) == 50
        assert cache_pfad(tmp_path, "XXBTZEUR", 5).exists()

    def test_frischer_cache_vermeidet_zweite_abfrage(self, tmp_path, ohlc_antwort):
        api = MagicMock()
        api.ohlc.return_value = ohlc_antwort(anzahl=50)

        hole_ohlc(api, "XXBTZEUR", 5, tmp_path, max_alter_sekunden=10**9)
        hole_ohlc(api, "XXBTZEUR", 5, tmp_path, max_alter_sekunden=10**9)

        assert api.ohlc.call_count == 1  # zweiter Aufruf kam aus dem Cache

    def test_alter_cache_loest_nachladen_mit_since_aus(self, tmp_path, ohlc_antwort):
        api = MagicMock()
        api.ohlc.return_value = ohlc_antwort(anzahl=50)

        hole_ohlc(api, "XXBTZEUR", 5, tmp_path, max_alter_sekunden=10**9)
        api.ohlc.reset_mock()
        hole_ohlc(api, "XXBTZEUR", 5, tmp_path, max_alter_sekunden=0)

        api.ohlc.assert_called_once()
        _, kwargs = api.ohlc.call_args
        assert kwargs.get("since") is not None

    def test_cache_wird_mit_neuen_kerzen_zusammengefuehrt(self, tmp_path, ohlc_antwort):
        api = MagicMock()
        erste_antwort = ohlc_antwort(anzahl=30)
        api.ohlc.return_value = erste_antwort
        hole_ohlc(api, "XXBTZEUR", 5, tmp_path, max_alter_sekunden=10**9)

        # Zweite Antwort enthaelt eine zusaetzliche, spaetere Kerze.
        letzte_zeit = erste_antwort["XXBTZEUR"][-1][0]
        neue_kerze = [letzte_zeit + 300, "105", "106", "104", "105", "105", "2", 7]
        api.ohlc.return_value = {"XXBTZEUR": [neue_kerze], "last": neue_kerze[0]}

        df = hole_ohlc(api, "XXBTZEUR", 5, tmp_path, max_alter_sekunden=0)

        assert len(df) == 31
        assert df["zeit"].is_monotonic_increasing

    def test_benutze_cache_false_schreibt_keine_datei(self, tmp_path, ohlc_antwort):
        api = MagicMock()
        api.ohlc.return_value = ohlc_antwort(anzahl=20)

        hole_ohlc(api, "XXBTZEUR", 5, tmp_path, benutze_cache=False)

        assert not cache_pfad(tmp_path, "XXBTZEUR", 5).exists()


class TestHoleMultiskalenOhlc:
    def test_laedt_alle_drei_skalen(self, tmp_path, ohlc_antwort):
        api = MagicMock()
        api.ohlc.side_effect = lambda paar, intervall, since=None: ohlc_antwort(
            paar=paar, anzahl=250, intervall_minuten=intervall
        )
        config = MultiSkalenConfig(kurz_anzahl_kerzen=200, mittel_anzahl_kerzen=200, lang_anzahl_kerzen=200)

        daten = hole_multiskalen_ohlc(api, "XXBTZEUR", config, tmp_path)

        assert set(daten) == {"kurz", "mittel", "lang"}
        assert len(daten["kurz"]) == 200  # auf konfigurierte Anzahl gekuerzt

    def test_fehlende_langzeitdaten_brechen_nicht_ab(self, tmp_path, ohlc_antwort):
        """Die Langzeit-Skala ist nur ein Zusatzscore - ihr Ausfall darf den Scan nicht stoppen."""
        api = MagicMock()

        def _ohlc(paar, intervall, since=None):
            if intervall == 1440:
                raise KrakenVerbindungsFehler("Tageskerzen gerade nicht verfuegbar")
            return ohlc_antwort(paar=paar, anzahl=250, intervall_minuten=intervall)

        api.ohlc.side_effect = _ohlc

        daten = hole_multiskalen_ohlc(api, "XXBTZEUR", MultiSkalenConfig(), tmp_path)

        assert not daten["kurz"].empty
        assert not daten["mittel"].empty
        assert daten["lang"].empty
