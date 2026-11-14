"""Tests fuer die atomare Zustandspersistenz."""

from __future__ import annotations

import pandas as pd
import pytest

from hurstbot.backtest import Lot
from hurstbot.state import BotZustand, PaarZustandsDaten, lade_zustand, speichere_zustand


class TestRoundtrip:
    def test_speichern_und_laden_erhaelt_lots(self, tmp_path):
        pfad = tmp_path / "zustand.json"
        lot = Lot(einstandspreis=101.5, fill_preis_kauf=100.0, menge=0.5, kauf_zeit=pd.Timestamp("2024-01-01T12:00:00"))
        paar_daten = PaarZustandsDaten(paar="XBTEUR", zustand="HANDELT", lots=[lot], kapital_eur=50.0, anker_preis=100.0)
        zustand = BotZustand(start_kapital_eur=100.0, kapital_frei_eur=50.0, paare={"XBTEUR": paar_daten})

        speichere_zustand(zustand, pfad)
        geladen = lade_zustand(pfad)

        assert geladen is not None
        assert geladen.start_kapital_eur == 100.0
        assert geladen.paare["XBTEUR"].zustand == "HANDELT"
        assert len(geladen.paare["XBTEUR"].lots) == 1
        geladenes_lot = geladen.paare["XBTEUR"].lots[0]
        assert geladenes_lot.einstandspreis == pytest.approx(101.5)
        assert geladenes_lot.kauf_zeit == pd.Timestamp("2024-01-01T12:00:00")

    def test_datei_existiert_nach_schreiben_nicht_als_tmp(self, tmp_path):
        pfad = tmp_path / "zustand.json"
        zustand = BotZustand(start_kapital_eur=100.0, kapital_frei_eur=100.0)
        speichere_zustand(zustand, pfad)
        dateien = list(tmp_path.iterdir())
        assert pfad in dateien
        assert all(not str(d).endswith(".tmp") for d in dateien if "tmp." not in d.name or True)
        assert not any(f.name.startswith("zustand.json.tmp") for f in dateien)


class TestFehlerfaelle:
    def test_fehlende_datei_liefert_none(self, tmp_path):
        assert lade_zustand(tmp_path / "nicht_da.json") is None

    def test_korrupte_datei_liefert_none_statt_absturz(self, tmp_path):
        pfad = tmp_path / "kaputt.json"
        pfad.write_text("{ das ist kein gueltiges JSON", encoding="utf-8")
        assert lade_zustand(pfad) is None
