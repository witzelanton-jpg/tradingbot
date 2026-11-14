"""Tests fuer die Schutzmechanismen: Exposure, Ausbruch, Wiedereinstieg, Kill-Switch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hurstbot.config import AusbruchConfig, ExposureConfig, KillSwitchConfig, WiedereinstiegConfig
from hurstbot.guards import (
    WiedereinstiegZustand,
    berechne_atr,
    ist_kauf_durch_exposure_blockiert,
    ist_kill_switch_ausgeloest,
    pruefe_ausbruch,
    pruefe_wiedereinstieg,
    robuste_sigma_mad,
)


class TestExposureGuard:
    def test_blockiert_ueber_limit(self):
        config = ExposureConfig(max_exposure_prozent=40.0)
        assert ist_kauf_durch_exposure_blockiert(45.0, 100.0, config) is True

    def test_erlaubt_unter_limit(self):
        config = ExposureConfig(max_exposure_prozent=40.0)
        assert ist_kauf_durch_exposure_blockiert(30.0, 100.0, config) is False

    def test_grenzfall_genau_am_limit_blockiert(self):
        config = ExposureConfig(max_exposure_prozent=40.0)
        assert ist_kauf_durch_exposure_blockiert(40.0, 100.0, config) is True


class TestATR:
    def test_atr_ist_nicht_negativ(self):
        df = pd.DataFrame(
            {
                "high": [101, 103, 102, 105, 104],
                "low": [99, 100, 99, 101, 100],
                "close": [100, 102, 100, 104, 102],
            }
        )
        atr = berechne_atr(df, periode=3)
        assert (atr.dropna() >= 0).all()


class TestRobusteSigma:
    def test_mad_sigma_positiv_bei_streuung(self):
        renditen = pd.Series([0.1, -0.2, 0.15, -0.1, 0.3, -0.25])
        assert robuste_sigma_mad(renditen) > 0

    def test_mad_sigma_null_bei_konstanten_werten(self):
        renditen = pd.Series([0.1, 0.1, 0.1, 0.1])
        assert robuste_sigma_mad(renditen) == 0.0


class TestAusbruchswaechter:
    def test_kein_ausbruch_bei_ruhigem_markt(self):
        config = AusbruchConfig(ausbruch_faktor_atr=3.0, ausbruch_prozent=3.0, extrem_sigma_faktor=6.0)
        letzte_preise = pd.Series([100, 100.1, 99.9, 100.2, 100.0])
        ergebnis = pruefe_ausbruch(100.1, 100.0, letzte_preise, atr_wert=0.5, config=config)
        assert ergebnis.ausgeloest is False

    def test_ausbruch_bei_starker_prozentualer_abweichung(self):
        config = AusbruchConfig(ausbruch_faktor_atr=100.0, ausbruch_prozent=3.0, extrem_sigma_faktor=100.0)
        letzte_preise = pd.Series([100, 100, 100, 100, 108])
        ergebnis = pruefe_ausbruch(108.0, 100.0, letzte_preise, atr_wert=None, config=config)
        assert ergebnis.ausgeloest is True

    def test_ausbruch_bei_atr_vielfachem(self):
        config = AusbruchConfig(ausbruch_faktor_atr=2.0, ausbruch_prozent=100.0, extrem_sigma_faktor=100.0)
        ergebnis = pruefe_ausbruch(105.0, 100.0, pd.Series([100, 101, 100, 101, 105]), atr_wert=1.0, config=config)
        assert ergebnis.ausgeloest is True

    def test_ausbruch_bei_extremer_einzelkerze(self):
        config = AusbruchConfig(ausbruch_faktor_atr=100.0, ausbruch_prozent=100.0, extrem_sigma_faktor=3.0)
        letzte_preise = pd.Series([100, 100.1, 99.9, 100.1, 99.9, 100.0, 115.0])
        ergebnis = pruefe_ausbruch(115.0, 100.0, letzte_preise, atr_wert=None, config=config)
        assert ergebnis.ausgeloest is True


class TestWiedereinstieg:
    def test_sofortiger_wiedereinstieg_im_toleranzband(self):
        config = WiedereinstiegConfig(toleranzband_prozent=1.0, neuanker_aktiv=False, neuanker_timeout_stunden=100)
        zustand = WiedereinstiegZustand(pausiert_seit=pd.Timestamp("2024-01-01"), anker_preis_bei_pause=100.0)
        ergebnis = pruefe_wiedereinstieg(zustand, 100.5, pd.Timestamp("2024-01-01 00:05"), h_aktuell=0.3, h_start=0.45, config=config)
        assert ergebnis.wieder_einsteigen is True
        assert ergebnis.neuer_anker_preis == 100.0

    def test_kein_wiedereinstieg_ausserhalb_band_ohne_neuanker(self):
        config = WiedereinstiegConfig(toleranzband_prozent=1.0, neuanker_aktiv=False, neuanker_timeout_stunden=100)
        zustand = WiedereinstiegZustand(pausiert_seit=pd.Timestamp("2024-01-01"), anker_preis_bei_pause=100.0)
        ergebnis = pruefe_wiedereinstieg(zustand, 110.0, pd.Timestamp("2024-01-01 00:05"), h_aktuell=0.3, h_start=0.45, config=config)
        assert ergebnis.wieder_einsteigen is False

    def test_neuanker_nach_stabilen_kerzen(self):
        config = WiedereinstiegConfig(
            toleranzband_prozent=1.0, neuanker_aktiv=True, neuanker_kerzen=3, neuanker_timeout_stunden=1000
        )
        zustand = WiedereinstiegZustand(pausiert_seit=pd.Timestamp("2024-01-01"), anker_preis_bei_pause=100.0)
        zeit = pd.Timestamp("2024-01-01 01:00")
        # 2 % Abweichung: ausserhalb des 1 %-Toleranzbands, aber innerhalb der 3 %-Stabilitaetsspanne.
        for _ in range(3):
            ergebnis = pruefe_wiedereinstieg(zustand, 102.0, zeit, h_aktuell=0.3, h_start=0.45, config=config)
        assert ergebnis.wieder_einsteigen is True
        assert ergebnis.neuer_anker_preis == 102.0

    def test_instabile_kerze_setzt_zaehler_zurueck(self):
        config = WiedereinstiegConfig(
            toleranzband_prozent=1.0, neuanker_aktiv=True, neuanker_kerzen=3, neuanker_timeout_stunden=1000
        )
        zustand = WiedereinstiegZustand(pausiert_seit=pd.Timestamp("2024-01-01"), anker_preis_bei_pause=100.0)
        zeit = pd.Timestamp("2024-01-01 01:00")
        pruefe_wiedereinstieg(zustand, 102.0, zeit, h_aktuell=0.3, h_start=0.45, config=config)
        pruefe_wiedereinstieg(zustand, 102.0, zeit, h_aktuell=0.3, h_start=0.45, config=config)
        # H steigt kurz ueber H_START -> Kerze zaehlt nicht als stabil, Zaehler faellt zurueck auf 0
        pruefe_wiedereinstieg(zustand, 102.0, zeit, h_aktuell=0.5, h_start=0.45, config=config)
        assert zustand.stabile_kerzen_zaehler == 0

    def test_timeout_warnung_wird_einmalig_ausgegeben(self):
        config = WiedereinstiegConfig(
            toleranzband_prozent=1.0, neuanker_aktiv=False, neuanker_timeout_stunden=24
        )
        zustand = WiedereinstiegZustand(pausiert_seit=pd.Timestamp("2024-01-01"), anker_preis_bei_pause=100.0)
        spaeter = pd.Timestamp("2024-01-03")
        ergebnis_1 = pruefe_wiedereinstieg(zustand, 110.0, spaeter, h_aktuell=0.3, h_start=0.45, config=config)
        assert "WARNUNG" in ergebnis_1.grund
        ergebnis_2 = pruefe_wiedereinstieg(zustand, 110.0, spaeter, h_aktuell=0.3, h_start=0.45, config=config)
        assert "WARNUNG" not in ergebnis_2.grund


class TestKillSwitch:
    def test_ausgeloest_bei_grossem_verlust(self):
        config = KillSwitchConfig(max_verlust_prozent=10.0)
        assert ist_kill_switch_ausgeloest(85.0, 100.0, config) is True

    def test_nicht_ausgeloest_bei_kleinem_verlust(self):
        config = KillSwitchConfig(max_verlust_prozent=10.0)
        assert ist_kill_switch_ausgeloest(95.0, 100.0, config) is False

    def test_nicht_ausgeloest_bei_gewinn(self):
        config = KillSwitchConfig(max_verlust_prozent=10.0)
        assert ist_kill_switch_ausgeloest(120.0, 100.0, config) is False
