"""Tests fuer die Paar-Strategie: Zustandsautomat, Lot-Handel, Waechter-Integration."""

from __future__ import annotations

import pandas as pd
import pytest

from hurstbot.broker import Ausfuehrung, Broker, PaarPraezision
from hurstbot.config import AppConfig
from hurstbot.guards import PaarZustand
from hurstbot.hurst import HurstErgebnis
from hurstbot.regime import RegimeErgebnis, SkalenErgebnis
from hurstbot.strategy import PaarStrategie

PRAEZISION = PaarPraezision(preis_dezimalstellen=2, mengen_dezimalstellen=8, mindest_ordervolumen=0.0001)


class FakeBroker(Broker):
    def __init__(self):
        self.kauf_aufrufe: list[tuple] = []
        self.verkauf_aufrufe: list[tuple] = []
        self.naechster_kauf_erfolg = True
        self.naechster_verkauf_erfolg = True

    def kaufe(self, paar, menge_quote_eur, referenzpreis, praezision):
        self.kauf_aufrufe.append((paar, menge_quote_eur, referenzpreis))
        if not self.naechster_kauf_erfolg:
            return Ausfuehrung(False, referenzpreis, 0.0, 0.0, None, "abgelehnt")
        fill_preis = referenzpreis * 1.0005
        menge = menge_quote_eur / fill_preis
        gebuehr = menge * fill_preis * 0.002
        return Ausfuehrung(True, fill_preis, menge, gebuehr, "FAKE-KAUF", "ok")

    def verkaufe(self, paar, menge_basis, referenzpreis, praezision):
        self.verkauf_aufrufe.append((paar, menge_basis, referenzpreis))
        if not self.naechster_verkauf_erfolg:
            return Ausfuehrung(False, referenzpreis, 0.0, 0.0, None, "abgelehnt")
        fill_preis = referenzpreis * 0.9995
        gebuehr = menge_basis * fill_preis * 0.002
        return Ausfuehrung(True, fill_preis, menge_basis, gebuehr, "FAKE-VERKAUF", "ok")


def _regime(handel_erlaubt=True, score=0.8, h=0.3, konfidenz=0.9, grund="test"):
    hurst = HurstErgebnis(h=h, ci_untere_grenze=h - 0.05, ci_obere_grenze=h + 0.05, konfidenz=konfidenz)
    skala = SkalenErgebnis(hurst=hurst, monotonie={"monotonie_score": 0.1})
    return RegimeErgebnis(kurz=skala, mittel=skala, lang=None, oszillations_score=score, handel_erlaubt=handel_erlaubt, grund=grund)


def _strategie(broker, kapital=1000.0, x_prozent=2.0) -> PaarStrategie:
    config = AppConfig()
    return PaarStrategie(paar="XBTEUR", config=config, broker=broker, praezision=PRAEZISION, kapital_eur=kapital, x_prozent=x_prozent)


class TestKaufUndVerkauf:
    def test_kauf_bei_schwellenunterschreitung(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        zeit = pd.Timestamp("2024-01-01")

        strategie.verarbeite_tick(100.0, zeit, _regime(), pd.Series([100.0]), None, gesamtkapital_bot=1000.0)
        assert broker.kauf_aufrufe == []  # erster Tick setzt nur den Anker

        strategie.verarbeite_tick(97.5, zeit + pd.Timedelta(minutes=1), _regime(), pd.Series([100.0, 97.5]), None, gesamtkapital_bot=1000.0)
        assert len(broker.kauf_aufrufe) == 1
        assert len(strategie.lots) == 1
        assert strategie.zustand == PaarZustand.HANDELT

    def test_verkauf_bei_profitabler_schwellenueberschreitung(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        zeit = pd.Timestamp("2024-01-01")

        strategie.verarbeite_tick(100.0, zeit, _regime(), pd.Series([100.0]), None, gesamtkapital_bot=1000.0)
        strategie.verarbeite_tick(97.5, zeit, _regime(), pd.Series([100.0, 97.5]), None, gesamtkapital_bot=1000.0)
        assert len(strategie.lots) == 1

        strategie.verarbeite_tick(100.2, zeit, _regime(), pd.Series([97.5, 100.2]), None, gesamtkapital_bot=1000.0)
        assert strategie.lots == []
        assert len(broker.verkauf_aufrufe) == 1

    def test_kein_kauf_bei_nur_verkaufen_wenn_gate_verloren(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        zeit = pd.Timestamp("2024-01-01")

        strategie.verarbeite_tick(100.0, zeit, _regime(handel_erlaubt=False), pd.Series([100.0]), None, gesamtkapital_bot=1000.0)
        strategie.verarbeite_tick(97.5, zeit, _regime(handel_erlaubt=False), pd.Series([100.0, 97.5]), None, gesamtkapital_bot=1000.0)

        assert broker.kauf_aufrufe == []
        assert strategie.zustand == PaarZustand.NUR_VERKAUFEN

    def test_bestehende_lots_werden_in_nur_verkaufen_weiter_verkauft(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        zeit = pd.Timestamp("2024-01-01")
        strategie.verarbeite_tick(100.0, zeit, _regime(), pd.Series([100.0]), None, gesamtkapital_bot=1000.0)
        strategie.verarbeite_tick(97.5, zeit, _regime(), pd.Series([100.0, 97.5]), None, gesamtkapital_bot=1000.0)
        assert len(strategie.lots) == 1

        # Gate jetzt verloren, aber das bestehende Lot ist profitabel -> darf verkauft werden.
        strategie.verarbeite_tick(100.2, zeit, _regime(handel_erlaubt=False), pd.Series([97.5, 100.2]), None, gesamtkapital_bot=1000.0)
        assert strategie.lots == []
        assert len(broker.verkauf_aufrufe) == 1
        assert strategie.zustand == PaarZustand.NUR_VERKAUFEN


class TestExposureGuard:
    def test_exposure_limit_blockiert_weitere_kaeufe(self):
        broker = FakeBroker()
        strategie = _strategie(broker, kapital=1000.0)
        # Schon stark investiert: 1 Lot mit hohem Wert relativ zum Gesamtkapital.
        from hurstbot.backtest import Lot

        strategie.lots.append(Lot(einstandspreis=100.0, fill_preis_kauf=100.0, menge=5.0, kauf_zeit=pd.Timestamp("2024-01-01")))
        strategie.anker_preis = 100.0

        # investierter Wert = 5*97.5=487.5, bei gesamtkapital_bot=1000 -> 48.75 % > 40 % Limit
        strategie.verarbeite_tick(97.5, pd.Timestamp("2024-01-01"), _regime(), pd.Series([100.0, 97.5]), None, gesamtkapital_bot=1000.0)

        assert broker.kauf_aufrufe == []
        assert strategie.zustand == PaarZustand.NUR_VERKAUFEN


class TestAusbruchswaechter:
    def test_starker_ausbruch_pausiert_und_blockiert_alles(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        zeit = pd.Timestamp("2024-01-01")
        strategie.verarbeite_tick(100.0, zeit, _regime(), pd.Series([100.0]), None, gesamtkapital_bot=1000.0)

        # 10 % Sprung -> deutlich ueber dem Default-Ausbruchsprozent (3 %).
        strategie.verarbeite_tick(90.0, zeit + pd.Timedelta(minutes=1), _regime(), pd.Series([100.0, 90.0]), None, gesamtkapital_bot=1000.0)

        assert strategie.zustand == PaarZustand.PAUSIERT
        assert broker.kauf_aufrufe == []
        assert broker.verkauf_aufrufe == []

    def test_wiedereinstieg_nach_rueckkehr_ins_toleranzband(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        zeit = pd.Timestamp("2024-01-01")
        strategie.verarbeite_tick(100.0, zeit, _regime(), pd.Series([100.0]), None, gesamtkapital_bot=1000.0)
        strategie.verarbeite_tick(90.0, zeit + pd.Timedelta(minutes=1), _regime(), pd.Series([100.0, 90.0]), None, gesamtkapital_bot=1000.0)
        assert strategie.zustand == PaarZustand.PAUSIERT

        # Kurs kehrt zurueck in die Naehe des urspruenglichen Ankers (100).
        strategie.verarbeite_tick(100.2, zeit + pd.Timedelta(minutes=2), _regime(), pd.Series([90.0, 100.2]), None, gesamtkapital_bot=1000.0)
        assert strategie.zustand == PaarZustand.HANDELT


class TestGestoppt:
    def test_gestoppt_blockiert_jede_aktion(self):
        broker = FakeBroker()
        strategie = _strategie(broker)
        strategie.zustand = PaarZustand.GESTOPPT
        strategie.anker_preis = 100.0

        strategie.verarbeite_tick(50.0, pd.Timestamp("2024-01-01"), _regime(), pd.Series([100.0, 50.0]), None, gesamtkapital_bot=1000.0)

        assert strategie.zustand == PaarZustand.GESTOPPT
        assert broker.kauf_aufrufe == []
        assert broker.verkauf_aufrufe == []
