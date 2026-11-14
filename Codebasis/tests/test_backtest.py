"""Tests fuer die Lot-Buchfuehrung und den Backtest (inkl. Look-Ahead-Bias-Test)."""

from __future__ import annotations

import pandas as pd
import pytest

from hurstbot.backtest import LotBuch, simulate


def _df(preise: list[float], start="2024-01-01", freq="h") -> pd.DataFrame:
    zeiten = pd.date_range(start, periods=len(preise), freq=freq)
    return pd.DataFrame({"zeit": zeiten, "close": preise})


class TestLotBuch:
    def test_verkauf_nur_bei_netto_gewinn(self):
        """Ein Lot darf nicht verkauft werden, wenn der Nettoerloes (nach Gebuehr/Spread)
        nicht ueber dem Einstandspreis liegt - selbst wenn der Rohpreis gestiegen ist."""
        buch = LotBuch(start_kapital_eur=1000, gebuehr_prozent=0.5, spread_prozent=0.2)
        zeit0 = pd.Timestamp("2024-01-01")
        buch.kaufen(100.0, zeit0, 100.0)
        lot = buch.lots[0]

        # Rohpreis minimal ueber dem Einstandspreis, aber Gebuehr+Spread fressen den Gewinn auf.
        knapp_ueber_einstand = lot.einstandspreis * 1.001
        verkaeufe = buch.verkaufe_faellige_lots(knapp_ueber_einstand, zeit0, x_prozent=0.05)
        assert verkaeufe == []
        assert len(buch.lots) == 1

    def test_verkauf_bei_echtem_netto_gewinn(self):
        buch = LotBuch(start_kapital_eur=1000, gebuehr_prozent=0.1, spread_prozent=0.0)
        zeit0 = pd.Timestamp("2024-01-01")
        buch.kaufen(100.0, zeit0, 100.0)
        verkaeufe = buch.verkaufe_faellige_lots(110.0, zeit0, x_prozent=1.0)
        assert len(verkaeufe) == 1
        assert buch.lots == []
        assert buch.kapital_eur > 900.0  # Kapital wieder da plus Gewinn

    def test_kauf_scheitert_ohne_kapital(self):
        buch = LotBuch(start_kapital_eur=50, gebuehr_prozent=0.1, spread_prozent=0.0)
        erfolg = buch.kaufen(100.0, pd.Timestamp("2024-01-01"), 100.0)
        assert erfolg is False
        assert buch.lots == []


class TestSimulate:
    def test_vollstaendiger_roundtrip_mit_bekannten_zahlen(self):
        df = _df([100, 100, 95, 95, 103])
        ergebnis = simulate(
            df, x_prozent=2.0, y_eur=100.0, gebuehr_prozent=0.1, spread_prozent=0.0, start_kapital_eur=1000.0
        )

        assert ergebnis.anzahl_roundtrips == 1
        assert ergebnis.offene_lots_anzahl == 0
        assert ergebnis.netto_gewinn_eur == pytest.approx(8.2126315789, rel=1e-6)
        assert ergebnis.mittlere_amplitude_prozent == pytest.approx(8.312739891687277, rel=1e-6)
        assert ergebnis.mittlere_haltedauer_stunden == pytest.approx(2.0, rel=1e-6)

    def test_kein_verkauf_ohne_bestand(self):
        df = _df([100, 101, 102, 103])
        ergebnis = simulate(df, x_prozent=5.0, y_eur=20.0, gebuehr_prozent=0.1, spread_prozent=0.0, start_kapital_eur=100.0)
        assert ergebnis.anzahl_roundtrips == 0
        assert ergebnis.trades == []

    def test_kein_look_ahead_bias(self):
        """Die Trades bis zu einem Zeitpunkt t duerfen sich nicht aendern, wenn weitere
        (zukuenftige) Kerzen an die Reihe angehaengt werden."""
        preise_basis = [100, 98, 97, 101, 99, 96, 94, 102, 108, 90]
        df_kurz = _df(preise_basis)
        df_lang = _df(preise_basis + [50, 200, 10, 300, 5])  # zusaetzliche "Zukunft"

        ergebnis_kurz = simulate(df_kurz, x_prozent=1.5, y_eur=20.0, gebuehr_prozent=0.2, spread_prozent=0.05, start_kapital_eur=200.0)
        ergebnis_lang = simulate(df_lang, x_prozent=1.5, y_eur=20.0, gebuehr_prozent=0.2, spread_prozent=0.05, start_kapital_eur=200.0)

        trades_lang_im_basis_zeitraum = [t for t in ergebnis_lang.trades if t.zeit <= df_kurz["zeit"].iloc[-1]]
        assert len(trades_lang_im_basis_zeitraum) == len(ergebnis_kurz.trades)
        for t_kurz, t_lang in zip(ergebnis_kurz.trades, trades_lang_im_basis_zeitraum):
            assert t_kurz.art == t_lang.art
            assert t_kurz.preis == pytest.approx(t_lang.preis)
            assert t_kurz.menge == pytest.approx(t_lang.menge)

    def test_max_drawdown_nicht_negativ(self):
        df = _df([100, 90, 80, 70, 120])
        ergebnis = simulate(df, x_prozent=1.0, y_eur=10.0, gebuehr_prozent=0.1, spread_prozent=0.0, start_kapital_eur=100.0)
        assert ergebnis.max_drawdown_prozent >= 0.0
