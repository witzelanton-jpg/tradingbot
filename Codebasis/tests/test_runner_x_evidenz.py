"""Test: der Runner uebernimmt ein statistisch unbelegtes X nicht."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from hurstbot.optimizer import FensterErgebnis, OptimierungsErgebnis
from hurstbot.backtest import BacktestResult


def _ergebnis(x_prozent: float, belastbar: bool, roundtrips: int) -> OptimierungsErgebnis:
    backtest_ergebnis = BacktestResult(
        trades=[], end_kapital_eur=500.0, end_bestand_wert_eur=0.0, netto_gewinn_eur=0.0,
        max_drawdown_prozent=0.0, anzahl_roundtrips=roundtrips, mittlere_amplitude_prozent=0.0,
        mittlere_haltedauer_stunden=0.0, offene_lots_anzahl=0, offene_lots_mittlere_dauer_stunden=0.0,
        equity_kurve=pd.Series([500.0, 500.0]),
    )
    fenster = FensterErgebnis(
        fenster_index=0, train_zeilen=100, test_zeilen=40, bestes_x_train=x_prozent,
        test_ergebnis=backtest_ergebnis, test_score=0.0,
    )
    return OptimierungsErgebnis(
        x_prozent=x_prozent, fenster_ergebnisse=[fenster], mittlerer_out_of_sample_score=0.0,
        mittlere_roundtrips=float(roundtrips), mittlere_amplitude_prozent=0.0, x_min_erlaubt=0.67,
        hat_belastbare_evidenz=belastbar,
    )


def test_unbelegtes_x_wird_nicht_uebernommen(runner_mit_start):
    runner = runner_mit_start
    strategie = runner.strategien["XXBTZEUR"]
    strategie.x_prozent = 2.5

    with patch("hurstbot.runner.walk_forward_optimize", return_value=_ergebnis(9.99, belastbar=False, roundtrips=0)):
        runner.optimiere_x_falls_faellig("XXBTZEUR", pd.Timestamp("2024-06-01", tz="UTC"))

    assert strategie.x_prozent == 2.5  # altes X bleibt erhalten


def test_belegtes_x_wird_uebernommen(runner_mit_start):
    runner = runner_mit_start
    strategie = runner.strategien["XXBTZEUR"]
    strategie.x_prozent = 2.5

    with patch("hurstbot.runner.walk_forward_optimize", return_value=_ergebnis(1.23, belastbar=True, roundtrips=7)):
        runner.optimiere_x_falls_faellig("XXBTZEUR", pd.Timestamp("2024-06-01", tz="UTC"))

    assert strategie.x_prozent == 1.23


def test_zeitstempel_wird_auch_ohne_evidenz_gesetzt(runner_mit_start):
    """Sonst wuerde der Bot bei jedem Zyklus erneut optimieren und die API fluten."""
    runner = runner_mit_start
    jetzt = pd.Timestamp("2024-06-01", tz="UTC")

    with patch("hurstbot.runner.walk_forward_optimize", return_value=_ergebnis(9.99, belastbar=False, roundtrips=0)):
        runner.optimiere_x_falls_faellig("XXBTZEUR", jetzt)

    assert runner._letzte_x_optimierung["XXBTZEUR"] == jetzt
