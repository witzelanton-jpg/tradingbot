"""Tests fuer die X-Optimierung (Walk-Forward) und die Zeitskalenauswahl."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hurstbot.config import GebuehrenConfig, XOptimizerConfig, ZeitskalaProfil
from hurstbot.optimizer import (
    FensterErgebnis,
    OptimierungsErgebnis,
    gebuehren_breakeven_x_prozent,
    waehle_zeitskala_profil,
    walk_forward_optimize,
    zielfunktion,
)


def _df_aus_preisen(preise: np.ndarray, freq="5min") -> pd.DataFrame:
    zeiten = pd.date_range("2024-01-01", periods=len(preise), freq=freq)
    return pd.DataFrame({"zeit": zeiten, "close": preise})


def _oszillierende_df(n=400, periode=20, amplitude=6.0, rauschen=0.1, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    preise = 100 + amplitude * np.sin(2 * np.pi * x / periode) + rng.normal(0, rauschen, n)
    return _df_aus_preisen(preise)


def _schnelle_x_config(**overrides) -> XOptimizerConfig:
    basis = dict(
        x_grid_min_prozent=0.2,
        x_grid_max_prozent=4.0,
        x_grid_schritte=6,
        verfeinerung_schritte=3,
        walk_forward_fenster_anzahl=2,
        walk_forward_test_anteil=0.3,
        glaettung_alpha=0.5,
    )
    basis.update(overrides)
    return XOptimizerConfig(**basis)


GEBUEHREN = GebuehrenConfig(
    maker_gebuehr_prozent=0.16, taker_gebuehr_prozent=0.2, geschaetzter_spread_prozent=0.05, sicherheitspuffer_prozent=0.1
)


class TestGebuehrenBreakeven:
    def test_formel(self):
        x_min = gebuehren_breakeven_x_prozent(GEBUEHREN)
        assert x_min == pytest.approx(2 * 0.2 + 0.05 + 0.1)

    def test_grid_untergrenze_wird_nie_unterschritten(self):
        df = _oszillierende_df()
        config = _schnelle_x_config(x_grid_min_prozent=0.001)  # absichtlich unterhalb des Breakeven
        ergebnis = walk_forward_optimize(df, config, GEBUEHREN, y_eur=20.0, start_kapital=500.0)
        breakeven = gebuehren_breakeven_x_prozent(GEBUEHREN)
        assert ergebnis.x_min_erlaubt >= breakeven
        assert ergebnis.x_prozent >= breakeven


class TestWalkForwardOptimize:
    def test_findet_profitables_x_bei_klarer_oszillation(self):
        df = _oszillierende_df(n=500, periode=25, amplitude=8.0, rauschen=0.05)
        config = _schnelle_x_config()
        ergebnis = walk_forward_optimize(df, config, GEBUEHREN, y_eur=20.0, start_kapital=500.0)

        assert ergebnis.x_min_erlaubt <= ergebnis.x_prozent <= config.x_grid_max_prozent
        assert ergebnis.mittlere_roundtrips > 0
        assert len(ergebnis.fenster_ergebnisse) > 0

    def test_zu_wenig_daten_wirft_fehler(self):
        df = _oszillierende_df(n=10)
        config = _schnelle_x_config()
        with pytest.raises(ValueError):
            walk_forward_optimize(df, config, GEBUEHREN, y_eur=20.0, start_kapital=500.0)

    def test_glaettung_begrenzt_sprung_gegenueber_vorherigem_x(self):
        df = _oszillierende_df(n=500, periode=25, amplitude=8.0, rauschen=0.05)
        config = _schnelle_x_config(glaettung_alpha=0.2)
        vorheriger_x = 3.5
        ergebnis = walk_forward_optimize(df, config, GEBUEHREN, y_eur=20.0, start_kapital=500.0, vorheriger_x=vorheriger_x)

        # Rohe Median-Wahl koennte weit vom vorherigen X abweichen; nach Glaettung
        # darf die Veraenderung nur ein kleiner Bruchteil davon sein (hier: alpha=0.2).
        ergebnis_ohne_glaettung = walk_forward_optimize(
            df, config.model_copy(update={"glaettung_alpha": 1.0}), GEBUEHREN, y_eur=20.0, start_kapital=500.0,
            vorheriger_x=vorheriger_x,
        )
        max_erwartete_aenderung = abs(ergebnis_ohne_glaettung.x_prozent - vorheriger_x) * config.glaettung_alpha + 1e-9
        assert abs(ergebnis.x_prozent - vorheriger_x) <= max_erwartete_aenderung + 1e-6


class TestBelastbareEvidenz:
    def test_oszillation_mit_roundtrips_gilt_als_belastbar(self):
        df = _oszillierende_df(n=600, periode=25, amplitude=8.0, rauschen=0.05)
        ergebnis = walk_forward_optimize(df, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0)
        assert ergebnis.hat_belastbare_evidenz is True
        assert ergebnis.mittlere_roundtrips > 0

    def test_zu_kurze_testfenster_gelten_als_nicht_belastbar(self):
        """Ein flacher Kurs ohne Bewegung erzeugt keinen einzigen Round-Trip -
        das daraus gewaehlte X darf nicht als validiert gelten."""
        flach = _df_aus_preisen(np.full(400, 100.0) + np.linspace(0, 0.01, 400))
        ergebnis = walk_forward_optimize(flach, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0)
        assert ergebnis.hat_belastbare_evidenz is False

    def test_textbericht_warnt_bei_fehlender_evidenz(self):
        flach = _df_aus_preisen(np.full(400, 100.0) + np.linspace(0, 0.01, 400))
        ergebnis = walk_forward_optimize(flach, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0)
        from hurstbot.optimizer import textbericht

        bericht = textbericht(ergebnis, "TESTEUR")
        assert "WARNUNG" in bericht
        assert "Round-Trip" in bericht
        assert "handelt mit einem solchen X nicht" in bericht

    def test_negatives_out_of_sample_ergebnis_gilt_nicht_als_belastbar(self):
        """Ein X, das out-of-sample im Mittel Verlust macht, ist nur das am wenigsten
        schlechte - es darf nicht als validiert gelten (gleiches Kriterium wie bei
        der Zeitskalen-Auswahl)."""
        rng = np.random.default_rng(7)
        # Fallender Trend: die Strategie kauft hinein und verliert.
        fallend = _df_aus_preisen(100 - np.arange(400) * 0.05 + rng.normal(0, 0.3, 400))
        ergebnis = walk_forward_optimize(fallend, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0)

        if ergebnis.mittlerer_out_of_sample_score <= 0:
            assert ergebnis.hat_belastbare_evidenz is False

    def test_belastbar_erfordert_roundtrips_und_gewinn(self):
        df = _oszillierende_df(n=600, periode=25, amplitude=8.0, rauschen=0.05)
        ergebnis = walk_forward_optimize(df, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0)

        assert ergebnis.hat_belastbare_evidenz == (
            ergebnis.mittlere_roundtrips > 0 and ergebnis.mittlerer_out_of_sample_score > 0
        )


class TestZielfunktion:
    def test_drawdown_senkt_score(self):
        from hurstbot.backtest import simulate

        config = _schnelle_x_config()
        df = _oszillierende_df(n=300)
        ergebnis = simulate(df, 1.0, 20.0, 0.2, 0.05, 500.0)
        score_normal = zielfunktion(ergebnis, config)
        score_mit_strafe = zielfunktion(ergebnis, config.model_copy(update={"max_drawdown_strafe": 100.0}))
        assert score_mit_strafe <= score_normal


class TestZeitskalaAuswahl:
    def test_waehlt_profil_mit_bestem_profit_nicht_nach_h(self):
        """Die Auswahl muss ausschliesslich auf Basis des Out-of-Sample-Nettogewinns
        erfolgen. Wir stubben walk_forward_optimize, damit kein echter Hurst-Wert
        ueberhaupt in die Entscheidung einfliessen kann."""
        profile = [
            ZeitskalaProfil(name="5m", intervall_minuten=5, hurst_fenster=50),
            ZeitskalaProfil(name="15m", intervall_minuten=15, hurst_fenster=50),
        ]
        daten = {p.name: _oszillierende_df(n=200) for p in profile}

        def fake_walk_forward(df, *_args, **_kwargs):
            # "5m" bekommt den schlechteren Score, "15m" den besseren - unabhaengig von H.
            ist_15m = df is daten["15m"]
            score = 50.0 if ist_15m else -5.0
            fenster = FensterErgebnis(
                fenster_index=0, train_zeilen=100, test_zeilen=50, bestes_x_train=1.0,
                test_ergebnis=_dummy_backtest_ergebnis(score), test_score=score,
            )
            return OptimierungsErgebnis(
                x_prozent=1.0, fenster_ergebnisse=[fenster], mittlerer_out_of_sample_score=score,
                mittlere_roundtrips=5.0, mittlere_amplitude_prozent=2.0, x_min_erlaubt=0.5,
            )

        with patch("hurstbot.optimizer.walk_forward_optimize", side_effect=fake_walk_forward):
            bestes, grund, ergebnisse = waehle_zeitskala_profil(
                daten, profile, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0,
                mindest_roundtrips=1, fallback_profil=profile[0],
            )

        assert bestes.name == "15m"

    def test_fallback_wenn_kein_kandidat_mindestkriterium_erfuellt(self):
        profile = [ZeitskalaProfil(name="5m", intervall_minuten=5, hurst_fenster=50)]
        daten = {"5m": _oszillierende_df(n=200)}

        def fake_walk_forward(*_args, **_kwargs):
            fenster = FensterErgebnis(
                fenster_index=0, train_zeilen=100, test_zeilen=50, bestes_x_train=1.0,
                test_ergebnis=_dummy_backtest_ergebnis(-10.0), test_score=-10.0,
            )
            return OptimierungsErgebnis(
                x_prozent=1.0, fenster_ergebnisse=[fenster], mittlerer_out_of_sample_score=-10.0,
                mittlere_roundtrips=0.0, mittlere_amplitude_prozent=0.0, x_min_erlaubt=0.5,
            )

        with patch("hurstbot.optimizer.walk_forward_optimize", side_effect=fake_walk_forward):
            bestes, grund, _ = waehle_zeitskala_profil(
                daten, profile, _schnelle_x_config(), GEBUEHREN, y_eur=20.0, start_kapital=500.0,
                mindest_roundtrips=3, fallback_profil=profile[0],
            )

        assert bestes.name == "5m"
        assert "Fallback" in grund

    def test_zufallsrauschen_fuehrt_nicht_systematisch_zu_erfolg(self):
        """Selektor-Bias-Test (Abschnitt 2.5): Auf reinem Rauschen (kein echtes
        Mean-Reversion-Signal) soll der Selektor in der Mehrheit der Faelle KEIN
        Profil als 'gut' einstufen, sondern auf den Fallback zurueckfallen."""
        profil = ZeitskalaProfil(name="5m", intervall_minuten=5, hurst_fenster=50)
        config = _schnelle_x_config(x_grid_min_prozent=0.5, x_grid_max_prozent=3.0)

        fallback_zaehler = 0
        anzahl_seeds = 6
        for seed in range(anzahl_seeds):
            rng = np.random.default_rng(seed + 100)
            reines_rauschen = 100 + np.cumsum(rng.normal(0, 0.2, 300))
            df = _df_aus_preisen(reines_rauschen)
            _, grund, _ = waehle_zeitskala_profil(
                {"5m": df}, [profil], config, GEBUEHREN, y_eur=20.0, start_kapital=500.0,
                mindest_roundtrips=5, fallback_profil=profil,
            )
            if "Fallback" in grund:
                fallback_zaehler += 1

        assert fallback_zaehler >= anzahl_seeds // 2


def _dummy_backtest_ergebnis(netto_gewinn: float):
    from hurstbot.backtest import BacktestResult

    return BacktestResult(
        trades=[],
        end_kapital_eur=500.0 + netto_gewinn,
        end_bestand_wert_eur=0.0,
        netto_gewinn_eur=netto_gewinn,
        max_drawdown_prozent=1.0,
        anzahl_roundtrips=5,
        mittlere_amplitude_prozent=2.0,
        mittlere_haltedauer_stunden=3.0,
        offene_lots_anzahl=0,
        offene_lots_mittlere_dauer_stunden=0.0,
        equity_kurve=pd.Series([500.0, 500.0 + netto_gewinn]),
    )
