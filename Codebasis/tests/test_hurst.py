"""Tests fuer die Hurst-Schaetzung: trendende, zufaellige und oszillierende Reihen."""

from __future__ import annotations

import numpy as np
import pytest

from hurstbot.config import HurstConfig
from hurstbot.hurst import (
    block_bootstrap_konfidenz,
    hurst_dfa,
    hurst_lag_std,
    hurst_rs,
    kombinierter_hurst,
    monotonie_score,
    zu_log_preisen,
)


def _schnelle_hurst_config(**overrides) -> HurstConfig:
    basis = dict(bootstrap_wiederholungen=60, bootstrap_blockgroesse=15, min_lag=2, max_lag=30)
    basis.update(overrides)
    return HurstConfig(**basis)


def _ar1_preisreihe(rho: float, n: int = 600, sigma: float = 0.01, start: float = 100.0, seed: int = 0) -> np.ndarray:
    """Erzeugt eine synthetische Preisreihe ueber ein AR(1)-Modell auf den Log-Renditen.

    rho > 0: positiv autokorrelierte Renditen -> persistenter Trend (H > 0.5)
    rho = 0: unkorrelierte Renditen -> reiner Zufallspfad (H ~ 0.5)
    rho < 0: negativ autokorrelierte Renditen -> Mean-Reversion/Oszillation (H < 0.5)
    """
    rng = np.random.default_rng(seed)
    renditen = np.zeros(n)
    for t in range(1, n):
        renditen[t] = rho * renditen[t - 1] + rng.normal(0, sigma)
    log_preise = np.log(start) + np.cumsum(renditen)
    return np.exp(log_preise)


@pytest.fixture
def trend_preise() -> np.ndarray:
    return _ar1_preisreihe(rho=0.6, seed=10)


@pytest.fixture
def zufalls_preise() -> np.ndarray:
    return _ar1_preisreihe(rho=0.0, seed=11)


@pytest.fixture
def oszillierende_preise() -> np.ndarray:
    return _ar1_preisreihe(rho=-0.6, seed=12)


class TestEinzelneSchaetzer:
    def test_lag_std_trend_ueber_0_5(self, trend_preise):
        h = hurst_lag_std(zu_log_preisen(trend_preise), min_lag=2, max_lag=30)
        assert h > 0.5

    def test_lag_std_oszillierend_unter_0_5(self, oszillierende_preise):
        h = hurst_lag_std(zu_log_preisen(oszillierende_preise), min_lag=2, max_lag=30)
        assert h < 0.5

    def test_rs_oszillierend_unter_0_5(self, oszillierende_preise):
        h = hurst_rs(zu_log_preisen(oszillierende_preise), min_lag=10, max_lag=100)
        assert h < 0.5

    def test_dfa_oszillierend_unter_0_5(self, oszillierende_preise):
        h = hurst_dfa(zu_log_preisen(oszillierende_preise), min_lag=10, max_lag=100)
        assert h < 0.5

    def test_zu_log_preisen_wirft_bei_negativen_preisen(self):
        with pytest.raises(ValueError):
            zu_log_preisen(np.array([1.0, -2.0, 3.0]))


class TestKombinierterHurst:
    def test_trend_liefert_h_ueber_0_5(self, trend_preise):
        config = _schnelle_hurst_config()
        h = kombinierter_hurst(trend_preise, config)
        assert h > 0.5

    def test_zufall_liefert_h_nahe_0_5(self, zufalls_preise):
        config = _schnelle_hurst_config()
        h = kombinierter_hurst(zufalls_preise, config)
        assert 0.3 < h < 0.7

    def test_oszillierend_liefert_h_deutlich_unter_0_5(self, oszillierende_preise):
        config = _schnelle_hurst_config()
        h = kombinierter_hurst(oszillierende_preise, config)
        assert h < 0.45


class TestBlockBootstrapKonfidenz:
    def test_ci_enthaelt_h_und_konfidenz_in_bereich(self, oszillierende_preise):
        config = _schnelle_hurst_config()
        ergebnis = block_bootstrap_konfidenz(oszillierende_preise, config, zufallsgenerator=np.random.default_rng(42))
        assert ergebnis.ci_untere_grenze <= ergebnis.ci_obere_grenze
        assert 0.0 <= ergebnis.konfidenz <= 1.0

    def test_kurze_reihe_liefert_niedrige_konfidenz(self):
        config = _schnelle_hurst_config(bootstrap_wiederholungen=10)
        kurze_reihe = 100 + np.cumsum(np.random.default_rng(2).normal(0, 0.1, 8))
        ergebnis = block_bootstrap_konfidenz(kurze_reihe, config)
        assert ergebnis.konfidenz == 0.0


class TestMonotonieScore:
    def test_trend_hat_hohen_score(self):
        # Fuer die Monotonie-Pruefung braucht es einen Kurs, der FAST OHNE Umkehrungen
        # steigt - das ist ein anderer Fall als "H > 0.5" (das AR(1)-Modell oben kann
        # trotz Persistenz viele einzelne Richtungswechsel haben).
        rng = np.random.default_rng(20)
        preise = 100 + np.arange(600) * 0.1 + rng.normal(0, 0.05, 600)
        ergebnis = monotonie_score(preise)
        assert ergebnis["monotonie_score"] > 0.5

    def test_oszillierend_hat_niedrigen_score(self, oszillierende_preise):
        ergebnis = monotonie_score(oszillierende_preise)
        assert ergebnis["monotonie_score"] < 0.5

    def test_score_liegt_in_bereich(self, zufalls_preise):
        ergebnis = monotonie_score(zufalls_preise)
        assert 0.0 <= ergebnis["monotonie_score"] <= 1.0

    def test_konstante_reihe_liefert_definierte_werte(self):
        ergebnis = monotonie_score(np.full(10, 100.0))
        assert ergebnis["monotonie_score"] == 0.0
