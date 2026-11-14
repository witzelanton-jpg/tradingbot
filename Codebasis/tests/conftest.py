"""Gemeinsame Test-Fixtures: gemockte Kraken-Antworten, nie die echte API."""

from __future__ import annotations

import numpy as np
import pytest

ASSET_PAIRS_ANTWORT = {
    "XXBTZEUR": {"altname": "XBTEUR", "pair_decimals": 1, "lot_decimals": 8, "ordermin": "0.0001"},
    "XDGEUR": {"altname": "XDGEUR", "pair_decimals": 7, "lot_decimals": 8, "ordermin": "50"},
}


def erzeuge_ohlc_rohdaten(anzahl: int = 300, start_zeit: int = 1700000000, intervall_minuten: int = 5, startpreis: float = 100.0, seed: int = 0) -> list:
    """Erzeugt OHLC-Rohdaten im exakten Kraken-Format (Listen aus Strings)."""
    rng = np.random.default_rng(seed)
    zeilen = []
    preis = startpreis
    for i in range(anzahl):
        # Oszillierende Reihe, damit Gate/Backtest in Tests etwas zu tun haben.
        preis = startpreis + 4 * np.sin(i / 8) + rng.normal(0, 0.1)
        zeit = start_zeit + i * intervall_minuten * 60
        hoch = preis * 1.002
        tief = preis * 0.998
        zeilen.append([zeit, f"{preis:.4f}", f"{hoch:.4f}", f"{tief:.4f}", f"{preis:.4f}", f"{preis:.4f}", "1.5", 10])
    return zeilen


@pytest.fixture
def ohlc_antwort():
    """OHLC-Antwort im Kraken-Format: {paar: [kerzen], "last": zeitstempel}."""

    def _erzeuge(paar: str = "XXBTZEUR", anzahl: int = 300, intervall_minuten: int = 5, seed: int = 0):
        rohdaten = erzeuge_ohlc_rohdaten(anzahl=anzahl, intervall_minuten=intervall_minuten, seed=seed)
        return {paar: rohdaten, "last": rohdaten[-1][0]}

    return _erzeuge


@pytest.fixture
def ticker_antwort():
    """Ticker-Antwort im Kraken-Format (nur die hier benutzten Felder)."""

    def _erzeuge(paar: str = "XXBTZEUR", preis: float = 100.0):
        return {
            paar: {
                "a": [f"{preis * 1.0005:.4f}", "1", "1.000"],
                "b": [f"{preis * 0.9995:.4f}", "1", "1.000"],
                "c": [f"{preis:.4f}", "0.1"],
            }
        }

    return _erzeuge


@pytest.fixture
def asset_pairs_antwort():
    return dict(ASSET_PAIRS_ANTWORT)


# ---------- Fixtures fuer die Runner-Integrationstests ----------


@pytest.fixture
def runner_config(tmp_path):
    """Kleine, schnelle Konfiguration mit Pfaden im Temp-Verzeichnis."""
    from hurstbot.config import AppConfig, KillSwitchConfig, MultiSkalenConfig, ScannerConfig, XOptimizerConfig

    return AppConfig(
        start_kapital_eur=200.0,
        abfrage_sekunden=1,
        scanner=ScannerConfig(watchlist=["XXBTZEUR", "XDGEUR"], max_parallele_paare=1),
        optimizer=XOptimizerConfig(
            x_grid_schritte=4, verfeinerung_schritte=2, walk_forward_fenster_anzahl=2, neuoptimierung_stunden=6.0
        ),
        kill_switch=KillSwitchConfig(max_verlust_prozent=10.0),
        pfade={
            "daten_verzeichnis": tmp_path / "data",
            "log_verzeichnis": tmp_path / "logs",
            "zustand_verzeichnis": tmp_path / "state",
        },
        hurst={
            "bootstrap_wiederholungen": 30,
            "multiskalen": MultiSkalenConfig(kurz_anzahl_kerzen=200, mittel_anzahl_kerzen=200, lang_anzahl_kerzen=100),
        },
    )


@pytest.fixture
def gemockte_kraken_api(ohlc_antwort, ticker_antwort, asset_pairs_antwort):
    """Gemockte Kraken-API mit konsistenten OHLC-, Ticker- und AssetPairs-Daten."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.asset_pairs.return_value = asset_pairs_antwort
    mock.ohlc.side_effect = lambda paar, intervall, since=None: ohlc_antwort(
        paar=paar, anzahl=300, intervall_minuten=intervall
    )
    mock.ticker.side_effect = lambda paare: ticker_antwort(paar=paare[0], preis=100.0)
    mock.open_orders.return_value = {"open": {}}
    return mock


@pytest.fixture
def runner(runner_config, gemockte_kraken_api, tmp_path):
    from hurstbot.broker import PaperBroker
    from hurstbot.runner import BotRunner

    return BotRunner(
        config=runner_config,
        api=gemockte_kraken_api,
        broker=PaperBroker(runner_config.gebuehren),
        zustand_pfad=tmp_path / "state" / "bot_zustand.json",
        trade_log_pfad=tmp_path / "logs" / "trade_log.csv",
        zustand_log_pfad=tmp_path / "logs" / "zustands_log.jsonl",
    )


@pytest.fixture
def runner_mit_start(runner):
    """Ein bereits gestarteter Runner (Watchlist geprueft, Strategien angelegt)."""
    runner.starte()
    return runner
