"""Tests fuer die CLI: Argumentverarbeitung, Broker-Wahl und Live-Sicherheit.

Die Kraken-API ist durchgehend gemockt. Insbesondere wird geprueft, dass der
Live-Modus ohne alle drei Freigaben NICHT zustande kommt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from hurstbot.broker import KrakenValidateBroker, PaperBroker
from hurstbot.cli import baue_parser, main


@pytest.fixture
def config_datei(tmp_path):
    """Schreibt eine minimale, gueltige Konfigurationsdatei ins Temp-Verzeichnis."""
    inhalt = {
        "start_kapital_eur": 100.0,
        "abfrage_sekunden": 1,
        "scanner": {"watchlist": ["XXBTZEUR"], "max_parallele_paare": 1},
        "hurst": {"bootstrap_wiederholungen": 20},
        "optimizer": {"x_grid_schritte": 4, "verfeinerung_schritte": 2, "walk_forward_fenster_anzahl": 2},
        "pfade": {
            "daten_verzeichnis": str(tmp_path / "data"),
            "log_verzeichnis": str(tmp_path / "logs"),
            "zustand_verzeichnis": str(tmp_path / "state"),
        },
    }
    pfad = tmp_path / "config.yaml"
    pfad.write_text(yaml.safe_dump(inhalt), encoding="utf-8")
    return pfad


@pytest.fixture
def gemockte_api(ohlc_antwort, ticker_antwort, asset_pairs_antwort):
    """Patcht KrakenAPI in cli.py, sodass keine echten Anfragen rausgehen."""
    mock = MagicMock()
    mock.asset_pairs.return_value = asset_pairs_antwort
    mock.ohlc.side_effect = lambda paar, intervall, since=None: ohlc_antwort(
        paar=paar, anzahl=300, intervall_minuten=intervall
    )
    mock.ticker.side_effect = lambda paare: ticker_antwort(paar=paare[0], preis=100.0)
    mock.open_orders.return_value = {"open": {}}
    with patch("hurstbot.cli.KrakenAPI", return_value=mock):
        yield mock


@pytest.fixture(autouse=True)
def _ohne_live_env(monkeypatch):
    monkeypatch.delenv("KRAKEN_LIVE_CONFIRM", raising=False)
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)


class TestParser:
    def test_befehl_ist_pflicht(self):
        with pytest.raises(SystemExit):
            baue_parser().parse_args([])

    def test_run_ist_standardmaessig_paper(self):
        args = baue_parser().parse_args(["run"])
        assert args.validate is False
        assert args.live is False

    def test_validate_und_live_schliessen_sich_aus(self):
        with pytest.raises(SystemExit):
            baue_parser().parse_args(["run", "--validate", "--live"])

    def test_backtest_benoetigt_pair(self):
        with pytest.raises(SystemExit):
            baue_parser().parse_args(["backtest"])


class TestFehlendeConfig:
    def test_fehlende_config_bricht_mit_hinweis_ab(self, tmp_path):
        with pytest.raises(SystemExit, match="config.example.yaml"):
            main(["--config", str(tmp_path / "gibtsnicht.yaml"), "scan"])


class TestBefehlScan:
    def test_scan_laeuft_und_handelt_nicht(self, config_datei, gemockte_api, capsys):
        code = main(["--config", str(config_datei), "scan"])

        assert code == 0
        ausgabe = capsys.readouterr().out
        assert "XXBTZEUR" in ausgabe
        assert "handelt nicht" in ausgabe
        gemockte_api.add_order.assert_not_called()


class TestBefehlBacktest:
    def test_backtest_schreibt_bericht(self, config_datei, gemockte_api, tmp_path, capsys):
        code = main(["--config", str(config_datei), "backtest", "--pair", "XXBTZEUR", "--x", "1.5", "--y", "20"])

        assert code == 0
        ausgabe = capsys.readouterr().out
        assert "Netto-Gewinn" in ausgabe
        assert (tmp_path / "logs" / "backtest_XXBTZEUR_5m.csv").exists()
        gemockte_api.add_order.assert_not_called()


class TestBefehlOptimize:
    def test_optimize_schreibt_csv_und_text(self, config_datei, gemockte_api, tmp_path, capsys):
        code = main(["--config", str(config_datei), "optimize", "--pair", "XXBTZEUR", "--y", "20"])

        assert code == 0
        ausgabe = capsys.readouterr().out
        assert "X-Optimierung" in ausgabe
        assert (tmp_path / "logs" / "optimierung_XXBTZEUR_5m.csv").exists()
        assert (tmp_path / "logs" / "optimierung_XXBTZEUR_5m.txt").exists()


class TestBefehlRunPaper:
    def test_run_nutzt_paper_broker_und_sendet_keine_orders(self, config_datei, gemockte_api):
        with patch("hurstbot.cli.BotRunner") as runner_klasse:
            code = main(["--config", str(config_datei), "run", "--max-zyklen", "1"])

        assert code == 0
        _, kwargs = runner_klasse.call_args
        assert isinstance(kwargs["broker"], PaperBroker)
        gemockte_api.add_order.assert_not_called()

    def test_run_paper_laeuft_echten_zyklus_ohne_orders(self, config_datei, gemockte_api):
        code = main(["--config", str(config_datei), "run", "--max-zyklen", "1"])

        assert code == 0
        gemockte_api.add_order.assert_not_called()


class TestBefehlRunValidate:
    def test_validate_benoetigt_api_schluessel(self, config_datei, gemockte_api):
        with pytest.raises(SystemExit, match="KRAKEN_API_KEY"):
            main(["--config", str(config_datei), "run", "--validate", "--max-zyklen", "1"])

    def test_validate_nutzt_validate_broker(self, config_datei, gemockte_api, monkeypatch):
        monkeypatch.setenv("KRAKEN_API_KEY", "test-key")
        monkeypatch.setenv("KRAKEN_API_SECRET", "dGVzdC1zZWNyZXQ=")

        with patch("hurstbot.cli.BotRunner") as runner_klasse:
            code = main(["--config", str(config_datei), "run", "--validate", "--max-zyklen", "1"])

        assert code == 0
        _, kwargs = runner_klasse.call_args
        assert isinstance(kwargs["broker"], KrakenValidateBroker)


class TestBefehlRunLiveSicherheit:
    """Der Live-Modus darf ohne alle drei Freigaben nie zustande kommen."""

    def test_live_ohne_api_schluessel_bricht_ab(self, config_datei, gemockte_api):
        with pytest.raises(SystemExit):
            main(["--config", str(config_datei), "run", "--live"])

    def test_live_ohne_env_var_bricht_ab(self, config_datei, gemockte_api, monkeypatch):
        monkeypatch.setenv("KRAKEN_API_KEY", "test-key")
        monkeypatch.setenv("KRAKEN_API_SECRET", "dGVzdC1zZWNyZXQ=")
        # Nutzer tippt sogar "JA", aber KRAKEN_LIVE_CONFIRM fehlt.
        monkeypatch.setattr("builtins.input", lambda _="": "JA")

        with pytest.raises(SystemExit, match="KRAKEN_LIVE_CONFIRM"):
            main(["--config", str(config_datei), "run", "--live"])

    def test_live_ohne_nutzerbestaetigung_bricht_ab(self, config_datei, gemockte_api, monkeypatch):
        monkeypatch.setenv("KRAKEN_API_KEY", "test-key")
        monkeypatch.setenv("KRAKEN_API_SECRET", "dGVzdC1zZWNyZXQ=")
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        monkeypatch.setattr("builtins.input", lambda _="": "nein")

        with pytest.raises(SystemExit, match="Nutzerbestaetigung"):
            main(["--config", str(config_datei), "run", "--live"])

        gemockte_api.add_order.assert_not_called()

    def test_live_bestaetigung_zeigt_paare_und_grenzen(self, config_datei, gemockte_api, monkeypatch, capsys):
        monkeypatch.setenv("KRAKEN_API_KEY", "test-key")
        monkeypatch.setenv("KRAKEN_API_SECRET", "dGVzdC1zZWNyZXQ=")
        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        monkeypatch.setattr("builtins.input", lambda _="": "nein")

        with pytest.raises(SystemExit):
            main(["--config", str(config_datei), "run", "--live"])

        ausgabe = capsys.readouterr().out
        assert "XXBTZEUR" in ausgabe          # Paar wird angezeigt
        assert "50.00 EUR" in ausgabe         # Betragsgrenze Y
        assert "100.00 EUR" in ausgabe        # Kapitalgrenze
        assert "WARNUNG" in ausgabe
