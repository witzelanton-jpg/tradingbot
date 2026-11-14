"""Integrationstests fuer die Hauptschleife (gegen gemockte Kraken-Antworten)."""

from __future__ import annotations

import pandas as pd
import pytest

from hurstbot.backtest import Lot
from hurstbot.guards import PaarZustand
from hurstbot.runner import BotRunner, profil_nach_name
from hurstbot.state import lade_zustand


class TestStart:
    def test_start_prueft_watchlist_und_legt_strategien_an(self, runner):
        runner.starte()

        assert set(runner.strategien) == {"XXBTZEUR", "XDGEUR"}
        assert runner.bot_zustand.kapital_frei_eur == pytest.approx(200.0)

    def test_x_startwert_liegt_ueber_gebuehren_breakeven(self, runner):
        from hurstbot.optimizer import gebuehren_breakeven_x_prozent

        runner.starte()
        breakeven = gebuehren_breakeven_x_prozent(runner.config.gebuehren)

        for strategie in runner.strategien.values():
            assert strategie.x_prozent > breakeven

    def test_zustand_wird_wiederhergestellt(self, runner, tmp_path):
        runner.starte()
        strategie = runner.strategien["XXBTZEUR"]
        strategie.lots.append(Lot(einstandspreis=99.0, fill_preis_kauf=98.5, menge=0.5, kauf_zeit=pd.Timestamp("2024-01-01", tz="UTC")))
        strategie.kapital_eur = 42.0
        strategie.x_prozent = 2.345
        runner.speichere()

        neuer_runner = BotRunner(
            config=runner.config, api=runner.api, broker=runner.broker,
            zustand_pfad=runner.zustand_pfad, trade_log_pfad=runner.trade_log_pfad, zustand_log_pfad=runner.zustand_log_pfad,
        )
        neuer_runner.starte()

        wiederhergestellt = neuer_runner.strategien["XXBTZEUR"]
        assert len(wiederhergestellt.lots) == 1
        assert wiederhergestellt.kapital_eur == pytest.approx(42.0)
        assert wiederhergestellt.x_prozent == pytest.approx(2.345)


class TestZyklus:
    def test_zyklus_laeuft_ohne_fehler_und_speichert(self, runner):
        runner.starte()
        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))

        assert runner.zustand_pfad.exists()
        gespeichert = lade_zustand(runner.zustand_pfad)
        assert gespeichert is not None
        assert set(gespeichert.paare) == {"XXBTZEUR", "XDGEUR"}

    def test_nur_max_parallele_paare_erhalten_kapital(self, runner):
        runner.starte()
        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))

        paare_mit_kapital = [p for p, s in runner.strategien.items() if s.kapital_eur > 0]
        assert len(paare_mit_kapital) <= runner.config.scanner.max_parallele_paare

    def test_netzwerkfehler_im_zyklus_fuehrt_nicht_zum_absturz(self, runner):
        from hurstbot.kraken_api import KrakenVerbindungsFehler

        runner.starte()
        runner.api.ohlc.side_effect = KrakenVerbindungsFehler("Netz weg")

        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))  # darf nicht werfen

        assert runner.bot_zustand.global_zustand == "HANDELT"

    def test_ticker_fehler_wird_abgefangen(self, runner):
        from hurstbot.kraken_api import KrakenAPIError

        runner.starte()
        runner.api.ticker.side_effect = KrakenAPIError("Ticker kaputt")

        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))  # darf nicht werfen


class TestKillSwitch:
    def test_kill_switch_stoppt_alle_paare(self, runner):
        runner.starte()
        # Grosser Verlust: freies Kapital fast weg, keine Lots als Gegenwert.
        runner.bot_zustand.kapital_frei_eur = 50.0
        for strategie in runner.strategien.values():
            strategie.kapital_eur = 0.0

        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))

        assert runner.bot_zustand.global_zustand == PaarZustand.GESTOPPT.value
        assert all(s.zustand == PaarZustand.GESTOPPT for s in runner.strategien.values())

    def test_gestoppter_bot_fuehrt_keine_zyklen_mehr_aus(self, runner):
        runner.starte()
        runner.bot_zustand.global_zustand = PaarZustand.GESTOPPT.value
        aufrufe_vorher = runner.api.ticker.call_count

        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))

        assert runner.api.ticker.call_count == aufrufe_vorher

    def test_zustandslog_enthaelt_kill_switch_eintrag(self, runner):
        runner.starte()
        runner.bot_zustand.kapital_frei_eur = 10.0
        for strategie in runner.strategien.values():
            strategie.kapital_eur = 0.0

        runner.fuehre_zyklus_aus(pd.Timestamp("2024-06-01", tz="UTC"))

        inhalt = runner.zustand_log_pfad.read_text(encoding="utf-8")
        assert "GESTOPPT" in inhalt
        assert "Kill-Switch" in inhalt


class TestZeitskalaWechsel:
    def test_wechsel_laesst_offene_lots_erhalten(self, runner):
        runner.starte()
        strategie = runner.strategien["XXBTZEUR"]
        strategie.lots.append(Lot(einstandspreis=99.0, fill_preis_kauf=98.5, menge=0.5, kauf_zeit=pd.Timestamp("2024-01-01", tz="UTC")))
        lots_vorher = list(strategie.lots)
        altes_profil = strategie.zeitskala_profil.name

        runner.bewerte_zeitskala_falls_faellig("XXBTZEUR", pd.Timestamp("2024-06-01", tz="UTC"))

        assert strategie.lots == lots_vorher
        assert strategie.zeitskala_profil is not None

    def test_neubewertung_wird_vor_ablauf_des_intervalls_uebersprungen(self, runner):
        runner.starte()
        jetzt = pd.Timestamp("2024-06-01", tz="UTC")
        runner._letzte_zeitskala_bewertung["XXBTZEUR"] = jetzt
        aufrufe_vorher = runner.api.ohlc.call_count

        runner.bewerte_zeitskala_falls_faellig("XXBTZEUR", jetzt + pd.Timedelta(hours=1))

        assert runner.api.ohlc.call_count == aufrufe_vorher


class TestXOptimierung:
    def test_optimierung_setzt_x_und_merkt_zeitpunkt(self, runner):
        runner.starte()
        jetzt = pd.Timestamp("2024-06-01", tz="UTC")

        runner.optimiere_x_falls_faellig("XXBTZEUR", jetzt)

        assert runner._letzte_x_optimierung.get("XXBTZEUR") == jetzt
        assert runner.strategien["XXBTZEUR"].x_prozent > 0

    def test_optimierung_wird_vor_ablauf_des_intervalls_uebersprungen(self, runner):
        runner.starte()
        jetzt = pd.Timestamp("2024-06-01", tz="UTC")
        runner._letzte_x_optimierung["XXBTZEUR"] = jetzt
        x_vorher = runner.strategien["XXBTZEUR"].x_prozent

        runner.optimiere_x_falls_faellig("XXBTZEUR", jetzt + pd.Timedelta(hours=1))

        assert runner.strategien["XXBTZEUR"].x_prozent == x_vorher


class TestLaufeUndBeenden:
    def test_laufe_beendet_nach_max_zyklen(self, runner):
        runner.starte()
        schlaf_aufrufe = []

        runner.laufe(max_zyklen=2, schlafen=schlaf_aufrufe.append)

        assert runner.zustand_pfad.exists()
        assert len(schlaf_aufrufe) == 1  # nach dem letzten Zyklus wird nicht mehr geschlafen

    def test_strg_c_speichert_zustand(self, runner, monkeypatch):
        runner.starte()

        def _werfe_interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(runner, "fuehre_zyklus_aus", _werfe_interrupt)
        runner.laufe(max_zyklen=5, schlafen=lambda _: None)

        assert runner.zustand_pfad.exists()

    def test_beende_sauber_storniert_offene_orders_im_live_modus(self, runner, monkeypatch):
        from unittest.mock import MagicMock

        from hurstbot.broker import KrakenLiveBroker

        monkeypatch.setenv("KRAKEN_LIVE_CONFIRM", "JA")
        runner.broker = KrakenLiveBroker(
            runner.api, runner.config.gebuehren, live_flag_gesetzt=True, nutzer_bestaetigung="JA"
        )
        runner.starte()
        runner.api.open_orders.return_value = {"open": {"ORDER-1": {}, "ORDER-2": {}}}

        runner.beende_sauber()

        assert runner.api.cancel_order.call_count == 2

    def test_beende_sauber_fragt_im_paper_modus_keine_orders_ab(self, runner):
        """Im Paper-Modus existiert kein privater API-Zugang - die Abfrage waere
        nur eine irrefuehrende Warnung."""
        runner.starte()

        runner.beende_sauber()

        runner.api.open_orders.assert_not_called()
        runner.api.cancel_order.assert_not_called()


class TestProfilNachName:
    def test_findet_bekanntes_profil(self, runner_config):
        assert profil_nach_name(runner_config, "15m").name == "15m"

    def test_unbekannter_name_liefert_fallback(self, runner_config):
        assert profil_nach_name(runner_config, "gibtsnicht").name == runner_config.zeitskala_auswahl.fallback_profil
