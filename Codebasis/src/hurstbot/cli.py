"""Kommandozeilen-Schnittstelle des Bots.

Befehle:
  scan                      Watchlist bewerten und Ranking ausgeben (kein Handel)
  backtest --pair PAAR      Einmalige Simulation mit festem X, Bericht als CSV + Text
  optimize --pair PAAR      Walk-Forward-X-Optimierung, Bericht als CSV + Text
  run                       Paper-Betrieb (Default, keine echten Orders)
  run --validate            Orders an Kraken senden, aber nur mit validate=true
  run --live                ECHTE Orders - nur mit allen drei Freigaben (Abschnitt 5)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from .backtest import simulate
from .broker import Broker, KrakenLiveBroker, KrakenValidateBroker, LiveModusNichtErlaubt, PaperBroker
from .config import AppConfig, lade_config
from .data import hole_multiskalen_ohlc, hole_ohlc
from .kraken_api import KrakenAPI, KrakenAPIError, KrakenVerbindungsFehler
from .logging_setup import richte_konsolen_logging_ein
from .optimizer import fenster_ergebnisse_als_dataframe, textbericht, walk_forward_optimize
from .regime import bewerte_regime
from .runner import BotRunner, profil_nach_name
from .scanner import formatiere_ranking, pruefe_watchlist, ranke_paare

logger = logging.getLogger("hurstbot.cli")


def _lade_umgebung() -> None:
    """Laedt .env, falls vorhanden (API-Schluessel kommen NUR aus der Umgebung)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.debug("python-dotenv nicht installiert - lese nur echte Umgebungsvariablen")


def _api_aus_config(config: AppConfig, mit_schluessel: bool) -> KrakenAPI:
    api_key = os.environ.get("KRAKEN_API_KEY") if mit_schluessel else None
    api_secret = os.environ.get("KRAKEN_API_SECRET") if mit_schluessel else None
    if mit_schluessel and (not api_key or not api_secret):
        raise SystemExit(
            "KRAKEN_API_KEY und KRAKEN_API_SECRET muessen als Umgebungsvariablen gesetzt sein "
            "(z. B. ueber eine .env-Datei). Siehe .env.example und README."
        )
    return KrakenAPI(config.kraken_api, api_key=api_key, api_secret=api_secret)


def _konfig_pfad(args: argparse.Namespace) -> Path:
    pfad = Path(args.config)
    if not pfad.exists():
        raise SystemExit(
            f"Konfigurationsdatei '{pfad}' nicht gefunden. "
            "Kopiere config.example.yaml nach config.yaml und passe sie an."
        )
    return pfad


def _frage_live_bestaetigung(config: AppConfig) -> str:
    """Zeigt Paare, Betragsgrenzen und Kapitalgrenze und fragt nach Bestaetigung."""
    print()
    print("=" * 70)
    print("  WARNUNG: LIVE-MODUS - es werden ECHTE Orders an Kraken gesendet!")
    print("=" * 70)
    print(f"  Watchlist:              {', '.join(config.scanner.watchlist)}")
    print(f"  Max. parallele Paare:   {config.scanner.max_parallele_paare}")
    print(f"  Betrag pro Kauf (Y):    {config.y.y_min_eur:.2f} - {config.y.y_max_eur:.2f} EUR")
    print(f"  Startkapital (Grenze):  {config.start_kapital_eur:.2f} EUR")
    print(f"  Max. Exposure/Paar:     {config.exposure.max_exposure_prozent:.1f} %")
    print(f"  Kill-Switch bei Verlust:{config.kill_switch.max_verlust_prozent:.1f} %")
    print("=" * 70)
    print("  Tippe JA (in Grossbuchstaben), um den Live-Handel zu starten.")
    print("  Alles andere bricht ab.")
    try:
        return input("  Bestaetigung: ").strip()
    except EOFError:
        return ""


def _erzeuge_broker(args: argparse.Namespace, config: AppConfig) -> tuple[Broker, KrakenAPI]:
    """Erzeugt den passenden Broker. Paper ist Default; Live nur mit allen Freigaben."""
    if getattr(args, "live", False):
        api = _api_aus_config(config, mit_schluessel=True)
        bestaetigung = _frage_live_bestaetigung(config)
        try:
            broker = KrakenLiveBroker(api, config.gebuehren, live_flag_gesetzt=True, nutzer_bestaetigung=bestaetigung)
        except LiveModusNichtErlaubt as exc:
            raise SystemExit(f"Live-Modus abgebrochen: {exc}")
        return broker, api

    if getattr(args, "validate", False):
        api = _api_aus_config(config, mit_schluessel=True)
        logger.info("Validate-Modus: Orders werden an Kraken gesendet, aber NICHT ausgefuehrt.")
        return KrakenValidateBroker(api, config.gebuehren), api

    api = _api_aus_config(config, mit_schluessel=False)
    logger.info("Paper-Modus: virtuelle Ausfuehrung, keine echten Orders.")
    return PaperBroker(config.gebuehren), api


# ---------- Befehle ----------


def befehl_scan(args: argparse.Namespace) -> int:
    config = lade_config(_konfig_pfad(args))
    api = _api_aus_config(config, mit_schluessel=False)

    try:
        infos = pruefe_watchlist(api, config.scanner.watchlist)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    fallback = profil_nach_name(config, config.zeitskala_auswahl.fallback_profil)
    bewertungen = {}
    for paar, info in infos.items():
        logger.info("Bewerte %s ...", paar)
        try:
            daten = hole_multiskalen_ohlc(api, info.kraken_name, config.hurst.multiskalen, config.pfade.daten_verzeichnis)
        except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
            logger.warning("Daten fuer %s nicht verfuegbar: %s", paar, exc)
            continue
        bewertungen[paar] = bewerte_regime(daten, config.hurst, config.gate, fallback, vorheriger_gate_zustand=False)

    if not bewertungen:
        logger.error("Kein Paar konnte bewertet werden.")
        return 1

    print()
    print(formatiere_ranking(ranke_paare(bewertungen)))
    print()
    print("Hinweis: 'scan' handelt nicht - es zeigt nur die aktuelle Regime-Bewertung.")
    return 0


def befehl_backtest(args: argparse.Namespace) -> int:
    config = lade_config(_konfig_pfad(args))
    api = _api_aus_config(config, mit_schluessel=False)
    profil = profil_nach_name(config, args.profil or config.zeitskala_auswahl.fallback_profil)

    try:
        df = hole_ohlc(api, args.pair, profil.intervall_minuten, config.pfade.daten_verzeichnis)
    except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
        logger.error("Konnte Kursdaten fuer %s nicht laden: %s", args.pair, exc)
        return 1

    if len(df) < 20:
        logger.error("Zu wenig Kerzen (%s) fuer einen sinnvollen Backtest.", len(df))
        return 1

    ergebnis = simulate(
        df, x_prozent=args.x, y_eur=args.y, gebuehr_prozent=config.gebuehren.taker_gebuehr_prozent,
        spread_prozent=config.gebuehren.geschaetzter_spread_prozent, start_kapital_eur=config.start_kapital_eur,
    )

    print()
    print(f"Backtest {args.pair} ({profil.name}-Kerzen, {len(df)} Kerzen von {df['zeit'].min()} bis {df['zeit'].max()})")
    print("=" * 70)
    print(f"  X (Schwelle):              {args.x:.4f} %")
    print(f"  Y (Betrag pro Kauf):       {args.y:.2f} EUR")
    print(f"  Startkapital:              {config.start_kapital_eur:.2f} EUR")
    print(f"  Netto-Gewinn:              {ergebnis.netto_gewinn_eur:+.2f} EUR")
    print(f"  Abgeschlossene Round-Trips:{ergebnis.anzahl_roundtrips:>6}")
    print(f"  Mittlere Amplitude:        {ergebnis.mittlere_amplitude_prozent:.4f} %")
    print(f"  Mittlere Haltedauer:       {ergebnis.mittlere_haltedauer_stunden:.2f} h")
    print(f"  Max. Drawdown:             {ergebnis.max_drawdown_prozent:.2f} %")
    print(f"  Offene Lots am Ende:       {ergebnis.offene_lots_anzahl}")
    print("=" * 70)

    bericht_verzeichnis = Path(config.pfade.log_verzeichnis)
    bericht_verzeichnis.mkdir(parents=True, exist_ok=True)
    csv_pfad = bericht_verzeichnis / f"backtest_{args.pair}_{profil.name}.csv"
    trades_df = pd.DataFrame(
        [
            {
                "zeit": t.zeit, "art": t.art, "preis": t.preis, "menge": t.menge,
                "gebuehr_eur": t.gebuehr_eur, "lot_einstandspreis": t.lot_einstandspreis,
            }
            for t in ergebnis.trades
        ]
    )
    trades_df.to_csv(csv_pfad, index=False)
    print(f"Trade-Liste geschrieben: {csv_pfad}")
    return 0


def befehl_optimize(args: argparse.Namespace) -> int:
    config = lade_config(_konfig_pfad(args))
    api = _api_aus_config(config, mit_schluessel=False)
    profil = profil_nach_name(config, args.profil or config.zeitskala_auswahl.fallback_profil)

    try:
        df = hole_ohlc(api, args.pair, profil.intervall_minuten, config.pfade.daten_verzeichnis)
    except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
        logger.error("Konnte Kursdaten fuer %s nicht laden: %s", args.pair, exc)
        return 1

    try:
        ergebnis = walk_forward_optimize(
            df, config.optimizer, config.gebuehren, y_eur=args.y, start_kapital=config.start_kapital_eur
        )
    except ValueError as exc:
        logger.error("Optimierung nicht moeglich: %s", exc)
        return 1

    bericht = textbericht(ergebnis, args.pair)
    print()
    print(bericht)

    bericht_verzeichnis = Path(config.pfade.log_verzeichnis)
    bericht_verzeichnis.mkdir(parents=True, exist_ok=True)
    csv_pfad = bericht_verzeichnis / f"optimierung_{args.pair}_{profil.name}.csv"
    txt_pfad = bericht_verzeichnis / f"optimierung_{args.pair}_{profil.name}.txt"
    fenster_ergebnisse_als_dataframe(ergebnis).to_csv(csv_pfad, index=False)
    txt_pfad.write_text(bericht, encoding="utf-8")
    print()
    print(f"Bericht geschrieben: {csv_pfad} und {txt_pfad}")
    return 0


def befehl_run(args: argparse.Namespace) -> int:
    config = lade_config(_konfig_pfad(args))
    broker, api = _erzeuge_broker(args, config)

    runner = BotRunner(
        config=config,
        api=api,
        broker=broker,
        zustand_pfad=Path(config.pfade.zustand_verzeichnis) / "bot_zustand.json",
        trade_log_pfad=Path(config.pfade.log_verzeichnis) / "trade_log.csv",
        zustand_log_pfad=Path(config.pfade.log_verzeichnis) / "zustands_log.jsonl",
    )

    try:
        runner.starte()
    except RuntimeError as exc:
        logger.error("Start fehlgeschlagen: %s", exc)
        return 1

    logger.info("Bot gestartet. Beenden mit Strg + C.")
    runner.laufe(max_zyklen=args.max_zyklen)
    return 0


# ---------- Argumente ----------


def baue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hurstbot", description="Hurst-basierter Grid-Trading-Bot fuer Kraken (Paper / Validate / Live)"
    )
    parser.add_argument("--config", default="config.yaml", help="Pfad zur Konfigurationsdatei (Default: config.yaml)")
    parser.add_argument("--debug", action="store_true", help="Ausfuehrliche Log-Ausgabe")

    unterbefehle = parser.add_subparsers(dest="befehl", required=True)

    p_scan = unterbefehle.add_parser("scan", help="Watchlist bewerten und Ranking ausgeben (kein Handel)")
    p_scan.set_defaults(funktion=befehl_scan)

    p_backtest = unterbefehle.add_parser("backtest", help="Einmalige Simulation mit festem X")
    p_backtest.add_argument("--pair", required=True, help="Handelspaar, z. B. XXBTZEUR")
    p_backtest.add_argument("--x", type=float, default=1.0, help="Schwelle X in Prozent (Default: 1.0)")
    p_backtest.add_argument("--y", type=float, default=20.0, help="Betrag pro Kauf in EUR (Default: 20)")
    p_backtest.add_argument("--profil", help="Name des Zeitskala-Profils (z. B. 5m, 15m, 1h)")
    p_backtest.set_defaults(funktion=befehl_backtest)

    p_optimize = unterbefehle.add_parser("optimize", help="Walk-Forward-X-Optimierung mit Bericht")
    p_optimize.add_argument("--pair", required=True, help="Handelspaar, z. B. XXBTZEUR")
    p_optimize.add_argument("--y", type=float, default=20.0, help="Betrag pro Kauf in EUR fuer die Simulation")
    p_optimize.add_argument("--profil", help="Name des Zeitskala-Profils (z. B. 5m, 15m, 1h)")
    p_optimize.set_defaults(funktion=befehl_optimize)

    p_run = unterbefehle.add_parser("run", help="Bot betreiben (Default: Paper, keine echten Orders)")
    modus = p_run.add_mutually_exclusive_group()
    modus.add_argument("--validate", action="store_true", help="Orders an Kraken senden, aber nur validieren")
    modus.add_argument("--live", action="store_true", help="ECHTE Orders senden (benoetigt alle drei Freigaben)")
    p_run.add_argument("--max-zyklen", dest="max_zyklen", type=int, help="Nach N Zyklen beenden (fuer Tests)")
    p_run.set_defaults(funktion=befehl_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    _lade_umgebung()
    parser = baue_parser()
    args = parser.parse_args(argv)
    richte_konsolen_logging_ein(logging.DEBUG if args.debug else logging.INFO)

    try:
        return args.funktion(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Abgebrochen durch Nutzer.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
