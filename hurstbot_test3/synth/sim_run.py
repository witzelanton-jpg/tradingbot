"""Simulierter Paper-Run ("hurstbot run" mit simulierter Uhr) auf einem Szenario-Ordner.

Der Bot wird exakt so zusammengesetzt wie in `hurstbot.cli.befehl_run` fuer den
Paper-Modus: Config laden, `PaperBroker(config.gebuehren)`, `BotRunner(...)` mit
denselben Pfaden fuer Zustand, Trade-Log und Zustands-Log, dann `starte()`.

Nur zwei Dinge sind anders als beim echten Befehl:
  1. Statt `KrakenAPI` bekommt der Runner die `SynthetischeKrakenAPI` (gleiche
     oeffentliche Methoden und Antwortformate, Kurse aus markt.npz).
  2. Statt `runner.laufe()` (Wanduhr + time.sleep) stellt dieser Treiber eine
     simulierte Uhr je Zyklus um `abfrage_sekunden` weiter und ruft
     `runner.fuehre_zyklus_aus(jetzt)` auf - dieselbe Methode, die `laufe()`
     aufruft. Abbruch bei GESTOPPT und `beende_sauber()` am Ende wie in `laufe()`.

Zusaetzlich wird je Zyklus mitgeschrieben, was der Bot gesehen und entschieden
hat (zyklen.csv). Dafuer wird `runner.bewerte_paar` von aussen umhuellt: das
Ergebnis wird nur kopiert, nicht veraendert.

Aufruf:  python synth/sim_run.py laeufe/F1_weiss [--max-zyklen N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from hurstbot.broker import PaperBroker
from hurstbot.config import lade_config
from hurstbot.logging_setup import richte_konsolen_logging_ein
from hurstbot.runner import BotRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_kraken import SimUhr, SynthetischeKrakenAPI  # noqa: E402
from kursmodell import SynthetischerMarkt  # noqa: E402

logger = logging.getLogger("hurstbot_test3.sim_run")


def _wert(x, nachkomma=4):
    return None if x is None else round(float(x), nachkomma)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulierter Paper-Run auf synthetischen Daten")
    parser.add_argument("ordner", type=Path, help="Szenario-Ordner (laeufe/<name>)")
    parser.add_argument("--max-zyklen", type=int, help="nach N Zyklen beenden (Kurztest)")
    args = parser.parse_args()

    ordner = args.ordner.resolve()
    os.chdir(ordner)  # Pfade data/logs/state der Config sind relativ zum Szenario-Ordner
    info = json.loads(Path("szenario.json").read_text(encoding="utf-8"))
    paar = info["paar"]

    richte_konsolen_logging_ein(logging.INFO)
    config = lade_config(Path("config.yaml"))

    # --- wie cli.befehl_run im Paper-Modus, nur mit synthetischer API ---
    uhr = SimUhr(pd.Timestamp(info["laufbeginn"]))
    markt = SynthetischerMarkt.lade(Path("markt.npz"))
    api = SynthetischeKrakenAPI({paar: markt}, uhr)
    broker = PaperBroker(config.gebuehren)
    logger.info("Paper-Modus: virtuelle Ausfuehrung, keine echten Orders. Kurse: synthetisch (%s)", info["formel"])
    runner = BotRunner(
        config=config,
        api=api,
        broker=broker,
        zustand_pfad=Path(config.pfade.zustand_verzeichnis) / "bot_zustand.json",
        trade_log_pfad=Path(config.pfade.log_verzeichnis) / "trade_log.csv",
        zustand_log_pfad=Path(config.pfade.log_verzeichnis) / "zustands_log.jsonl",
    )
    runner.starte()

    # --- Beobachtung: Regime-Ergebnis je Zyklus mitschreiben (unveraendert durchgereicht) ---
    letzte_bewertung: dict = {}
    original_bewerte_paar = runner.bewerte_paar

    def bewerte_und_merke(p):
        ergebnis = original_bewerte_paar(p)
        letzte_bewertung[p] = ergebnis
        return ergebnis

    runner.bewerte_paar = bewerte_und_merke

    schritt = pd.Timedelta(seconds=config.abfrage_sekunden)
    ende = pd.Timestamp(info["laufende"])
    zyklen: list[dict] = []
    t_lauf = time.perf_counter()
    logger.info("Bot gestartet (simulierte Uhr ab %s, Schritt %s, bis %s)", uhr.jetzt, schritt, ende)

    zyklus = 0
    while uhr.jetzt < ende and (args.max_zyklen is None or zyklus < args.max_zyklen):
        if runner.bot_zustand.global_zustand == "GESTOPPT":
            logger.critical("Bot ist GESTOPPT (Kill-Switch) - keine weiteren Zyklen.")
            break
        t0 = time.perf_counter()
        letzte_bewertung.pop(paar, None)
        runner.fuehre_zyklus_aus(jetzt=uhr.jetzt)
        dauer = time.perf_counter() - t0

        s = runner.strategien[paar]
        r = letzte_bewertung.get(paar)
        preis = markt.preis(uhr.sekunden())
        lots_menge = sum(l.menge for l in s.lots)
        zyklen.append({
            "zeit": uhr.jetzt.isoformat(),
            "preis": preis,
            "basispreis": _wert(markt.basispreis(uhr.sekunden())),
            "gate": None if r is None else bool(r.handel_erlaubt),
            "score": None if r is None else _wert(r.oszillations_score),
            "h_kurz": _wert(r.kurz.hurst.h) if r and r.kurz else None,
            "ci_kurz_oben": _wert(r.kurz.hurst.ci_obere_grenze) if r and r.kurz else None,
            "h_mittel": _wert(r.mittel.hurst.h) if r and r.mittel else None,
            "ci_mittel_oben": _wert(r.mittel.hurst.ci_obere_grenze) if r and r.mittel else None,
            "h_lang": _wert(r.lang.hurst.h) if r and r.lang else None,
            "konfidenz": _wert(min(r.kurz.hurst.konfidenz, r.mittel.hurst.konfidenz)) if r and r.kurz and r.mittel else None,
            "grund": None if r is None else r.grund,
            "zustand": s.zustand.value,
            "x_prozent": _wert(s.x_prozent),
            "profil": s.zeitskala_profil.name,
            "anker": _wert(s.anker_preis),
            "lots": len(s.lots),
            "lots_menge": _wert(lots_menge, 8),
            "kapital_paar": _wert(s.kapital_eur),
            "kapital_frei": _wert(runner.bot_zustand.kapital_frei_eur),
            "wert_bot_buchwert": _wert(runner.gesamtwert_bot()),  # so rechnet der Bot (Lots zum Kaufpreis)
            "wert_marktwert": _wert(runner.bot_zustand.kapital_frei_eur + s.kapital_eur + lots_menge * preis),
            "global": runner.bot_zustand.global_zustand,
            "dauer_s": round(dauer, 2),
        })
        zyklus += 1
        if zyklus % 20 == 0:
            pd.DataFrame(zyklen).to_csv("zyklen.csv", index=False)
            print(f"### Fortschritt {info['name']}: Zyklus {zyklus}, Sim-Zeit {uhr.jetzt}, "
                  f"{time.perf_counter() - t_lauf:.0f} s gelaufen", flush=True)
        uhr.jetzt = uhr.jetzt + schritt

    runner.beende_sauber()
    pd.DataFrame(zyklen).to_csv("zyklen.csv", index=False)
    lauf = {
        "zyklen": zyklus,
        "sim_von": info["laufbeginn"],
        "sim_bis": uhr.jetzt.isoformat(),
        "schritt_sekunden": config.abfrage_sekunden,
        "rechenzeit_s": round(time.perf_counter() - t_lauf, 1),
        "api_anfragen": api.anfragen,
        "global_zustand": runner.bot_zustand.global_zustand,
    }
    Path("lauf.json").write_text(json.dumps(lauf, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Lauf beendet: %s", lauf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
