"""Logging-Einrichtung: Konsole (Deutsch), CSV-Trade-Log und Zustands-Log (JSONL)."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .backtest import Trade


def richte_konsolen_logging_ein(level: int = logging.INFO) -> None:
    """Konfiguriert das Root-Logging fuer die Konsolenausgabe."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def protokolliere_trade(pfad: Path, paar: str, trade: Trade) -> None:
    """Haengt einen Trade an das CSV-Trade-Log an (legt bei Bedarf die Kopfzeile an)."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    neu = not pfad.exists()
    with open(pfad, "a", newline="", encoding="utf-8") as datei:
        writer = csv.writer(datei)
        if neu:
            writer.writerow(["zeitpunkt", "paar", "art", "preis_eur", "menge", "gebuehr_eur", "lot_einstandspreis"])
        zeit = trade.zeit.isoformat() if isinstance(trade.zeit, pd.Timestamp) else str(trade.zeit)
        writer.writerow(
            [
                zeit,
                paar,
                trade.art,
                f"{trade.preis:.8f}",
                f"{trade.menge:.8f}",
                f"{trade.gebuehr_eur:.8f}",
                f"{trade.lot_einstandspreis:.8f}" if trade.lot_einstandspreis else "",
            ]
        )


def protokolliere_zustandswechsel(
    pfad: Path, paar: str, alter_zustand: str, neuer_zustand: str, grund: str, zeitpunkt: pd.Timestamp
) -> None:
    """Haengt einen Zustandswechsel (HANDELT/PAUSIERT/NUR_VERKAUFEN/GESTOPPT) als JSON-Zeile an."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    eintrag = {
        "zeitpunkt": zeitpunkt.isoformat() if hasattr(zeitpunkt, "isoformat") else str(zeitpunkt),
        "paar": paar,
        "alter_zustand": alter_zustand,
        "neuer_zustand": neuer_zustand,
        "grund": grund,
    }
    with open(pfad, "a", encoding="utf-8") as datei:
        datei.write(json.dumps(eintrag, ensure_ascii=False) + "\n")

    logger = logging.getLogger("hurstbot.zustand")
    if alter_zustand != neuer_zustand:
        logger.warning("Paar %s: %s -> %s (%s)", paar, alter_zustand, neuer_zustand, grund)
