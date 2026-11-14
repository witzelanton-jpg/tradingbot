"""Atomare Zustandspersistenz (JSON).

Nach jeder Orderausfuehrung wird der komplette Zustand neu geschrieben - erst in
eine temporaere Datei im selben Verzeichnis, dann per `os.replace` umbenannt.
`os.replace` ist auf allen unterstuetzten Plattformen atomar, sodass ein Absturz
mitten im Schreiben nie eine halb geschriebene/korrupte Zustandsdatei hinterlaesst.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import Lot

logger = logging.getLogger("hurstbot.state")


def atomar_schreiben(pfad: Path, inhalt: str) -> None:
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    temp_pfad = pfad.with_name(f"{pfad.name}.tmp.{os.getpid()}")
    with open(temp_pfad, "w", encoding="utf-8") as datei:
        datei.write(inhalt)
        datei.flush()
        os.fsync(datei.fileno())
    os.replace(temp_pfad, pfad)


def _lot_zu_dict(lot: Lot) -> dict:
    return {
        "einstandspreis": lot.einstandspreis,
        "fill_preis_kauf": lot.fill_preis_kauf,
        "menge": lot.menge,
        "kauf_zeit": lot.kauf_zeit.isoformat(),
    }


def _dict_zu_lot(d: dict) -> Lot:
    return Lot(
        einstandspreis=d["einstandspreis"],
        fill_preis_kauf=d["fill_preis_kauf"],
        menge=d["menge"],
        kauf_zeit=pd.Timestamp(d["kauf_zeit"]),
    )


@dataclass
class PaarZustandsDaten:
    """Persistenter Zustand eines einzelnen Handelspaares."""

    paar: str
    zustand: str = "HANDELT"
    lots: list[Lot] = field(default_factory=list)
    kapital_eur: float = 0.0
    anker_preis: float | None = None
    gate_handelt: bool = False
    zeitskala_profil_name: str = "5m"
    letzte_zeitskala_neubewertung: str | None = None
    x_prozent: float | None = None
    letzte_x_optimierung: str | None = None
    pausiert_seit: str | None = None
    anker_preis_bei_pause: float | None = None
    stabile_kerzen_zaehler: int = 0
    wiedereinstieg_warnung_ausgegeben: bool = False

    def zu_dict(self) -> dict:
        d = asdict(self)
        d["lots"] = [_lot_zu_dict(l) for l in self.lots]
        return d

    @classmethod
    def aus_dict(cls, d: dict) -> "PaarZustandsDaten":
        d = dict(d)
        lots_roh = d.pop("lots", [])
        instanz = cls(**d)
        instanz.lots = [_dict_zu_lot(l) for l in lots_roh]
        return instanz


@dataclass
class BotZustand:
    """Gesamtzustand des Bots: globales (freies) Kapital + Zustand je Paar."""

    start_kapital_eur: float
    kapital_frei_eur: float
    paare: dict[str, PaarZustandsDaten] = field(default_factory=dict)
    global_zustand: str = "HANDELT"
    erstellt_am: str = ""

    def zu_dict(self) -> dict:
        return {
            "start_kapital_eur": self.start_kapital_eur,
            "kapital_frei_eur": self.kapital_frei_eur,
            "global_zustand": self.global_zustand,
            "erstellt_am": self.erstellt_am,
            "paare": {name: p.zu_dict() for name, p in self.paare.items()},
        }

    @classmethod
    def aus_dict(cls, d: dict) -> "BotZustand":
        paare = {name: PaarZustandsDaten.aus_dict(p) for name, p in d.get("paare", {}).items()}
        return cls(
            start_kapital_eur=d["start_kapital_eur"],
            kapital_frei_eur=d["kapital_frei_eur"],
            paare=paare,
            global_zustand=d.get("global_zustand", "HANDELT"),
            erstellt_am=d.get("erstellt_am", ""),
        )


def speichere_zustand(zustand: BotZustand, pfad: Path) -> None:
    atomar_schreiben(Path(pfad), json.dumps(zustand.zu_dict(), indent=2, ensure_ascii=False))


def lade_zustand(pfad: Path) -> BotZustand | None:
    """Laedt den Zustand. Gibt bei fehlender oder beschaedigter Datei None zurueck
    (kein Absturz), damit der Aufrufer bewusst entscheiden kann, frisch zu starten."""
    pfad = Path(pfad)
    if not pfad.exists():
        return None
    try:
        with open(pfad, "r", encoding="utf-8") as datei:
            daten = json.load(datei)
        return BotZustand.aus_dict(daten)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Zustandsdatei %s ist beschaedigt oder unvollstaendig: %s", pfad, exc)
        return None
