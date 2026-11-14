"""Idealisierte Kursreihen fuer hurstbot_test3.

    y(t) = P0 + A * sin(2*pi*f*t + phi) + Stoerung(t)

Die Reihe wird mit Minutenaufloesung erzeugt (ein Wert je Minute, gueltig fuer
diese Minute). Daraus werden Kerzen beliebiger Laenge (5m, 15m, 60m, 1440m)
genau so gebildet, wie Kraken sie liefert: Oeffnungszeit als Zeitstempel,
ausgerichtet auf volle Intervalle in UTC.

Stoerungen (beliebig kombinierbar):
  weiss        unabhaengiges Rauschen um die Kurve, Staerke = Standardabweichung in EUR je Minutenwert
  random_walk  der "feste" Preis wandert zufaellig, Staerke = Standardabweichung in EUR pro Tag
               (ueber die ganze Historie, so verschoben, dass der Weg zum Laufbeginn bei 0 steht)
  trend        stetige Drift ab einem Zeitpunkt, Staerke = EUR pro Tag (negativ = fallend)
  sprung       einmalige Niveauverschiebung ab einem Zeitpunkt, Staerke = EUR

Zeitpunkte ("ab_stunden") sind relativ zum Laufbeginn angegeben.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ARTEN = {
    "weiss": "Std.-Abw. in EUR je Minutenwert",
    "random_walk": "Std.-Abw. in EUR pro Tag",
    "trend": "Drift in EUR pro Tag ab 'ab_stunden'",
    "sprung": "Niveauverschiebung in EUR ab 'ab_stunden'",
}


@dataclass
class Stoerung:
    art: str
    staerke: float
    ab_stunden: float = 0.0  # nur fuer trend/sprung: Beginn relativ zum Laufbeginn

    def __post_init__(self) -> None:
        if self.art not in ARTEN:
            raise ValueError(f"Unbekannte Stoerung '{self.art}'. Moeglich: {', '.join(ARTEN)}")

    def beschreibung(self) -> str:
        if self.art == "weiss":
            return f"weisses Rauschen sigma={self.staerke:g} EUR"
        if self.art == "random_walk":
            return f"Random Walk {self.staerke:g} EUR/Tag"
        if self.art == "trend":
            return f"Trend {self.staerke:+g} EUR/Tag ab {self.ab_stunden:g} h"
        return f"Sprung {self.staerke:+g} EUR bei {self.ab_stunden:g} h"

    @classmethod
    def aus_text(cls, text: str) -> "Stoerung":
        """Kurzform fuer die Kommandozeile: 'art=staerke' oder 'art=staerke@ab_stunden'."""
        art, _, rest = text.partition("=")
        if not rest:
            raise ValueError(f"Stoerung '{text}' bitte als art=staerke[@ab_stunden] angeben")
        staerke, _, ab = rest.partition("@")
        return cls(art=art.strip(), staerke=float(staerke), ab_stunden=float(ab) if ab else 0.0)


@dataclass
class KursModell:
    basispreis: float = 100.0  # P0, der feste Preis, um den der Kurs schwingt (EUR)
    amplitude: float = 1.4  # A in EUR
    frequenz_pro_tag: float = 40.0  # f in Schwingungen pro Tag (40 = Periode 36 min)
    phase_grad: float = 0.0  # phi
    stoerungen: list[Stoerung] = field(default_factory=list)
    seed: int = 1

    @property
    def periode_minuten(self) -> float:
        return 1440.0 / self.frequenz_pro_tag

    def formel(self) -> str:
        teile = [f"{self.basispreis:g} + {self.amplitude:g}*sin(2*pi*{self.frequenz_pro_tag:g}/Tag*t)"]
        teile += [s.beschreibung() for s in self.stoerungen]
        return " + ".join(teile)


def erzeuge_minutenreihe(
    modell: KursModell, laufbeginn: pd.Timestamp, historie_tage: int, lauf_stunden: float
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Erzeugt die Minutenreihe von (Laufbeginn - Historie) bis (Laufende + 1 Tag Puffer).

    Rueckgabe: (t0 in Unix-Sekunden, Index des Laufbeginns, Kurs je Minute, Basispreis je Minute ohne Sinus).
    Der Start liegt auf 00:00 UTC, damit 5m/15m/60m/1440m-Kerzen sauber ausgerichtet sind.
    """
    laufbeginn = pd.Timestamp(laufbeginn).tz_convert("UTC").floor("min")
    start = laufbeginn.floor("D") - pd.Timedelta(days=historie_tage)
    ende = laufbeginn + pd.Timedelta(hours=lauf_stunden) + pd.Timedelta(days=1)
    n = int((ende - start).total_seconds() // 60)
    i_lauf = int((laufbeginn - start).total_seconds() // 60)
    t_min = np.arange(n, dtype=float) - i_lauf  # Minuten relativ zum Laufbeginn

    omega = 2 * np.pi * modell.frequenz_pro_tag / 1440.0
    sinus = modell.amplitude * np.sin(omega * t_min + np.deg2rad(modell.phase_grad))
    basis = np.full(n, modell.basispreis, dtype=float)
    rauschen = np.zeros(n)

    rng = np.random.default_rng(modell.seed)
    for s in modell.stoerungen:
        ab_min = s.ab_stunden * 60.0
        if s.art == "weiss":
            rauschen += s.staerke * rng.standard_normal(n)
        elif s.art == "random_walk":
            weg = np.cumsum(rng.standard_normal(n)) * s.staerke / np.sqrt(1440.0)
            basis += weg - weg[i_lauf]
        elif s.art == "trend":
            basis += s.staerke / 1440.0 * np.clip(t_min - ab_min, 0.0, None)
        elif s.art == "sprung":
            basis += s.staerke * (t_min >= ab_min)

    kurs = np.round(basis + sinus + rauschen, 5)
    if kurs.min() <= 0:
        raise ValueError(f"Der Kurs faellt auf {kurs.min():.4f} EUR - Preise muessen positiv sein (Stoerung zu stark?)")
    return int(start.timestamp()), i_lauf, kurs, basis


class SynthetischerMarkt:
    """Eine erzeugte Minutenreihe plus Kerzenbildung im Kraken-Format."""

    def __init__(self, t0: int, kurs: np.ndarray, basis: np.ndarray | None = None) -> None:
        self.t0 = int(t0)
        self.kurs = np.asarray(kurs, dtype=float)
        self.basis = np.asarray(basis, dtype=float) if basis is not None else None
        self._kerzen_cache: dict[int, dict[str, np.ndarray]] = {}

    # ---------- Speichern / Laden ----------

    def speichere(self, pfad: Path) -> None:
        np.savez_compressed(pfad, t0=np.array([self.t0]), kurs=self.kurs, basis=self.basis)

    @classmethod
    def lade(cls, pfad: Path) -> "SynthetischerMarkt":
        d = np.load(pfad)
        return cls(int(d["t0"][0]), d["kurs"], d["basis"] if "basis" in d else None)

    # ---------- Zugriff ----------

    def minuten_index(self, zeit_s: float) -> int:
        m = int((zeit_s - self.t0) // 60)
        if m < 0 or m >= len(self.kurs):
            raise ValueError("Zeitpunkt liegt ausserhalb der erzeugten Reihe")
        return m

    def preis(self, zeit_s: float) -> float:
        return float(self.kurs[self.minuten_index(zeit_s)])

    def basispreis(self, zeit_s: float) -> float | None:
        return None if self.basis is None else float(self.basis[self.minuten_index(zeit_s)])

    def _abgeschlossene_kerzen(self, k: int) -> dict[str, np.ndarray]:
        """Alle vollstaendigen Kerzen der Laenge k Minuten ueber die gesamte Reihe (einmal berechnet)."""
        if k not in self._kerzen_cache:
            anzahl = len(self.kurs) // k
            block = self.kurs[: anzahl * k].reshape(anzahl, k)
            self._kerzen_cache[k] = {
                "zeit": self.t0 + np.arange(anzahl) * k * 60,
                "open": block[:, 0],
                "high": block.max(axis=1),
                "low": block.min(axis=1),
                "close": block[:, -1],
                "vwap": block.mean(axis=1),
            }
        return self._kerzen_cache[k]

    def kerzen(
        self, k: int, bis_s: float, since_s: int | None = None, max_anzahl: int = 720, mit_laufender: bool = True
    ) -> tuple[list[list], int]:
        """Kerzen wie Kraken-OHLC zum Zeitpunkt `bis_s`.

        - abgeschlossene Kerzen mit Oeffnungszeit > since (ohne since: die juengsten),
        - plus die laufende, noch nicht abgeschlossene Kerze (Kraken: "the last entry in the
          OHLC array is for the current, not-yet-committed frame and will always be present,
          regardless of the value of since"),
        - hoechstens `max_anzahl` Eintraege (Kraken: 720).
        Rueckgabe: (Zeilen im Kraken-Format, 'last' = Oeffnungszeit der letzten abgeschlossenen Kerze).
        """
        m = self.minuten_index(bis_s)
        aktuell = m // k  # Index der laufenden Kerze
        voll = self._abgeschlossene_kerzen(k)
        i_von = 0
        if since_s is not None:
            i_von = max(0, int((since_s - self.t0) // (k * 60)) + 1)
        i_bis = aktuell  # exklusiv: nur abgeschlossene

        zeilen: list[list] = []
        for i in range(max(i_von, 0), i_bis):
            zeilen.append(_kraken_zeile(int(voll["zeit"][i]), voll["open"][i], voll["high"][i], voll["low"][i],
                                        voll["close"][i], voll["vwap"][i], k))
        if mit_laufender:
            teil = self.kurs[aktuell * k : m + 1]
            zeilen.append(_kraken_zeile(self.t0 + aktuell * k * 60, teil[0], teil.max(), teil.min(), teil[-1],
                                        teil.mean(), len(teil)))
        zeilen = zeilen[-max_anzahl:]
        last = int(self.t0 + (aktuell - 1) * k * 60)
        return zeilen, last

    def wahre_kerzen(self, k: int, von_s: float, bis_s: float) -> pd.DataFrame:
        """Vollstaendige (wahre) Kerzen im Zeitraum - zum Vergleich mit dem Cache des Bots."""
        voll = self._abgeschlossene_kerzen(k)
        df = pd.DataFrame(voll)
        df = df[(df["zeit"] >= von_s) & (df["zeit"] + k * 60 <= bis_s)].copy()
        df["zeit"] = pd.to_datetime(df["zeit"], unit="s", utc=True)
        return df.reset_index(drop=True)


def _kraken_zeile(zeit: int, o: float, h: float, l: float, c: float, vwap: float, minuten: int) -> list:
    return [zeit, f"{o:.5f}", f"{h:.5f}", f"{l:.5f}", f"{c:.5f}", f"{vwap:.5f}", f"{float(minuten):.8f}", int(minuten)]


def modell_aus_dict(d: dict) -> KursModell:
    d = dict(d)
    stoerungen = [Stoerung(**s) if isinstance(s, dict) else Stoerung.aus_text(s) for s in d.pop("stoerungen", []) or []]
    erlaubt = {"basispreis", "amplitude", "frequenz_pro_tag", "phase_grad", "seed"}
    unbekannt = set(d) - erlaubt
    if unbekannt:
        raise ValueError(f"Unbekannte Modellparameter: {', '.join(sorted(unbekannt))}")
    return KursModell(stoerungen=stoerungen, **d)


def modell_als_dict(modell: KursModell) -> dict:
    return json.loads(json.dumps(asdict(modell)))
