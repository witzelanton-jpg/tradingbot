"""Konfiguration des Bots: Laden, Validieren und Standardwerte.

Alle Parameter aus der Spezifikation sind hier als `pydantic`-Modelle abgebildet.
Die Konfiguration wird aus einer YAML-Datei geladen (siehe `config.example.yaml`)
und vollstaendig validiert, bevor der Bot startet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ZeitskalaProfil(BaseModel):
    """Ein Kandidat fuer die paarspezifische Zeitskala (Kerzenlaenge + Hurst-Fenster)."""

    name: str
    intervall_minuten: int = Field(gt=0)
    hurst_fenster: int = Field(gt=10)


class MultiSkalenConfig(BaseModel):
    """Feste Zeitfenster fuer die Mehrskalen-Hurst-Analyse (Abschnitt 2.1)."""

    kurz_intervall_minuten: int = 5
    kurz_anzahl_kerzen: int = 720  # ca. 60 h bei 5-Minuten-Kerzen
    mittel_intervall_minuten: int = 60
    mittel_anzahl_kerzen: int = 720  # ca. 30 Tage bei 1-Stunden-Kerzen
    lang_intervall_minuten: int = 1440
    lang_anzahl_kerzen: int = 365  # so viele Tageskerzen, wie Kraken liefert


class HurstConfig(BaseModel):
    """Parameter fuer die robuste Hurst-Schaetzung (Abschnitt 2.1)."""

    max_lag: int = 20
    min_lag: int = 2
    schaetzer: list[Literal["lag_std", "rs", "dfa"]] = Field(
        default_factory=lambda: ["lag_std", "rs"]
    )
    kombination: Literal["median", "mittelwert"] = "median"
    bootstrap_wiederholungen: int = 500
    bootstrap_blockgroesse: int = 20
    konfidenzniveau: float = 0.90
    multiskalen: MultiSkalenConfig = Field(default_factory=MultiSkalenConfig)


class GateConfig(BaseModel):
    """Handels-Gate mit Hysterese (Abschnitt 2.2)."""

    h_start: float = 0.45
    h_stop: float = 0.50

    @model_validator(mode="after")
    def _pruefe_reihenfolge(self) -> "GateConfig":
        if self.h_start >= self.h_stop:
            raise ValueError("h_start muss kleiner als h_stop sein")
        return self


class YConfig(BaseModel):
    """Betragsberechnung Y (Abschnitt 2.3)."""

    y_min_eur: float = 5.0
    y_max_eur: float = 50.0

    @model_validator(mode="after")
    def _pruefe_reihenfolge(self) -> "YConfig":
        if self.y_min_eur <= 0 or self.y_max_eur <= self.y_min_eur:
            raise ValueError("Es muss gelten: 0 < y_min_eur < y_max_eur")
        return self


class GebuehrenConfig(BaseModel):
    """Gebuehren- und Spread-Annahmen.

    Gerechnet wird durchgehend mit `taker_gebuehr_prozent` - auch dort, wo der
    Live-Broker Limit-Orders (Maker) schickt. Das ist die konservative Richtung:
    die Realitaet darf guenstiger ausfallen, nie teurer. `maker_gebuehr_prozent`
    dient nur der Dokumentation der eigenen Gebuehrenstufe.
    """

    maker_gebuehr_prozent: float = 0.16
    taker_gebuehr_prozent: float = 0.26
    geschaetzter_spread_prozent: float = 0.05
    sicherheitspuffer_prozent: float = 0.10


class XOptimizerConfig(BaseModel):
    """X-Optimierung per Walk-Forward-Backtest (Abschnitt 2.4)."""

    x_grid_min_prozent: float = 0.2
    x_grid_max_prozent: float = 5.0
    x_grid_schritte: int = 20
    verfeinerung_schritte: int = 10
    mit_optuna: bool = False
    optuna_versuche: int = 30
    neuoptimierung_stunden: float = 6.0
    glaettung_alpha: float = 0.3  # Exponentielle Glaettung neuer vs. alter X-Wert
    walk_forward_fenster_anzahl: int = 5
    walk_forward_test_anteil: float = 0.3
    max_drawdown_strafe: float = 1.0
    haltedauer_strafe_pro_stunde: float = 0.0005


class ZeitskalaAuswahlConfig(BaseModel):
    """Paarspezifische Zeitskalenauswahl (Abschnitt 2.5)."""

    kandidaten: list[ZeitskalaProfil] = Field(
        default_factory=lambda: [
            ZeitskalaProfil(name="5m", intervall_minuten=5, hurst_fenster=200),
            ZeitskalaProfil(name="15m", intervall_minuten=15, hurst_fenster=200),
            ZeitskalaProfil(name="1h", intervall_minuten=60, hurst_fenster=200),
        ]
    )
    neubewertung_stunden: float = 24.0
    wechsel_vorteil_prozent: float = 20.0
    mindest_roundtrips: int = 3
    fallback_profil: str = "5m"
    dominante_zyklusdauer_vorschlag: bool = False


class ExposureConfig(BaseModel):
    """Exposure-Limit (Abschnitt 3.1)."""

    max_exposure_prozent: float = 40.0


class AusbruchConfig(BaseModel):
    """Trend-/Ausbruchswaechter (Abschnitt 3.2)."""

    atr_periode: int = 14
    ausbruch_faktor_atr: float = 3.0
    ausbruch_prozent: float = 3.0
    ausbruch_zeitfenster_kerzen: int = 12
    extrem_sigma_faktor: float = 6.0  # MAD-basierte robuste Sigma-Schwelle


class WiedereinstiegConfig(BaseModel):
    """Wiedereinstiegslogik nach Pause (Abschnitt 3.3)."""

    toleranzband_prozent: float = 1.0
    neuanker_aktiv: bool = True
    neuanker_kerzen: int = 12
    neuanker_timeout_stunden: float = 48.0


class KillSwitchConfig(BaseModel):
    """Kill-Switch (Abschnitt 3.5)."""

    max_verlust_prozent: float = 10.0


class HurstWaechterConfig(BaseModel):
    """Hurst-Waechter im laufenden Betrieb (Abschnitt 3.4)."""

    bestand_abbauen: bool = False


class ScannerConfig(BaseModel):
    """Watchlist-Scanner (Abschnitt 4)."""

    watchlist: list[str] = Field(
        default_factory=lambda: ["XDGEUR", "XXBTZEUR", "XETHZEUR", "ADAEUR", "SOLEUR"]
    )
    max_parallele_paare: int = 2


class KrakenApiConfig(BaseModel):
    """Zugriff auf die Kraken-API."""

    basis_url: str = "https://api.kraken.com"
    api_version: str = "0"
    timeout_sekunden: float = 15.0
    max_wiederholungen: int = 5
    backoff_basis_sekunden: float = 1.0
    anfragen_pro_sekunde: float = 1.0  # konservatives Rate-Limit fuer public+private


class PfadeConfig(BaseModel):
    """Verzeichnisse fuer Daten, Logs und Zustand."""

    daten_verzeichnis: Path = Path("data")
    log_verzeichnis: Path = Path("logs")
    zustand_verzeichnis: Path = Path("state")

    @field_validator("daten_verzeichnis", "log_verzeichnis", "zustand_verzeichnis", mode="before")
    @classmethod
    def _zu_pfad(cls, wert: str | Path) -> Path:
        return Path(wert)


class AppConfig(BaseModel):
    """Gesamtkonfiguration des Bots."""

    start_kapital_eur: float = 100.0
    abfrage_sekunden: int = 60

    hurst: HurstConfig = Field(default_factory=HurstConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    y: YConfig = Field(default_factory=YConfig)
    gebuehren: GebuehrenConfig = Field(default_factory=GebuehrenConfig)
    optimizer: XOptimizerConfig = Field(default_factory=XOptimizerConfig)
    zeitskala_auswahl: ZeitskalaAuswahlConfig = Field(default_factory=ZeitskalaAuswahlConfig)
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    ausbruch: AusbruchConfig = Field(default_factory=AusbruchConfig)
    wiedereinstieg: WiedereinstiegConfig = Field(default_factory=WiedereinstiegConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    hurst_waechter: HurstWaechterConfig = Field(default_factory=HurstWaechterConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    kraken_api: KrakenApiConfig = Field(default_factory=KrakenApiConfig)
    pfade: PfadeConfig = Field(default_factory=PfadeConfig)

    model_config = {"extra": "forbid"}


def lade_config(pfad: str | Path) -> AppConfig:
    """Laedt und validiert die YAML-Konfigurationsdatei.

    Wirft `pydantic.ValidationError`, wenn Werte fehlerhaft sind, bzw.
    `FileNotFoundError`, wenn die Datei nicht existiert.
    """
    pfad = Path(pfad)
    if not pfad.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {pfad}")
    with open(pfad, "r", encoding="utf-8") as datei:
        rohdaten = yaml.safe_load(datei) or {}
    return AppConfig.model_validate(rohdaten)
