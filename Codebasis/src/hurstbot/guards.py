"""Schutzmechanismen ("Trendwaechter"): Exposure-Limit, Ausbruchs-/Trendwaechter
mit Wiedereinstiegslogik, Hurst-Waechter und Kill-Switch (Abschnitt 3).

Alle Funktionen sind bewusst zustandsarm gehalten (reine Entscheidungsfunktionen);
den Zustand pro Paar (PAUSIERT seit wann, wie viele stabile Kerzen, etc.) haelt
die aufrufende Strategie (`strategy.py`), damit er zusammen mit den Lots
atomar gespeichert werden kann.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .config import AusbruchConfig, ExposureConfig, KillSwitchConfig, WiedereinstiegConfig


class PaarZustand(str, Enum):
    """Zustandsautomat eines Handelspaares."""

    HANDELT = "HANDELT"
    PAUSIERT = "PAUSIERT"
    NUR_VERKAUFEN = "NUR_VERKAUFEN"
    GESTOPPT = "GESTOPPT"


def ist_kauf_durch_exposure_blockiert(investierter_wert_eur: float, gesamtwert_eur: float, config: ExposureConfig) -> bool:
    """Exposure-Limit (3.1): blockiert nur weitere KAEUFE, Verkaeufe bleiben erlaubt."""
    if gesamtwert_eur <= 0:
        return True
    exposure_prozent = investierter_wert_eur / gesamtwert_eur * 100
    return exposure_prozent >= config.max_exposure_prozent


def berechne_atr(
    df: pd.DataFrame, periode: int, hoch_spalte: str = "high", tief_spalte: str = "low", schluss_spalte: str = "close"
) -> pd.Series:
    """Average True Range: Standardmass fuer die durchschnittliche Kursschwankung pro Kerze."""
    vorheriger_schluss = df[schluss_spalte].shift(1)
    spanne_1 = df[hoch_spalte] - df[tief_spalte]
    spanne_2 = (df[hoch_spalte] - vorheriger_schluss).abs()
    spanne_3 = (df[tief_spalte] - vorheriger_schluss).abs()
    true_range = pd.concat([spanne_1, spanne_2, spanne_3], axis=1).max(axis=1)
    return true_range.rolling(window=periode).mean()


def robuste_sigma_mad(renditen: pd.Series) -> float:
    """Robuste Streuungsschaetzung ueber den Median Absolute Deviation (MAD),
    skaliert auf Normalverteilungs-Konsistenz (Faktor 1.4826)."""
    if len(renditen) == 0:
        return 0.0
    median = renditen.median()
    mad = (renditen - median).abs().median()
    return float(mad * 1.4826)


@dataclass
class AusbruchErgebnis:
    ausgeloest: bool
    grund: str


def pruefe_ausbruch(
    aktueller_preis: float,
    anker_preis: float,
    letzte_preise: pd.Series,
    atr_wert: float | None,
    config: AusbruchConfig,
) -> AusbruchErgebnis:
    """Prueft die drei Ausbruchs-Kriterien aus Abschnitt 3.2 (ODER-verknuepft):
    1. Abweichung vom Anker > ausbruch_faktor_atr * ATR
    2. Abweichung vom Anker > ausbruch_prozent (gemessen als reine Preisabweichung)
    3. Einzelkerze mit extremer Bewegung (> k * robuste MAD-Sigma der juengsten Renditen)
    """
    if anker_preis <= 0:
        return AusbruchErgebnis(False, "ungueltiger Ankerpreis")

    abweichung_prozent = abs(aktueller_preis - anker_preis) / anker_preis * 100

    if atr_wert and atr_wert > 0:
        abweichung_atr = abs(aktueller_preis - anker_preis) / atr_wert
        if abweichung_atr >= config.ausbruch_faktor_atr:
            return AusbruchErgebnis(
                True, f"Abweichung {abweichung_atr:.2f}x ATR >= Ausbruchsfaktor {config.ausbruch_faktor_atr}"
            )

    if abweichung_prozent >= config.ausbruch_prozent:
        return AusbruchErgebnis(
            True, f"Abweichung vom Anker {abweichung_prozent:.2f}% >= {config.ausbruch_prozent}%"
        )

    if len(letzte_preise) >= 3:
        renditen = letzte_preise.pct_change().dropna() * 100
        if len(renditen) >= 2:
            sigma = robuste_sigma_mad(renditen)
            letzte_rendite = float(renditen.iloc[-1])
            if sigma > 0 and abs(letzte_rendite) >= config.extrem_sigma_faktor * sigma:
                return AusbruchErgebnis(
                    True,
                    f"Einzelkerzen-Ausschlag {letzte_rendite:.2f}% >= {config.extrem_sigma_faktor}x "
                    f"robuste Sigma ({sigma:.3f}%)",
                )

    return AusbruchErgebnis(False, "keine Ausbruchsbedingung erfuellt")


@dataclass
class WiedereinstiegZustand:
    """Zustand der Wiedereinstiegslogik fuer ein pausiertes Paar."""

    pausiert_seit: pd.Timestamp
    anker_preis_bei_pause: float
    stabile_kerzen_zaehler: int = 0
    warnung_ausgegeben: bool = False


@dataclass
class WiedereinstiegErgebnis:
    wieder_einsteigen: bool
    neuer_anker_preis: float
    grund: str


def pruefe_wiedereinstieg(
    zustand: WiedereinstiegZustand,
    aktueller_preis: float,
    aktuelle_zeit: pd.Timestamp,
    h_aktuell: float,
    h_start: float,
    config: WiedereinstiegConfig,
) -> WiedereinstiegErgebnis:
    """Abschnitt 3.3: Wiedereinstieg erst, wenn der Kurs ins urspruengliche
    Toleranzband zurueckgekehrt ist. Optionaler Fallback: Neuanker nach
    `neuanker_kerzen` stabilen Kerzen in neuer Spanne, wenn H < H_START."""
    abweichung_prozent = abs(aktueller_preis - zustand.anker_preis_bei_pause) / zustand.anker_preis_bei_pause * 100

    if abweichung_prozent <= config.toleranzband_prozent:
        return WiedereinstiegErgebnis(True, zustand.anker_preis_bei_pause, "Kurs zurueck im urspruenglichen Toleranzband")

    if config.neuanker_aktiv:
        ist_stabil = abweichung_prozent <= config.toleranzband_prozent * 3 and h_aktuell < h_start
        zustand.stabile_kerzen_zaehler = zustand.stabile_kerzen_zaehler + 1 if ist_stabil else 0

        if zustand.stabile_kerzen_zaehler >= config.neuanker_kerzen:
            return WiedereinstiegErgebnis(
                True,
                aktueller_preis,
                f"Neuanker gesetzt nach {zustand.stabile_kerzen_zaehler} stabilen Kerzen (H={h_aktuell:.3f} < H_START)",
            )

    zeit_seit_pause_stunden = (aktuelle_zeit - zustand.pausiert_seit).total_seconds() / 3600
    if zeit_seit_pause_stunden >= config.neuanker_timeout_stunden and not zustand.warnung_ausgegeben:
        zustand.warnung_ausgegeben = True
        return WiedereinstiegErgebnis(
            False,
            zustand.anker_preis_bei_pause,
            f"WARNUNG: Paar seit {zeit_seit_pause_stunden:.1f} h pausiert "
            f"(Timeout {config.neuanker_timeout_stunden} h ueberschritten), noch keine Wiedereinstiegs-Bedingung erfuellt",
        )

    return WiedereinstiegErgebnis(False, zustand.anker_preis_bei_pause, "warte auf Rueckkehr ins Toleranzband")


def ist_kill_switch_ausgeloest(gesamtwert_eur: float, start_kapital_eur: float, config: KillSwitchConfig) -> bool:
    """Abschnitt 3.5: Gesamtwert faellt unter -MAX_VERLUST_PROZENT vom Startkapital."""
    if start_kapital_eur <= 0:
        return False
    verlust_prozent = (start_kapital_eur - gesamtwert_eur) / start_kapital_eur * 100
    return verlust_prozent >= config.max_verlust_prozent
