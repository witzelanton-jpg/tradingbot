"""Regime-Erkennung: Mehrskalen-Score, Handels-Gate mit Hysterese, Betrag Y.

Dieses Modul kombiniert die Hurst-Schaetzung aus `hurst.py` ueber mehrere
Zeitskalen (kurz/mittel/lang) zu einer Handelsentscheidung:

  - Das Gate (Abschnitt 2.2) entscheidet mit Hysterese, ob ein Paar gerade
    gehandelt werden darf. Es ist eine reine Funktion `vorheriger_zustand -> neuer_zustand`,
    damit der Aufrufer (Strategie/Scanner) den Zustand selbst persistiert.
  - Der `oszillations_score` in [0,1] wird unabhaengig vom Gate berechnet, damit
    der Scanner auch knapp nicht qualifizierte Paare einordnen kann.
  - `berechne_y_betrag` setzt den Score in einen EUR-Betrag innerhalb der
    konfigurierten und externen Grenzen um.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GateConfig, HurstConfig, YConfig, ZeitskalaProfil
from .hurst import HurstErgebnis, block_bootstrap_konfidenz, monotonie_score


@dataclass
class SkalenErgebnis:
    """Hurst- und Monotonie-Ergebnis fuer eine einzelne Zeitskala."""

    hurst: HurstErgebnis
    monotonie: dict[str, float]


@dataclass
class RegimeErgebnis:
    """Gesamtergebnis der Regime-Bewertung eines Paares."""

    kurz: SkalenErgebnis | None
    mittel: SkalenErgebnis | None
    lang: SkalenErgebnis | None
    oszillations_score: float
    handel_erlaubt: bool
    grund: str


def berechne_skalen_ergebnis(
    df: pd.DataFrame | None, hurst_config: HurstConfig, fenster: int, spalte: str = "close"
) -> SkalenErgebnis | None:
    """Berechnet Hurst + Monotonie fuer die letzten `fenster` Kerzen einer Zeitskala."""
    if df is None or len(df) < fenster:
        return None
    ausschnitt = df[spalte].iloc[-fenster:].to_numpy()
    hurst_ergebnis = block_bootstrap_konfidenz(ausschnitt, hurst_config)
    monotonie = monotonie_score(ausschnitt)
    return SkalenErgebnis(hurst=hurst_ergebnis, monotonie=monotonie)


def aktualisiere_gate(
    vorheriger_zustand: bool,
    kurz: SkalenErgebnis,
    mittel: SkalenErgebnis,
    gate_config: GateConfig,
) -> tuple[bool, str]:
    """Wendet die Gate-Hysterese an. Handel ist nur erlaubt, wenn kurz- UND
    mittelfristig die Kriterien erfuellt sind - dafuer wird jeweils der
    schlechtere (hoehere) Wert aus beiden Skalen verwendet."""
    h_effektiv = max(kurz.hurst.h, mittel.hurst.h)
    ci_oben_effektiv = max(kurz.hurst.ci_obere_grenze, mittel.hurst.ci_obere_grenze)

    # Absolute Sperre: unabhaengig von der Hysterese wird bei H >= 0.5 nie gehandelt.
    if h_effektiv >= 0.5 or ci_oben_effektiv >= 0.5:
        return False, f"H bzw. obere Konfidenzgrenze >= 0.5 (H={h_effektiv:.3f}, CI_oben={ci_oben_effektiv:.3f})"

    if h_effektiv < gate_config.h_start:
        return True, f"H={h_effektiv:.3f} < H_START={gate_config.h_start}"
    if h_effektiv >= gate_config.h_stop:
        return False, f"H={h_effektiv:.3f} >= H_STOP={gate_config.h_stop}"

    return (
        vorheriger_zustand,
        f"Hysterese-Zone ({gate_config.h_start} <= H={h_effektiv:.3f} < {gate_config.h_stop}): "
        f"Zustand '{'HANDELT' if vorheriger_zustand else 'PAUSIERT'}' bleibt erhalten",
    )


def berechne_oszillations_score(
    h_effektiv: float,
    gate_config: GateConfig,
    konfidenz_effektiv: float,
    monotonie_effektiv: float,
    skalen_abweichung: float,
    langfrist_faktor: float,
) -> float:
    """Kombiniert H (klein = gut), Konfidenz, Monotonie und Mehrskalen-Uebereinstimmung
    zu einem Score in [0,1]. Je kleiner H und je sicherer/eindeutiger die Lage, desto hoeher."""
    if h_effektiv >= gate_config.h_stop:
        return 0.0
    h_score = float(np.clip((gate_config.h_stop - h_effektiv) / gate_config.h_stop, 0.0, 1.0))
    skalen_score = float(np.clip(1 - skalen_abweichung / gate_config.h_stop, 0.0, 1.0))
    monotonie_faktor = 1 - monotonie_effektiv
    score = h_score * konfidenz_effektiv * monotonie_faktor * skalen_score * langfrist_faktor
    return float(np.clip(score, 0.0, 1.0))


def bewerte_regime(
    daten: dict[str, pd.DataFrame | None],
    hurst_config: HurstConfig,
    gate_config: GateConfig,
    zeitskala_profil: ZeitskalaProfil,
    vorheriger_gate_zustand: bool,
) -> RegimeErgebnis:
    """Fuehrt die vollstaendige Mehrskalen-Regimebewertung fuer ein Paar aus."""
    kurz = berechne_skalen_ergebnis(daten.get("kurz"), hurst_config, zeitskala_profil.hurst_fenster)
    mittel = berechne_skalen_ergebnis(daten.get("mittel"), hurst_config, zeitskala_profil.hurst_fenster)

    lang_df = daten.get("lang")
    lang_fenster = min(zeitskala_profil.hurst_fenster, len(lang_df)) if lang_df is not None else 0
    lang = berechne_skalen_ergebnis(lang_df, hurst_config, lang_fenster) if lang_fenster >= 20 else None

    if kurz is None or mittel is None:
        return RegimeErgebnis(
            kurz=kurz,
            mittel=mittel,
            lang=lang,
            oszillations_score=0.0,
            handel_erlaubt=False,
            grund="Nicht genug Kerzen fuer kurz- oder mittelfristige Hurst-Analyse",
        )

    handel_erlaubt, grund = aktualisiere_gate(vorheriger_gate_zustand, kurz, mittel, gate_config)

    h_effektiv = max(kurz.hurst.h, mittel.hurst.h)
    konfidenz_effektiv = min(kurz.hurst.konfidenz, mittel.hurst.konfidenz)
    monotonie_effektiv = max(kurz.monotonie["monotonie_score"], mittel.monotonie["monotonie_score"])
    skalen_abweichung = abs(kurz.hurst.h - mittel.hurst.h)

    langfrist_faktor = 1.0
    if lang is not None:
        trend_ueberschuss = max(0.0, lang.hurst.h - gate_config.h_stop)
        langfrist_faktor = float(np.clip(1 - 2 * trend_ueberschuss, 0.0, 1.0))
        langfrist_faktor *= float(np.clip(1 - 0.5 * lang.monotonie["monotonie_score"], 0.0, 1.0))

    score = berechne_oszillations_score(
        h_effektiv, gate_config, konfidenz_effektiv, monotonie_effektiv, skalen_abweichung, langfrist_faktor
    )

    return RegimeErgebnis(kurz=kurz, mittel=mittel, lang=lang, oszillations_score=score, handel_erlaubt=handel_erlaubt, grund=grund)


def berechne_y_betrag(
    score: float,
    konfidenz: float,
    y_config: YConfig,
    verfuegbares_kapital_eur: float,
    exposure_limit_eur: float,
    mindest_betrag_eur: float,
) -> float:
    """Setzt Score und Konfidenz in einen Kaufbetrag Y um, begrenzt durch Kapital,
    Exposure-Limit und Kraken-Mindestordergroesse (in EUR umgerechnet).

    Ist H unsicher oder nahe 0.5 (score oder konfidenz klein), bleibt Y nahe y_min.
    Reicht das verfuegbare Kapital/Limit nicht einmal fuer die Mindestordergroesse,
    wird 0 zurueckgegeben (kein Kauf moeglich).
    """
    f = float(np.clip(score, 0.0, 1.0)) * float(np.clip(konfidenz, 0.0, 1.0))
    y = y_config.y_min_eur + (y_config.y_max_eur - y_config.y_min_eur) * f
    y = min(y, verfuegbares_kapital_eur, exposure_limit_eur)
    if y < mindest_betrag_eur:
        return 0.0
    return float(y)
