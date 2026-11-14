"""Robuste Schaetzung des Hurst-Exponenten mit Konfidenzintervall und Monotonie-Mass.

  H < 0.5   -> Kurs neigt zum Oszillieren/Zurueckpendeln (gut fuer den Grid-Bot)
  H ~ 0.5   -> kein erkennbares Muster (Zufallsbewegung)
  H > 0.5   -> Kurs neigt zum Trend (schlecht fuer den Grid-Bot)

Es werden mehrere unabhaengige Schaetzer kombiniert (Median/Mittelwert), die
Unsicherheit per Block-Bootstrap quantifiziert, und zusaetzlich ein Monotonie-Mass
berechnet, das reine Trendbewegungen erkennt, die der Hurst-Exponent allein
manchmal nicht klar genug straft.

Alle Schaetzer rechnen auf Log-Preisen (bzw. deren Renditen), nicht auf
Rohpreisen, wie in der Spezifikation gefordert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HurstConfig


@dataclass
class HurstErgebnis:
    """Ergebnis der robusten Hurst-Schaetzung fuer eine einzelne Zeitskala."""

    h: float
    ci_untere_grenze: float
    ci_obere_grenze: float
    konfidenz: float  # in [0, 1]: 1 = sehr sicher (schmales Intervall), 0 = sehr unsicher


def zu_log_preisen(preise: np.ndarray | pd.Series) -> np.ndarray:
    """Wandelt eine Preisreihe in Log-Preise um. Preise muessen > 0 sein."""
    preise = np.asarray(preise, dtype=float)
    if len(preise) == 0:
        raise ValueError("Preisreihe ist leer")
    if np.any(preise <= 0):
        raise ValueError("Preise muessen positiv sein fuer die Log-Transformation")
    return np.log(preise)


def hurst_lag_std(log_preise: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """Schaetzer ueber die Standardabweichung der Differenzen bei wachsendem Lag.

    Erweiterung von `regime_signal.hurst_exponent`: rechnet auf Log-Preisen statt
    Rohpreisen, sonst identisches Prinzip (Steigung im Log-Log-Raum).
    """
    log_preise = np.asarray(log_preise, dtype=float)
    max_lag = min(max_lag, len(log_preise) - 1)
    if max_lag <= min_lag:
        return float("nan")

    lags = np.arange(min_lag, max_lag)
    tau = np.array([np.std(log_preise[lag:] - log_preise[:-lag]) for lag in lags])
    gueltig = tau > 0
    if gueltig.sum() < 2:
        return float("nan")
    steigung = np.polyfit(np.log(lags[gueltig].astype(float)), np.log(tau[gueltig]), 1)[0]
    return float(steigung)


def hurst_rs(log_preise: np.ndarray, min_lag: int = 10, max_lag: int | None = None, anzahl_fenster: int = 20) -> float:
    """Klassische Rescaled-Range-(R/S)-Analyse auf den Log-Renditen."""
    log_preise = np.asarray(log_preise, dtype=float)
    renditen = np.diff(log_preise)
    n = len(renditen)
    if n < min_lag * 2:
        return float("nan")

    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max(max_lag, min_lag + 1), n // 2)
    if max_lag <= min_lag:
        return float("nan")

    fenstergroessen = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), num=anzahl_fenster).astype(int)
    )
    fenstergroessen = fenstergroessen[fenstergroessen >= min_lag]

    rs_werte: list[float] = []
    gueltige_fenster: list[int] = []
    for fenster in fenstergroessen:
        anzahl_segmente = n // fenster
        if anzahl_segmente < 1:
            continue
        rs_liste = []
        for i in range(anzahl_segmente):
            segment = renditen[i * fenster : (i + 1) * fenster]
            abweichung = segment - segment.mean()
            kumulativ = np.cumsum(abweichung)
            spannweite = kumulativ.max() - kumulativ.min()
            std = segment.std(ddof=0)
            if std > 0:
                rs_liste.append(spannweite / std)
        if rs_liste:
            rs_werte.append(float(np.mean(rs_liste)))
            gueltige_fenster.append(int(fenster))

    if len(gueltige_fenster) < 2:
        return float("nan")
    steigung = np.polyfit(np.log(gueltige_fenster), np.log(rs_werte), 1)[0]
    return float(steigung)


def hurst_dfa(
    log_preise: np.ndarray, min_lag: int = 10, max_lag: int | None = None, anzahl_fenster: int = 20, ordnung: int = 1
) -> float:
    """Detrended Fluctuation Analysis (DFA) auf den Log-Renditen."""
    log_preise = np.asarray(log_preise, dtype=float)
    renditen = np.diff(log_preise)
    n = len(renditen)
    if n < min_lag * 2:
        return float("nan")

    profil = np.cumsum(renditen - renditen.mean())

    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max(max_lag, min_lag + 1), n // 2)
    if max_lag <= min_lag:
        return float("nan")

    fenstergroessen = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), num=anzahl_fenster).astype(int)
    )
    fenstergroessen = fenstergroessen[fenstergroessen >= min_lag]

    fluktuationen: list[float] = []
    gueltige_fenster: list[int] = []
    for fenster in fenstergroessen:
        anzahl_segmente = n // fenster
        if anzahl_segmente < 1:
            continue
        rms_liste = []
        for i in range(anzahl_segmente):
            segment = profil[i * fenster : (i + 1) * fenster]
            x = np.arange(fenster, dtype=float)
            koeffizienten = np.polyfit(x, segment, ordnung)
            trend = np.polyval(koeffizienten, x)
            rms_liste.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_liste:
            fluktuationen.append(float(np.mean(rms_liste)))
            gueltige_fenster.append(int(fenster))

    if len(gueltige_fenster) < 2:
        return float("nan")
    fluktuationen_arr = np.array(fluktuationen)
    positiv = fluktuationen_arr > 0
    if positiv.sum() < 2:
        return float("nan")
    gueltige_arr = np.array(gueltige_fenster, dtype=float)
    steigung = np.polyfit(np.log(gueltige_arr[positiv]), np.log(fluktuationen_arr[positiv]), 1)[0]
    return float(steigung)


_SCHAETZER_FUNKTIONEN = {
    "lag_std": hurst_lag_std,
    "rs": hurst_rs,
    "dfa": hurst_dfa,
}


def kombinierter_hurst(preise: np.ndarray | pd.Series, hurst_config: HurstConfig) -> float:
    """Berechnet mehrere Hurst-Schaetzer auf Log-Preisen und kombiniert sie (Median/Mittelwert)."""
    log_preise = zu_log_preisen(preise)
    werte: list[float] = []
    for name in hurst_config.schaetzer:
        func = _SCHAETZER_FUNKTIONEN[name]
        try:
            wert = func(log_preise, min_lag=hurst_config.min_lag, max_lag=hurst_config.max_lag)
        except Exception:
            continue
        if wert is not None and not np.isnan(wert):
            werte.append(wert)

    if not werte:
        return float("nan")
    if hurst_config.kombination == "median":
        return float(np.median(werte))
    return float(np.mean(werte))


def block_bootstrap_konfidenz(
    preise: np.ndarray | pd.Series,
    hurst_config: HurstConfig,
    zufallsgenerator: np.random.Generator | None = None,
) -> HurstErgebnis:
    """Schaetzt H und dessen Unsicherheit per Block-Bootstrap auf den Log-Renditen.

    Es werden Bloecke von Log-Renditen (statt einzelner Renditen) resampled, um die
    Autokorrelationsstruktur naeherungsweise zu erhalten, und daraus neue
    Preispfade rekonstruiert. Auf jedem Pfad wird H neu geschaetzt; aus der
    Verteilung ergibt sich das Konfidenzintervall.
    """
    preise = np.asarray(preise, dtype=float)
    basis_h = kombinierter_hurst(preise, hurst_config)

    log_preise = zu_log_preisen(preise)
    log_renditen = np.diff(log_preise)
    n = len(log_renditen)
    if n < 10:
        return HurstErgebnis(h=basis_h, ci_untere_grenze=float("-inf"), ci_obere_grenze=float("inf"), konfidenz=0.0)

    blockgroesse = max(1, min(hurst_config.bootstrap_blockgroesse, n))
    rng = zufallsgenerator or np.random.default_rng()
    anzahl_bloecke = int(np.ceil(n / blockgroesse))

    bootstrap_werte: list[float] = []
    for _ in range(hurst_config.bootstrap_wiederholungen):
        max_start = n - blockgroesse
        start_indizes = rng.integers(0, max_start + 1, size=anzahl_bloecke)
        bloecke = [log_renditen[i : i + blockgroesse] for i in start_indizes]
        resample_renditen = np.concatenate(bloecke)[:n]
        resample_log_preise = log_preise[0] + np.concatenate(([0.0], np.cumsum(resample_renditen)))
        resample_preise = np.exp(resample_log_preise)
        try:
            h = kombinierter_hurst(resample_preise, hurst_config)
        except Exception:
            continue
        if not np.isnan(h):
            bootstrap_werte.append(h)

    if len(bootstrap_werte) < 10:
        return HurstErgebnis(h=basis_h, ci_untere_grenze=float("-inf"), ci_obere_grenze=float("inf"), konfidenz=0.0)

    alpha = 1 - hurst_config.konfidenzniveau
    ci_unten = float(np.quantile(bootstrap_werte, alpha / 2))
    ci_oben = float(np.quantile(bootstrap_werte, 1 - alpha / 2))

    breite = max(ci_oben - ci_unten, 0.0)
    # Referenzbreite 0.5: ein Intervall so breit wie der gesamte sinnvolle H-Bereich
    # bedeutet praktisch keine Aussagekraft (Konfidenz nahe 0).
    konfidenz = float(np.clip(1 - breite / 0.5, 0.0, 1.0))

    return HurstErgebnis(h=basis_h, ci_untere_grenze=ci_unten, ci_obere_grenze=ci_oben, konfidenz=konfidenz)


def _spearman_korrelation(x: np.ndarray, y: np.ndarray) -> float:
    """Rang-Korrelation ohne scipy-Abhaengigkeit: Pearson-Korrelation der Raenge."""
    rang_x = pd.Series(x).rank().to_numpy()
    rang_y = pd.Series(y).rank().to_numpy()
    if np.std(rang_x) == 0 or np.std(rang_y) == 0:
        return 0.0
    return float(np.corrcoef(rang_x, rang_y)[0, 1])


def monotonie_score(preise: np.ndarray | pd.Series) -> dict[str, float]:
    """Misst, wie stark eine Preisreihe monoton in eine Richtung laeuft.

    Kombiniert drei Teilmasse zu einem Score in [0, 1] (hoch = stark monoton/trendend,
    was den Oszillations-Score senken soll):
      - Anteil gleichgerichteter Kerzen (Mehrheitsrichtung)
      - Betrag der Rang-Korrelation (Spearman) zwischen Zeit und Kurs
      - Verhaeltnis von Gesamtbewegung zu zurueckgelegter Pfadlaenge
    """
    preise = np.asarray(preise, dtype=float)
    if len(preise) < 2:
        return {
            "anteil_gleichgerichtet": 0.0,
            "spearman_korrelation": 0.0,
            "bewegung_pfad_verhaeltnis": 0.0,
            "monotonie_score": 0.0,
        }

    diffs = np.diff(preise)
    diffs_ungleich_null = diffs[diffs != 0]
    if len(diffs_ungleich_null) == 0:
        anteil_gleichgerichtet = 0.5
    else:
        vorzeichen = np.sign(diffs_ungleich_null)
        anteil_gleichgerichtet = float(max(np.mean(vorzeichen == 1), np.mean(vorzeichen == -1)))

    if len(set(preise)) > 1:
        zeit_index = np.arange(len(preise), dtype=float)
        spearman = abs(_spearman_korrelation(zeit_index, preise))
    else:
        spearman = 0.0

    gesamtbewegung = abs(preise[-1] - preise[0])
    pfadlaenge = float(np.sum(np.abs(diffs)))
    verhaeltnis = gesamtbewegung / pfadlaenge if pfadlaenge > 0 else 0.0

    # Anteil liegt in [0.5, 1] -> auf [0, 1] normieren, bevor er einfliesst.
    anteil_norm = max(0.0, (anteil_gleichgerichtet - 0.5) * 2)
    score = float(np.clip((anteil_norm + spearman + verhaeltnis) / 3, 0.0, 1.0))

    return {
        "anteil_gleichgerichtet": anteil_gleichgerichtet,
        "spearman_korrelation": spearman,
        "bewegung_pfad_verhaeltnis": verhaeltnis,
        "monotonie_score": score,
    }


def rollierender_hurst(df: pd.DataFrame, hurst_config: HurstConfig, fenster: int, spalte: str = "close") -> pd.Series:
    """Berechnet den kombinierten Hurst-Exponenten rollierend ueber ein Zeitfenster."""
    ergebnisse = []
    for i in range(len(df)):
        if i < fenster:
            ergebnisse.append(np.nan)
            continue
        ausschnitt = df[spalte].iloc[i - fenster : i]
        ergebnisse.append(kombinierter_hurst(ausschnitt.values, hurst_config))
    return pd.Series(ergebnisse, index=df.index)
