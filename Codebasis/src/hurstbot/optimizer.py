"""X-Optimierung per Walk-Forward-Backtest (Grid-Search + Verfeinerung) und
paarspezifische Zeitskalenauswahl.

Die Zielfunktion maximiert den Netto-Gewinn nach Gebuehren, bestraft aber hohe
Drawdowns und lange Haltedauern offener Lots. Um Overfitting zu vermeiden, wird
der Datensatz in mehrere rollierende Fenster aufgeteilt; pro Fenster wird X auf
dem Trainingsteil gesucht und auf dem anschliessenden Testteil (Out-of-Sample)
bewertet. Der finale X-Wert ist der Median der pro Fenster gewaehlten Werte,
zusaetzlich mit dem vorherigen X-Wert exponentiell geglaettet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, simulate
from .config import GebuehrenConfig, XOptimizerConfig, ZeitskalaProfil

logger = logging.getLogger("hurstbot.optimizer")


def gebuehren_breakeven_x_prozent(gebuehren_config: GebuehrenConfig) -> float:
    """Untere Grenze fuer X: 2x Gebuehr (Kauf+Verkauf) + Spread + Sicherheitspuffer.

    Liegt X darunter, kann ein Round-Trip rein rechnerisch nie netto profitabel sein.
    """
    return (
        2 * gebuehren_config.taker_gebuehr_prozent
        + gebuehren_config.geschaetzter_spread_prozent
        + gebuehren_config.sicherheitspuffer_prozent
    )


def zielfunktion(ergebnis: BacktestResult, x_config: XOptimizerConfig) -> float:
    """Netto-Gewinn, bestraft fuer Drawdown und fuer lange Haltedauer offener Lots."""
    strafe_drawdown = x_config.max_drawdown_strafe * ergebnis.max_drawdown_prozent
    strafe_haltedauer = (
        x_config.haltedauer_strafe_pro_stunde
        * ergebnis.offene_lots_mittlere_dauer_stunden
        * ergebnis.offene_lots_anzahl
    )
    return ergebnis.netto_gewinn_eur - strafe_drawdown - strafe_haltedauer


def _optuna_verfeinerung(
    train_df: pd.DataFrame,
    x_min: float,
    x_max: float,
    x_config: XOptimizerConfig,
    gebuehren_config: GebuehrenConfig,
    y_eur: float,
    start_kapital: float,
) -> float | None:
    """Optionale Bayes'sche Verfeinerung mit optuna, falls installiert."""
    try:
        import optuna
    except ImportError:
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial: "optuna.Trial") -> float:
        x = trial.suggest_float("x_prozent", x_min, x_max)
        ergebnis = simulate(
            train_df, x, y_eur, gebuehren_config.taker_gebuehr_prozent, gebuehren_config.geschaetzter_spread_prozent, start_kapital
        )
        return zielfunktion(ergebnis, x_config)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=x_config.optuna_versuche, show_progress_bar=False)
    return float(study.best_params["x_prozent"])


@dataclass
class FensterErgebnis:
    """Ergebnis eines einzelnen Walk-Forward-Fensters."""

    fenster_index: int
    train_zeilen: int
    test_zeilen: int
    bestes_x_train: float
    test_ergebnis: BacktestResult
    test_score: float


@dataclass
class OptimierungsErgebnis:
    """Endergebnis der Walk-Forward-X-Optimierung ueber alle Fenster."""

    x_prozent: float
    fenster_ergebnisse: list[FensterErgebnis]
    mittlerer_out_of_sample_score: float
    mittlere_roundtrips: float
    mittlere_amplitude_prozent: float
    x_min_erlaubt: float
    hat_belastbare_evidenz: bool = True


def walk_forward_optimize(
    df: pd.DataFrame,
    x_config: XOptimizerConfig,
    gebuehren_config: GebuehrenConfig,
    y_eur: float,
    start_kapital: float,
    vorheriger_x: float | None = None,
    preis_spalte: str = "close",
    zeit_spalte: str = "zeit",
) -> OptimierungsErgebnis:
    """Fuehrt die Walk-Forward-Optimierung von X ueber mehrere rollierende Fenster aus."""
    x_min_erlaubt = max(x_config.x_grid_min_prozent, gebuehren_breakeven_x_prozent(gebuehren_config))
    x_max = max(x_config.x_grid_max_prozent, x_min_erlaubt + 0.01)
    grobes_grid = [x for x in np.linspace(x_min_erlaubt, x_max, x_config.x_grid_schritte) if x > 0]

    n = len(df)
    fenster_anzahl = max(1, x_config.walk_forward_fenster_anzahl)
    fenstergroesse = n // fenster_anzahl
    fenster_ergebnisse: list[FensterErgebnis] = []

    if fenstergroesse < 20:
        raise ValueError(
            f"Zu wenig Kerzen ({n}) fuer {fenster_anzahl} Walk-Forward-Fenster. "
            "Mehr Historie laden oder Fensteranzahl reduzieren."
        )

    for i in range(fenster_anzahl):
        start = i * fenstergroesse
        ende = n if i == fenster_anzahl - 1 else (i + 1) * fenstergroesse
        fenster_df = df.iloc[start:ende].reset_index(drop=True)

        split_idx = max(int(len(fenster_df) * (1 - x_config.walk_forward_test_anteil)), 1)
        train_df = fenster_df.iloc[:split_idx]
        test_df = fenster_df.iloc[split_idx:]
        if len(train_df) < 10 or len(test_df) < 5:
            continue

        train_scores: dict[float, float] = {}
        for x in grobes_grid:
            ergebnis = simulate(
                train_df, x, y_eur, gebuehren_config.taker_gebuehr_prozent, gebuehren_config.geschaetzter_spread_prozent,
                start_kapital, preis_spalte, zeit_spalte,
            )
            train_scores[x] = zielfunktion(ergebnis, x_config)

        bestes_x_grob = max(train_scores, key=train_scores.get)

        schrittweite_grob = (x_max - x_min_erlaubt) / max(x_config.x_grid_schritte, 1)
        feines_grid = np.linspace(
            max(bestes_x_grob - schrittweite_grob, x_min_erlaubt), bestes_x_grob + schrittweite_grob, x_config.verfeinerung_schritte
        )
        for x in feines_grid:
            x = float(x)
            if x in train_scores:
                continue
            ergebnis = simulate(
                train_df, x, y_eur, gebuehren_config.taker_gebuehr_prozent, gebuehren_config.geschaetzter_spread_prozent,
                start_kapital, preis_spalte, zeit_spalte,
            )
            train_scores[x] = zielfunktion(ergebnis, x_config)

        bestes_x_train = max(train_scores, key=train_scores.get)

        if x_config.mit_optuna:
            optuna_x = _optuna_verfeinerung(train_df, x_min_erlaubt, x_max, x_config, gebuehren_config, y_eur, start_kapital)
            if optuna_x is not None:
                optuna_ergebnis = simulate(
                    train_df, optuna_x, y_eur, gebuehren_config.taker_gebuehr_prozent,
                    gebuehren_config.geschaetzter_spread_prozent, start_kapital, preis_spalte, zeit_spalte,
                )
                if zielfunktion(optuna_ergebnis, x_config) > train_scores[bestes_x_train]:
                    bestes_x_train = optuna_x

        test_ergebnis = simulate(
            test_df, bestes_x_train, y_eur, gebuehren_config.taker_gebuehr_prozent,
            gebuehren_config.geschaetzter_spread_prozent, start_kapital, preis_spalte, zeit_spalte,
        )
        test_score = zielfunktion(test_ergebnis, x_config)

        fenster_ergebnisse.append(
            FensterErgebnis(
                fenster_index=i,
                train_zeilen=len(train_df),
                test_zeilen=len(test_df),
                bestes_x_train=bestes_x_train,
                test_ergebnis=test_ergebnis,
                test_score=test_score,
            )
        )

    if not fenster_ergebnisse:
        raise ValueError("Keines der Walk-Forward-Fenster hatte genug Daten fuer Training und Test")

    kandidaten_x = [f.bestes_x_train for f in fenster_ergebnisse]
    endgueltiges_x = float(np.median(kandidaten_x))

    if vorheriger_x is not None:
        endgueltiges_x = vorheriger_x + x_config.glaettung_alpha * (endgueltiges_x - vorheriger_x)
    endgueltiges_x = max(endgueltiges_x, x_min_erlaubt)

    scores = [f.test_score for f in fenster_ergebnisse]
    roundtrips = [f.test_ergebnis.anzahl_roundtrips for f in fenster_ergebnisse]
    amplituden = [f.test_ergebnis.mittlere_amplitude_prozent for f in fenster_ergebnisse]

    mittlerer_score = float(np.mean(scores))

    # X gilt nur als belegt, wenn es out-of-sample ueberhaupt gehandelt hat UND dabei
    # im Mittel Gewinn erzielte. Ohne Round-Trip war die Zielfunktion in jedem
    # Testfenster 0 (der Median der Trainingswerte also beliebig); bei negativem
    # Score ist das beste X nur das am wenigsten schlechte. Beides darf der Aufrufer
    # nicht als validiertes Ergebnis behandeln - dasselbe Kriterium gilt in
    # `waehle_zeitskala_profil`.
    ohne_roundtrip = sum(roundtrips) == 0
    nicht_profitabel = mittlerer_score <= 0
    hat_belastbare_evidenz = not (ohne_roundtrip or nicht_profitabel)

    if ohne_roundtrip:
        logger.warning(
            "Walk-Forward-Optimierung ohne einen einzigen Out-of-Sample-Round-Trip "
            "(%s Fenster, je %s Testkerzen): X=%.4f %% ist statistisch NICHT belegt. "
            "Mehr Historie sammeln oder walk_forward_fenster_anzahl verkleinern.",
            len(fenster_ergebnisse), fenster_ergebnisse[0].test_zeilen, endgueltiges_x,
        )
    elif nicht_profitabel:
        logger.warning(
            "Walk-Forward-Optimierung mit negativem Out-of-Sample-Ergebnis "
            "(%.4f EUR im Mittel): X=%.4f %% waere auf diesen Daten verlustbringend "
            "und wird NICHT als belegt gewertet.",
            mittlerer_score, endgueltiges_x,
        )

    return OptimierungsErgebnis(
        x_prozent=endgueltiges_x,
        fenster_ergebnisse=fenster_ergebnisse,
        mittlerer_out_of_sample_score=mittlerer_score,
        mittlere_roundtrips=float(np.mean(roundtrips)),
        mittlere_amplitude_prozent=float(np.mean(amplituden)),
        x_min_erlaubt=x_min_erlaubt,
        hat_belastbare_evidenz=hat_belastbare_evidenz,
    )


def fenster_ergebnisse_als_dataframe(ergebnis: OptimierungsErgebnis) -> pd.DataFrame:
    """Wandelt die Walk-Forward-Fenster in ein DataFrame fuer den CSV-Bericht um."""
    zeilen = []
    for f in ergebnis.fenster_ergebnisse:
        zeilen.append(
            {
                "fenster": f.fenster_index,
                "train_zeilen": f.train_zeilen,
                "test_zeilen": f.test_zeilen,
                "bestes_x_prozent": round(f.bestes_x_train, 4),
                "test_netto_gewinn_eur": round(f.test_ergebnis.netto_gewinn_eur, 4),
                "test_max_drawdown_prozent": round(f.test_ergebnis.max_drawdown_prozent, 4),
                "test_roundtrips": f.test_ergebnis.anzahl_roundtrips,
                "test_mittlere_amplitude_prozent": round(f.test_ergebnis.mittlere_amplitude_prozent, 4),
                "test_score": round(f.test_score, 4),
            }
        )
    return pd.DataFrame(zeilen)


def textbericht(ergebnis: OptimierungsErgebnis, paar: str) -> str:
    """Erstellt eine menschenlesbare Textzusammenfassung der Optimierung."""
    zeilen = [
        f"X-Optimierung fuer {paar}",
        "=" * 40,
        f"Gewaehltes X (nach Glaettung):      {ergebnis.x_prozent:.4f} %",
        f"Gebuehren-Breakeven (untere Grenze): {ergebnis.x_min_erlaubt:.4f} %",
        f"Mittlerer Out-of-Sample-Score:       {ergebnis.mittlerer_out_of_sample_score:.4f} EUR",
        f"Mittlere Round-Trips je Fenster:     {ergebnis.mittlere_roundtrips:.2f}",
        f"Mittlere Amplitude je Round-Trip:    {ergebnis.mittlere_amplitude_prozent:.4f} %",
        f"Anzahl ausgewerteter Fenster:         {len(ergebnis.fenster_ergebnisse)}",
    ]
    if not ergebnis.hat_belastbare_evidenz:
        zeilen.append("")
        if ergebnis.mittlere_roundtrips == 0:
            zeilen += [
                "!! WARNUNG: In keinem Testfenster kam ein Round-Trip zustande.",
                "!! Dieses X ist damit NICHT out-of-sample validiert. Ursache ist meist zu",
                "!! kurze Historie: Kraken liefert max. 720 Kerzen pro Abfrage, und das",
                f"!! Testfenster betraegt hier nur {ergebnis.fenster_ergebnisse[0].test_zeilen} Kerzen.",
                "!! Abhilfe: Bot laenger laufen lassen (der OHLC-Cache waechst mit), ein",
                "!! groesseres Kerzenintervall waehlen oder walk_forward_fenster_anzahl senken.",
            ]
        else:
            zeilen += [
                "!! WARNUNG: Das Out-of-Sample-Ergebnis ist im Mittel NEGATIV",
                f"!! ({ergebnis.mittlerer_out_of_sample_score:+.4f} EUR). Dieses X war auf den Testdaten",
                "!! verlustbringend - es ist nur das am wenigsten schlechte aus dem Suchbereich.",
                "!! Der Bot uebernimmt es deshalb NICHT und behaelt sein bisheriges X.",
                "!! Das ist ein Hinweis darauf, dass dieses Paar in diesem Zeitraum nicht",
                "!! oszilliert hat - die Strategie passt hier derzeit nicht.",
            ]
        zeilen.append("!! Der Bot handelt mit einem solchen X nicht.")
    zeilen += [
        "",
        "Pro Fenster gewaehltes X (Training) vs. Out-of-Sample-Ergebnis:",
    ]
    for f in ergebnis.fenster_ergebnisse:
        zeilen.append(
            f"  Fenster {f.fenster_index}: X={f.bestes_x_train:.4f} % | "
            f"Netto {f.test_ergebnis.netto_gewinn_eur:+.2f} EUR | "
            f"Drawdown {f.test_ergebnis.max_drawdown_prozent:.2f} % | "
            f"Round-Trips {f.test_ergebnis.anzahl_roundtrips}"
        )
    return "\n".join(zeilen)


@dataclass
class ZeitskalaKandidatErgebnis:
    """Walk-Forward-Ergebnis eines Zeitskala-Kandidaten (fuer Abschnitt 2.5)."""

    profil: ZeitskalaProfil
    optimierung: OptimierungsErgebnis | None
    erfuellt_mindestkriterium: bool
    fehler: str | None = None


def waehle_zeitskala_profil(
    kandidaten_daten: dict[str, pd.DataFrame],
    kandidaten: list[ZeitskalaProfil],
    x_config: XOptimizerConfig,
    gebuehren_config: GebuehrenConfig,
    y_eur: float,
    start_kapital: float,
    mindest_roundtrips: int,
    fallback_profil: ZeitskalaProfil,
) -> tuple[ZeitskalaProfil, str, list[ZeitskalaKandidatErgebnis]]:
    """Waehlt das beste Zeitskala-Profil anhand des Out-of-Sample-Nettogewinns.

    WICHTIG: Die Auswahl erfolgt NICHT nach dem kleinsten Hurst-Exponenten
    (Selection Bias), sondern ausschliesslich nach validiertem Out-of-Sample-
    Nettogewinn mit Mindestanzahl Round-Trips. `kandidaten_daten` muss fuer
    jeden Kandidatennamen die passende OHLC-Reihe enthalten.
    """
    ergebnisse: list[ZeitskalaKandidatErgebnis] = []
    for profil in kandidaten:
        df = kandidaten_daten.get(profil.name)
        if df is None or len(df) < 50:
            ergebnisse.append(
                ZeitskalaKandidatErgebnis(profil=profil, optimierung=None, erfuellt_mindestkriterium=False, fehler="zu wenig Daten")
            )
            continue
        try:
            optimierung = walk_forward_optimize(df, x_config, gebuehren_config, y_eur, start_kapital)
        except ValueError as exc:
            ergebnisse.append(
                ZeitskalaKandidatErgebnis(profil=profil, optimierung=None, erfuellt_mindestkriterium=False, fehler=str(exc))
            )
            continue
        erfuellt = (
            optimierung.mittlere_roundtrips >= mindest_roundtrips and optimierung.mittlerer_out_of_sample_score > 0
        )
        ergebnisse.append(
            ZeitskalaKandidatErgebnis(profil=profil, optimierung=optimierung, erfuellt_mindestkriterium=erfuellt)
        )

    gueltige = [e for e in ergebnisse if e.erfuellt_mindestkriterium and e.optimierung is not None]
    if not gueltige:
        return fallback_profil, "Kein Kandidat erfuellt Mindestkriterien (Round-Trips/Nettogewinn) -> Fallback", ergebnisse

    bestes = max(gueltige, key=lambda e: e.optimierung.mittlerer_out_of_sample_score)
    grund = (
        f"Bestes Out-of-Sample-Nettogewinn-Ergebnis unter Kandidaten mit >= {mindest_roundtrips} Round-Trips "
        f"(Score={bestes.optimierung.mittlerer_out_of_sample_score:.2f} EUR)"
    )
    return bestes.profil, grund, ergebnisse
