"""Watchlist-Scanner: Paar-Pruefung, Ranking nach Score und Kapitalverteilung.

Beim Start wird ueber `AssetPairs` geprueft, ob die Paare der Watchlist
existieren und welche Mindestordergroessen sowie Preis-/Mengenpraezision gelten.
Danach bewertet der Scanner regelmaessig alle Paare (Abschnitt 2) und rankt sie.

Wichtig fuer die Rotation (Abschnitt 4): Ein Paar, das das Gate verliert oder aus
den Top-`max_parallele_paare` herausfaellt, darf seine offenen Lots NICHT
verwaisen lassen. Es wechselt stattdessen in den Modus "nur verkaufen, keine
neuen Kaeufe" - entschieden wird das hier, ausgefuehrt in `strategy.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .broker import PaarPraezision
from .config import AppConfig
from .kraken_api import KrakenAPI, KrakenAPIError, KrakenVerbindungsFehler
from .regime import RegimeErgebnis

logger = logging.getLogger("hurstbot.scanner")


@dataclass
class PaarInfo:
    """Von Kraken bestaetigte Handelsregeln eines Paares."""

    kraken_name: str  # interner Kraken-Name (kann vom Watchlist-Namen abweichen)
    watchlist_name: str
    praezision: PaarPraezision


@dataclass
class RankingEintrag:
    """Ein Paar mit seinem aktuellen Score und der Gate-Entscheidung."""

    paar: str
    regime: RegimeErgebnis
    score: float
    gate_bestanden: bool


@dataclass
class KapitalZuteilung:
    """Ergebnis der Kapitalverteilung: wie viel EUR jedes aktive Paar bekommt."""

    aktive_paare: list[str]
    nur_verkaufen_paare: list[str]
    kapital_je_paar: dict[str, float]


def pruefe_watchlist(api: KrakenAPI, watchlist: list[str]) -> dict[str, PaarInfo]:
    """Prueft beim Start, welche Watchlist-Paare bei Kraken existieren, und liest
    deren Mindestordervolumen sowie Preis-/Mengenpraezision aus.

    Nicht existierende Paare werden mit einer Warnung uebersprungen statt den
    Start abzubrechen - so laeuft der Bot auch, wenn ein einzelnes Paar
    delistet wurde.
    """
    try:
        antwort = api.asset_pairs(watchlist)
    except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
        raise RuntimeError(f"Konnte Handelsregeln (AssetPairs) nicht laden: {exc}") from exc

    # Kraken antwortet unter seinem internen Namen, der vom angefragten abweichen kann
    # (z. B. "XDGEUR" -> "XDGEUR", aber "BTCEUR" -> "XXBTZEUR"). Wir ordnen ueber
    # das Feld "altname" und den Schluessel selbst zu.
    infos: dict[str, PaarInfo] = {}
    for kraken_name, regeln in antwort.items():
        altname = regeln.get("altname", kraken_name)
        passender_wunsch = next(
            (w for w in watchlist if w.upper() in (kraken_name.upper(), altname.upper())), None
        )
        if passender_wunsch is None:
            continue
        praezision = PaarPraezision(
            preis_dezimalstellen=int(regeln.get("pair_decimals", 2)),
            mengen_dezimalstellen=int(regeln.get("lot_decimals", 8)),
            mindest_ordervolumen=float(regeln.get("ordermin", 0.0)),
        )
        infos[passender_wunsch] = PaarInfo(
            kraken_name=kraken_name, watchlist_name=passender_wunsch, praezision=praezision
        )

    fehlende = [w for w in watchlist if w not in infos]
    for paar in fehlende:
        logger.warning("Paar %s existiert bei Kraken nicht (oder wurde delistet) - wird uebersprungen", paar)
    if not infos:
        raise RuntimeError("Kein einziges Paar der Watchlist existiert bei Kraken - bitte Config pruefen")

    logger.info("Handelsregeln fuer %s Paare geladen: %s", len(infos), ", ".join(sorted(infos)))
    return infos


def ranke_paare(bewertungen: dict[str, RegimeErgebnis]) -> list[RankingEintrag]:
    """Sortiert alle bewerteten Paare nach Oszillations-Score (hoechster zuerst).

    Paare, die das Gate nicht bestehen, bleiben im Ranking enthalten (damit der
    Scanner sie protokollieren kann), werden aber bei der Kapitalverteilung
    uebergangen.
    """
    eintraege = [
        RankingEintrag(paar=paar, regime=ergebnis, score=ergebnis.oszillations_score, gate_bestanden=ergebnis.handel_erlaubt)
        for paar, ergebnis in bewertungen.items()
    ]
    return sorted(eintraege, key=lambda e: (e.gate_bestanden, e.score), reverse=True)


def verteile_kapital(
    ranking: list[RankingEintrag],
    config: AppConfig,
    verfuegbares_kapital_eur: float,
    paare_mit_offenen_lots: set[str],
) -> KapitalZuteilung:
    """Waehlt die aktiven Paare und verteilt das Kapital nach Score.

    Aktiv werden nur die hoechstbewerteten Paare, die das Gate bestehen, maximal
    `max_parallele_paare`. Jedes andere Paar, das noch offene Lots haelt, landet
    in `nur_verkaufen_paare` - so verwaist bei einer Rotation kein Bestand.
    """
    kandidaten = [e for e in ranking if e.gate_bestanden and e.score > 0]
    aktive = kandidaten[: config.scanner.max_parallele_paare]
    aktive_namen = [e.paar for e in aktive]

    nur_verkaufen = sorted(
        paar for paar in paare_mit_offenen_lots if paar not in aktive_namen
    )

    score_summe = sum(e.score for e in aktive)
    kapital_je_paar: dict[str, float] = {}
    if aktive and score_summe > 0:
        for eintrag in aktive:
            kapital_je_paar[eintrag.paar] = verfuegbares_kapital_eur * (eintrag.score / score_summe)
    elif aktive:
        # Alle Scores exakt 0 (sollte durch den Filter oben nicht vorkommen) -> gleichmaessig teilen.
        gleichanteil = verfuegbares_kapital_eur / len(aktive)
        kapital_je_paar = {e.paar: gleichanteil for e in aktive}

    return KapitalZuteilung(
        aktive_paare=aktive_namen, nur_verkaufen_paare=nur_verkaufen, kapital_je_paar=kapital_je_paar
    )


def formatiere_ranking(ranking: list[RankingEintrag]) -> str:
    """Erstellt eine lesbare Tabelle des aktuellen Rankings fuer die Konsole."""
    if not ranking:
        return "Kein Paar konnte bewertet werden."

    zeilen = [
        f"{'Paar':<12} {'Score':>7} {'H kurz':>8} {'H mittel':>9} {'Konfid.':>8} {'Gate':>6}  Grund",
        "-" * 100,
    ]
    for eintrag in ranking:
        h_kurz = f"{eintrag.regime.kurz.hurst.h:.3f}" if eintrag.regime.kurz else "n/a"
        h_mittel = f"{eintrag.regime.mittel.hurst.h:.3f}" if eintrag.regime.mittel else "n/a"
        konfidenz = (
            f"{min(eintrag.regime.kurz.hurst.konfidenz, eintrag.regime.mittel.hurst.konfidenz):.2f}"
            if eintrag.regime.kurz and eintrag.regime.mittel
            else "n/a"
        )
        gate = "JA" if eintrag.gate_bestanden else "NEIN"
        zeilen.append(
            f"{eintrag.paar:<12} {eintrag.score:>7.3f} {h_kurz:>8} {h_mittel:>9} {konfidenz:>8} {gate:>6}  {eintrag.regime.grund}"
        )
    return "\n".join(zeilen)
