"""Broker-Schnittstelle mit drei Sicherheitsstufen: Paper (Default), Validate, Live.

Sicherheit hat Vorrang (Abschnitt 5): `KrakenLiveBroker` verweigert den Betrieb,
wenn nicht ALLE DREI Freigaben vorliegen:
  1. die Programmoption `--live` wurde gesetzt,
  2. die Umgebungsvariable `KRAKEN_LIVE_CONFIRM=JA` ist gesetzt,
  3. der Nutzer hat beim Start eine explizite Bestaetigung eingegeben.
Diese Pruefung sitzt direkt im Konstruktor (nicht nur in der CLI), damit es
keinen Codepfad gibt, der sie umgeht.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import GebuehrenConfig
from .kraken_api import KrakenAPI, KrakenAPIError, KrakenVerbindungsFehler

logger = logging.getLogger("hurstbot.broker")


class LiveModusNichtErlaubt(Exception):
    """Wird geworfen, wenn eine der drei Live-Freigaben fehlt."""


@dataclass
class PaarPraezision:
    """Handelsregeln eines Paares aus Krakens `AssetPairs` (Abschnitt 4/5)."""

    preis_dezimalstellen: int
    mengen_dezimalstellen: int
    mindest_ordervolumen: float  # in Basis-Einheiten (z. B. BTC), Kraken-Feld "ordermin"


@dataclass
class Ausfuehrung:
    """Ergebnis eines Order-Versuchs."""

    erfolgreich: bool
    fill_preis: float
    menge: float
    gebuehr_eur: float
    order_id: str | None
    grund: str


def runde_praezision(wert: float, dezimalstellen: int) -> float:
    faktor = 10**dezimalstellen
    return int(wert * faktor) / faktor  # immer abrunden, nie "nach oben schummeln"


class Broker(ABC):
    """Gemeinsame Schnittstelle fuer Paper-, Validate- und Live-Ausfuehrung."""

    @abstractmethod
    def kaufe(self, paar: str, menge_quote_eur: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        """Kauft fuer `menge_quote_eur` EUR zum aktuellen Referenzpreis (Limit nahe Markt)."""

    @abstractmethod
    def verkaufe(self, paar: str, menge_basis: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        """Verkauft `menge_basis` Einheiten zum aktuellen Referenzpreis (Limit nahe Markt)."""


class PaperBroker(Broker):
    """Virtuelle Ausfuehrung mit realistischem Gebuehren- und Spread-Modell.

    Das Fill-/Gebuehrenmodell entspricht bewusst dem der historischen Simulation
    (`backtest.LotBuch`), damit Paper-Betrieb und Backtest vergleichbare Resultate
    liefern - hier aber pro Einzelorder statt fuer einen ganzen Lauf.
    """

    def __init__(self, gebuehren_config: GebuehrenConfig) -> None:
        self._fee = gebuehren_config.taker_gebuehr_prozent / 100
        self._halber_spread = gebuehren_config.geschaetzter_spread_prozent / 100 / 2

    def kaufe(self, paar: str, menge_quote_eur: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        fill_preis = referenzpreis * (1 + self._halber_spread)
        menge = runde_praezision(menge_quote_eur / fill_preis, praezision.mengen_dezimalstellen)
        if menge < praezision.mindest_ordervolumen:
            return Ausfuehrung(False, fill_preis, 0.0, 0.0, None, "unter Kraken-Mindestordervolumen")
        gebuehr = menge * fill_preis * self._fee
        return Ausfuehrung(True, fill_preis, menge, gebuehr, f"PAPER-{uuid.uuid4().hex[:12]}", "Paper-Kauf ausgefuehrt")

    def verkaufe(self, paar: str, menge_basis: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        fill_preis = referenzpreis * (1 - self._halber_spread)
        menge = runde_praezision(menge_basis, praezision.mengen_dezimalstellen)
        if menge < praezision.mindest_ordervolumen:
            return Ausfuehrung(False, fill_preis, 0.0, 0.0, None, "unter Kraken-Mindestordervolumen")
        gebuehr = menge * fill_preis * self._fee
        return Ausfuehrung(True, fill_preis, menge, gebuehr, f"PAPER-{uuid.uuid4().hex[:12]}", "Paper-Verkauf ausgefuehrt")


class KrakenValidateBroker(Broker):
    """Sendet Orders an die private Kraken-API mit `validate=true`.

    Es wird NICHTS ausgefuehrt - Kraken prueft nur, ob die Order so angenommen
    wuerde (Format, Mindestgroesse, Guthaben). Nuetzlich als letzter Test vor Live.
    """

    def __init__(self, api: KrakenAPI, gebuehren_config: GebuehrenConfig) -> None:
        self._api = api
        self._fee = gebuehren_config.taker_gebuehr_prozent / 100
        self._halber_spread = gebuehren_config.geschaetzter_spread_prozent / 100 / 2

    def _validiere_order(
        self, paar: str, typ: str, referenzpreis: float, menge: float, praezision: PaarPraezision
    ) -> Ausfuehrung:
        limit_preis = round(referenzpreis, praezision.preis_dezimalstellen)
        menge = runde_praezision(menge, praezision.mengen_dezimalstellen)
        if menge < praezision.mindest_ordervolumen:
            return Ausfuehrung(False, limit_preis, 0.0, 0.0, None, "unter Kraken-Mindestordervolumen")
        try:
            self._api.add_order(paar, typ, "limit", menge, preis=limit_preis, validate=True)
        except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
            return Ausfuehrung(False, limit_preis, 0.0, 0.0, None, f"Validierung fehlgeschlagen: {exc}")
        gebuehr_schaetzung = menge * limit_preis * self._fee
        return Ausfuehrung(True, limit_preis, menge, gebuehr_schaetzung, None, "Von Kraken validiert (nicht ausgefuehrt)")

    def kaufe(self, paar: str, menge_quote_eur: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        limit_preis = referenzpreis * (1 + self._halber_spread)
        menge = menge_quote_eur / limit_preis
        return self._validiere_order(paar, "buy", limit_preis, menge, praezision)

    def verkaufe(self, paar: str, menge_basis: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        limit_preis = referenzpreis * (1 - self._halber_spread)
        return self._validiere_order(paar, "sell", limit_preis, menge_basis, praezision)


class KrakenLiveBroker(Broker):
    """Echte Orders. Nur aktiv, wenn ALLE drei Freigaben aus Abschnitt 5 vorliegen."""

    ERWARTETE_BESTAETIGUNG = "JA"

    def __init__(
        self,
        api: KrakenAPI,
        gebuehren_config: GebuehrenConfig,
        live_flag_gesetzt: bool,
        nutzer_bestaetigung: str,
        order_zeitlimit_sekunden: int = 60,
    ) -> None:
        self._pruefe_freigaben(live_flag_gesetzt, nutzer_bestaetigung)
        self._api = api
        self._fee = gebuehren_config.taker_gebuehr_prozent / 100
        self._halber_spread = gebuehren_config.geschaetzter_spread_prozent / 100 / 2
        self._order_zeitlimit = order_zeitlimit_sekunden
        logger.warning("KrakenLiveBroker aktiviert - es werden ECHTE Orders gesendet!")

    @classmethod
    def _pruefe_freigaben(cls, live_flag_gesetzt: bool, nutzer_bestaetigung: str) -> None:
        fehlende = []
        if not live_flag_gesetzt:
            fehlende.append("Programmoption --live fehlt")
        if os.environ.get("KRAKEN_LIVE_CONFIRM") != "JA":
            fehlende.append("Umgebungsvariable KRAKEN_LIVE_CONFIRM=JA fehlt")
        if (nutzer_bestaetigung or "").strip().upper() != cls.ERWARTETE_BESTAETIGUNG:
            fehlende.append("interaktive Nutzerbestaetigung fehlt oder falsch")
        if fehlende:
            raise LiveModusNichtErlaubt(
                "Live-Modus verweigert, es fehlen folgende Freigaben: " + "; ".join(fehlende)
            )

    def _sende_order_und_warte(
        self, paar: str, typ: str, limit_preis: float, menge: float, praezision: PaarPraezision
    ) -> Ausfuehrung:
        limit_preis = round(limit_preis, praezision.preis_dezimalstellen)
        menge = runde_praezision(menge, praezision.mengen_dezimalstellen)
        if menge < praezision.mindest_ordervolumen:
            return Ausfuehrung(False, limit_preis, 0.0, 0.0, None, "unter Kraken-Mindestordervolumen")

        zeitlimit = f"+{self._order_zeitlimit}"
        try:
            antwort = self._api.add_order(paar, typ, "limit", menge, preis=limit_preis, validate=False, zeitlimit=zeitlimit)
        except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
            logger.error("Order fehlgeschlagen fuer %s (%s): %s", paar, typ, exc)
            return Ausfuehrung(False, limit_preis, 0.0, 0.0, None, f"Order fehlgeschlagen: {exc}")

        order_ids = antwort.get("txid", [])
        order_id = order_ids[0] if order_ids else None

        erfuellt = self._warte_auf_ausfuehrung(order_id)
        if erfuellt:
            gebuehr = menge * limit_preis * self._fee
            return Ausfuehrung(True, limit_preis, menge, gebuehr, order_id, "Live-Order ausgefuehrt")

        if order_id:
            try:
                self._api.cancel_order(order_id)
            except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
                logger.error("Konnte offene Order %s nicht stornieren: %s", order_id, exc)
        return Ausfuehrung(False, limit_preis, 0.0, 0.0, order_id, "Order innerhalb des Zeitlimits nicht ausgefuehrt, storniert")

    def _warte_auf_ausfuehrung(self, order_id: str | None) -> bool:
        if order_id is None:
            return False
        start = time.monotonic()
        while time.monotonic() - start < self._order_zeitlimit:
            try:
                offene_orders = self._api.open_orders()
            except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
                logger.warning("Konnte offene Orders nicht abfragen: %s", exc)
                time.sleep(2)
                continue
            if order_id not in offene_orders.get("open", {}):
                return True  # nicht mehr offen -> ausgefuehrt (oder extern storniert)
            time.sleep(2)
        return False

    def kaufe(self, paar: str, menge_quote_eur: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        limit_preis = referenzpreis * (1 + self._halber_spread)
        menge = menge_quote_eur / limit_preis
        return self._sende_order_und_warte(paar, "buy", limit_preis, menge, praezision)

    def verkaufe(self, paar: str, menge_basis: float, referenzpreis: float, praezision: PaarPraezision) -> Ausfuehrung:
        limit_preis = referenzpreis * (1 - self._halber_spread)
        return self._sende_order_und_warte(paar, "sell", limit_preis, menge_basis, praezision)
