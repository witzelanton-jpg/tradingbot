"""Grid-Logik pro Paar: fuehrt Lots, Anker, Zustandsautomat und Waechter zusammen.

`PaarStrategie` ist die einzige Stelle, die tatsaechlich Kauf-/Verkaufsentscheidungen
trifft und den `Broker` aufruft. Die Lot-Buchfuehrung nutzt dieselben `Lot`/`Trade`-
Datenklassen wie `backtest.py`, damit Paper-/Live-Betrieb und Backtest konsistent
bleiben - die Fill-/Gebuehrenmechanik selbst liegt aber beim `Broker`.

Zustandsautomat je Paar (Abschnitt 3.6):
  HANDELT       - Gate bestanden, Kaeufe und Verkaeufe erlaubt
  NUR_VERKAUFEN - Gate verloren (z. B. H zu hoch) ODER Exposure-Limit erreicht;
                  bestehende Lots duerfen weiter verkauft werden, keine neuen Kaeufe
  PAUSIERT      - Ausbruchswaechter hat ausgeloest; GAR KEINE neuen Orders (auch
                  keine Verkaeufe), bis der Wiedereinstieg erlaubt ist
  GESTOPPT      - globaler Kill-Switch; wird von aussen gesetzt, keine Orders mehr
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import Lot, Trade
from .broker import Broker, PaarPraezision
from .config import AppConfig, ZeitskalaProfil
from .guards import (
    PaarZustand,
    WiedereinstiegZustand,
    ist_kauf_durch_exposure_blockiert,
    pruefe_ausbruch,
    pruefe_wiedereinstieg,
)
from .logging_setup import protokolliere_trade, protokolliere_zustandswechsel
from .regime import RegimeErgebnis, berechne_y_betrag

logger = logging.getLogger("hurstbot.strategy")


@dataclass
class PaarStrategie:
    """Haelt den kompletten Live-/Paper-Zustand eines einzelnen Handelspaares."""

    paar: str
    config: AppConfig
    broker: Broker
    praezision: PaarPraezision
    zustand: PaarZustand = PaarZustand.HANDELT
    lots: list[Lot] = field(default_factory=list)
    kapital_eur: float = 0.0
    anker_preis: float | None = None
    gate_handelt: bool = False
    x_prozent: float = 1.0
    zeitskala_profil: ZeitskalaProfil | None = None
    wiedereinstieg_zustand: WiedereinstiegZustand | None = None
    trade_log_pfad: Path | None = None
    zustand_log_pfad: Path | None = None

    def __post_init__(self) -> None:
        if self.zeitskala_profil is None:
            kandidaten = self.config.zeitskala_auswahl.kandidaten
            fallback_name = self.config.zeitskala_auswahl.fallback_profil
            self.zeitskala_profil = next(
                (p for p in kandidaten if p.name == fallback_name), kandidaten[0]
            )

    def investierter_wert(self, aktueller_preis: float) -> float:
        return sum(lot.menge for lot in self.lots) * aktueller_preis

    def gesamtwert(self, aktueller_preis: float) -> float:
        return self.kapital_eur + self.investierter_wert(aktueller_preis)

    def _wechsel_zustand(self, neuer_zustand: PaarZustand, grund: str, zeit: pd.Timestamp) -> None:
        if neuer_zustand == self.zustand:
            return
        if self.zustand_log_pfad:
            protokolliere_zustandswechsel(self.zustand_log_pfad, self.paar, self.zustand.value, neuer_zustand.value, grund, zeit)
        else:
            logger.warning("Paar %s: %s -> %s (%s)", self.paar, self.zustand.value, neuer_zustand.value, grund)
        self.zustand = neuer_zustand

    def _protokolliere(self, trade: Trade) -> None:
        if self.trade_log_pfad:
            protokolliere_trade(self.trade_log_pfad, self.paar, trade)

    def _verkaufe_faellige_lots(self, preis: float, zeit: pd.Timestamp) -> None:
        gebuehr = self.config.gebuehren.taker_gebuehr_prozent / 100
        halber_spread = self.config.gebuehren.geschaetzter_spread_prozent / 100 / 2
        geschaetzter_netto_pro_einheit = preis * (1 - halber_spread) * (1 - gebuehr)

        verbleibende_lots: list[Lot] = []
        for lot in self.lots:
            trigger_preis = lot.einstandspreis * (1 + self.x_prozent / 100)
            ist_faellig = preis >= trigger_preis and geschaetzter_netto_pro_einheit > lot.einstandspreis
            if not ist_faellig:
                verbleibende_lots.append(lot)
                continue

            ausfuehrung = self.broker.verkaufe(self.paar, lot.menge, preis, self.praezision)
            if not ausfuehrung.erfolgreich:
                logger.info("Verkauf fuer %s nicht ausgefuehrt: %s", self.paar, ausfuehrung.grund)
                verbleibende_lots.append(lot)
                continue

            erloes = ausfuehrung.menge * ausfuehrung.fill_preis - ausfuehrung.gebuehr_eur
            self.kapital_eur += erloes
            trade = Trade(
                zeit=zeit, art="VERKAUF", preis=ausfuehrung.fill_preis, menge=ausfuehrung.menge,
                gebuehr_eur=ausfuehrung.gebuehr_eur, lot_einstandspreis=lot.einstandspreis, lot_kauf_zeit=lot.kauf_zeit,
            )
            self._protokolliere(trade)
            self.anker_preis = preis

        self.lots = verbleibende_lots

    def _versuche_kauf(self, preis: float, zeit: pd.Timestamp, y_eur: float) -> None:
        if y_eur <= 0:
            return
        ausfuehrung = self.broker.kaufe(self.paar, y_eur, preis, self.praezision)
        if not ausfuehrung.erfolgreich:
            logger.info("Kauf fuer %s nicht ausgefuehrt: %s", self.paar, ausfuehrung.grund)
            return

        gebuehr_pro_einheit = ausfuehrung.gebuehr_eur / ausfuehrung.menge if ausfuehrung.menge else 0.0
        einstandspreis = ausfuehrung.fill_preis + gebuehr_pro_einheit
        kosten = ausfuehrung.menge * ausfuehrung.fill_preis + ausfuehrung.gebuehr_eur
        if kosten > self.kapital_eur:
            logger.warning("Kauf fuer %s uebersteigt verfuegbares Kapital - wird trotzdem verbucht (Kapital negativ geloggt)", self.paar)

        self.kapital_eur -= kosten
        self.lots.append(Lot(einstandspreis=einstandspreis, fill_preis_kauf=ausfuehrung.fill_preis, menge=ausfuehrung.menge, kauf_zeit=zeit))
        trade = Trade(zeit=zeit, art="KAUF", preis=ausfuehrung.fill_preis, menge=ausfuehrung.menge, gebuehr_eur=ausfuehrung.gebuehr_eur)
        self._protokolliere(trade)
        self.anker_preis = preis

    def verarbeite_tick(
        self,
        aktueller_preis: float,
        aktuelle_zeit: pd.Timestamp,
        regime_ergebnis: RegimeErgebnis,
        letzte_preise: pd.Series,
        atr_wert: float | None,
        gesamtkapital_bot: float,
    ) -> None:
        """Verarbeitet einen Preiszyklus: Waechter pruefen, dann ggf. handeln."""
        if self.zustand == PaarZustand.GESTOPPT:
            return

        if self.anker_preis is None:
            self.anker_preis = aktueller_preis

        if self.zustand == PaarZustand.PAUSIERT:
            self._verarbeite_pausiert(aktueller_preis, aktuelle_zeit, regime_ergebnis)
            return

        ausbruch = pruefe_ausbruch(aktueller_preis, self.anker_preis, letzte_preise, atr_wert, self.config.ausbruch)
        if ausbruch.ausgeloest:
            self.wiedereinstieg_zustand = WiedereinstiegZustand(pausiert_seit=aktuelle_zeit, anker_preis_bei_pause=self.anker_preis)
            self._wechsel_zustand(PaarZustand.PAUSIERT, f"Ausbruchswaechter: {ausbruch.grund}", aktuelle_zeit)
            return

        self.gate_handelt = regime_ergebnis.handel_erlaubt
        neuer_zustand = PaarZustand.HANDELT if self.gate_handelt else PaarZustand.NUR_VERKAUFEN

        self._verkaufe_faellige_lots(aktueller_preis, aktuelle_zeit)

        if neuer_zustand == PaarZustand.HANDELT:
            investiert = self.investierter_wert(aktueller_preis)
            exposure_blockiert = ist_kauf_durch_exposure_blockiert(investiert, gesamtkapital_bot, self.config.exposure)
            veraenderung_prozent = (aktueller_preis - self.anker_preis) / self.anker_preis * 100

            if not exposure_blockiert and veraenderung_prozent <= -self.x_prozent:
                konfidenz_effektiv = 0.0
                if regime_ergebnis.kurz and regime_ergebnis.mittel:
                    konfidenz_effektiv = min(regime_ergebnis.kurz.hurst.konfidenz, regime_ergebnis.mittel.hurst.konfidenz)
                exposure_limit_eur = max(
                    self.config.exposure.max_exposure_prozent / 100 * gesamtkapital_bot - investiert, 0.0
                )
                mindest_betrag_eur = self.praezision.mindest_ordervolumen * aktueller_preis
                y_eur = berechne_y_betrag(
                    regime_ergebnis.oszillations_score, konfidenz_effektiv, self.config.y,
                    verfuegbares_kapital_eur=self.kapital_eur, exposure_limit_eur=exposure_limit_eur,
                    mindest_betrag_eur=mindest_betrag_eur,
                )
                self._versuche_kauf(aktueller_preis, aktuelle_zeit, y_eur)
            elif exposure_blockiert:
                neuer_zustand = PaarZustand.NUR_VERKAUFEN

        grund = "Gate bestanden" if neuer_zustand == PaarZustand.HANDELT else regime_ergebnis.grund
        self._wechsel_zustand(neuer_zustand, grund, aktuelle_zeit)

    def _verarbeite_pausiert(self, aktueller_preis: float, aktuelle_zeit: pd.Timestamp, regime_ergebnis: RegimeErgebnis) -> None:
        if self.wiedereinstieg_zustand is None:
            # Sollte nicht vorkommen, aber kein Absturz: Zustand reparieren statt haengen bleiben.
            self.wiedereinstieg_zustand = WiedereinstiegZustand(pausiert_seit=aktuelle_zeit, anker_preis_bei_pause=aktueller_preis)

        h_aktuell = 0.5
        if regime_ergebnis.kurz and regime_ergebnis.mittel:
            h_aktuell = max(regime_ergebnis.kurz.hurst.h, regime_ergebnis.mittel.hurst.h)

        ergebnis = pruefe_wiedereinstieg(
            self.wiedereinstieg_zustand, aktueller_preis, aktuelle_zeit, h_aktuell, self.config.gate.h_start, self.config.wiedereinstieg
        )
        if "WARNUNG" in ergebnis.grund:
            logger.warning("%s: %s", self.paar, ergebnis.grund)

        if ergebnis.wieder_einsteigen:
            self.anker_preis = ergebnis.neuer_anker_preis
            self.wiedereinstieg_zustand = None
            neuer_zustand = PaarZustand.HANDELT if regime_ergebnis.handel_erlaubt else PaarZustand.NUR_VERKAUFEN
            self._wechsel_zustand(neuer_zustand, f"Wiedereinstieg: {ergebnis.grund}", aktuelle_zeit)
