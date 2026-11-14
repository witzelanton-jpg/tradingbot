"""Hauptschleife des Bots: verbindet Scanner, Regime, Optimierer, Strategie und Waechter.

Der `BotRunner` haelt den Gesamtzustand (freies Kapital + Zustand je Paar),
fuehrt periodisch den Scan aus, optimiert X und das Zeitskala-Profil nach Plan
und speichert nach jeder Orderausfuehrung atomar (Idempotenz, Abschnitt 5).

Ein `KeyboardInterrupt` (Strg + C) wird sauber behandelt: die Schleife bricht ab,
der Zustand wird final gespeichert und offene Live-Orders werden storniert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .broker import Broker, PaarPraezision
from .config import AppConfig, ZeitskalaProfil
from .data import hole_multiskalen_ohlc, hole_ohlc
from .guards import PaarZustand, berechne_atr, ist_kill_switch_ausgeloest
from .kraken_api import KrakenAPI, KrakenAPIError, KrakenVerbindungsFehler
from .logging_setup import protokolliere_zustandswechsel
from .optimizer import gebuehren_breakeven_x_prozent, waehle_zeitskala_profil, walk_forward_optimize
from .regime import RegimeErgebnis, bewerte_regime
from .scanner import PaarInfo, formatiere_ranking, pruefe_watchlist, ranke_paare, verteile_kapital
from .state import BotZustand, PaarZustandsDaten, lade_zustand, speichere_zustand
from .strategy import PaarStrategie

logger = logging.getLogger("hurstbot.runner")


def profil_nach_name(config: AppConfig, name: str) -> ZeitskalaProfil:
    """Sucht ein Zeitskala-Profil nach Namen, mit Fallback auf das konfigurierte Standardprofil."""
    kandidaten = config.zeitskala_auswahl.kandidaten
    for profil in kandidaten:
        if profil.name == name:
            return profil
    for profil in kandidaten:
        if profil.name == config.zeitskala_auswahl.fallback_profil:
            return profil
    return kandidaten[0]


@dataclass
class BotRunner:
    """Orchestriert den gesamten Handelsbetrieb ueber alle Paare der Watchlist."""

    config: AppConfig
    api: KrakenAPI
    broker: Broker
    zustand_pfad: Path
    trade_log_pfad: Path
    zustand_log_pfad: Path

    def __post_init__(self) -> None:
        self.paar_infos: dict[str, PaarInfo] = {}
        self.strategien: dict[str, PaarStrategie] = {}
        self.bot_zustand: BotZustand | None = None
        self._letzte_x_optimierung: dict[str, pd.Timestamp] = {}
        self._letzte_zeitskala_bewertung: dict[str, pd.Timestamp] = {}

    # ---------- Start / Zustand ----------

    def starte(self) -> None:
        """Prueft die Watchlist bei Kraken und stellt den gespeicherten Zustand wieder her."""
        self.paar_infos = pruefe_watchlist(self.api, self.config.scanner.watchlist)

        geladen = lade_zustand(self.zustand_pfad)
        if geladen is None:
            logger.info("Kein gueltiger Zustand gefunden - starte frisch mit %.2f EUR", self.config.start_kapital_eur)
            self.bot_zustand = BotZustand(
                start_kapital_eur=self.config.start_kapital_eur,
                kapital_frei_eur=self.config.start_kapital_eur,
                erstellt_am=pd.Timestamp.now(tz="UTC").isoformat(),
            )
        else:
            self.bot_zustand = geladen
            logger.info(
                "Zustand wiederhergestellt: %.2f EUR frei, %s Paare mit Historie",
                geladen.kapital_frei_eur, len(geladen.paare),
            )

        for paar, info in self.paar_infos.items():
            self._stelle_strategie_her(paar, info.praezision)

    def _stelle_strategie_her(self, paar: str, praezision: PaarPraezision) -> None:
        gespeichert = self.bot_zustand.paare.get(paar)
        profil_name = gespeichert.zeitskala_profil_name if gespeichert else self.config.zeitskala_auswahl.fallback_profil
        breakeven = gebuehren_breakeven_x_prozent(self.config.gebuehren)

        strategie = PaarStrategie(
            paar=paar,
            config=self.config,
            broker=self.broker,
            praezision=praezision,
            trade_log_pfad=self.trade_log_pfad,
            zustand_log_pfad=self.zustand_log_pfad,
        )
        if gespeichert:
            strategie.zustand = PaarZustand(gespeichert.zustand)
            strategie.lots = list(gespeichert.lots)
            strategie.kapital_eur = gespeichert.kapital_eur
            strategie.anker_preis = gespeichert.anker_preis
            strategie.gate_handelt = gespeichert.gate_handelt
            strategie.x_prozent = gespeichert.x_prozent or breakeven * 2
        else:
            strategie.x_prozent = breakeven * 2

        strategie.zeitskala_profil = profil_nach_name(self.config, profil_name)
        self.strategien[paar] = strategie

    def speichere(self) -> None:
        """Schreibt den aktuellen Gesamtzustand atomar auf die Platte."""
        if self.bot_zustand is None:
            return
        for paar, strategie in self.strategien.items():
            vorher = self.bot_zustand.paare.get(paar)
            self.bot_zustand.paare[paar] = PaarZustandsDaten(
                paar=paar,
                zustand=strategie.zustand.value,
                lots=list(strategie.lots),
                kapital_eur=strategie.kapital_eur,
                anker_preis=strategie.anker_preis,
                gate_handelt=strategie.gate_handelt,
                zeitskala_profil_name=getattr(strategie, "zeitskala_profil", profil_nach_name(self.config, "5m")).name,
                letzte_zeitskala_neubewertung=(
                    self._letzte_zeitskala_bewertung[paar].isoformat() if paar in self._letzte_zeitskala_bewertung else None
                ),
                x_prozent=strategie.x_prozent,
                letzte_x_optimierung=(
                    self._letzte_x_optimierung[paar].isoformat() if paar in self._letzte_x_optimierung else None
                ),
                pausiert_seit=(
                    strategie.wiedereinstieg_zustand.pausiert_seit.isoformat()
                    if strategie.wiedereinstieg_zustand else None
                ),
                anker_preis_bei_pause=(
                    strategie.wiedereinstieg_zustand.anker_preis_bei_pause if strategie.wiedereinstieg_zustand else None
                ),
                stabile_kerzen_zaehler=(
                    strategie.wiedereinstieg_zustand.stabile_kerzen_zaehler if strategie.wiedereinstieg_zustand else 0
                ),
                wiedereinstieg_warnung_ausgegeben=(
                    strategie.wiedereinstieg_zustand.warnung_ausgegeben if strategie.wiedereinstieg_zustand else False
                ),
            )
        speichere_zustand(self.bot_zustand, self.zustand_pfad)

    # ---------- Bewertung / Optimierung ----------

    def bewerte_paar(self, paar: str) -> RegimeErgebnis | None:
        """Laedt Mehrskalen-Daten und bewertet das Regime eines Paares."""
        strategie = self.strategien[paar]
        kraken_name = self.paar_infos[paar].kraken_name
        try:
            daten = hole_multiskalen_ohlc(self.api, kraken_name, self.config.hurst.multiskalen, self.config.pfade.daten_verzeichnis)
        except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
            logger.warning("Konnte Daten fuer %s nicht laden: %s", paar, exc)
            return None

        return bewerte_regime(
            daten, self.config.hurst, self.config.gate, strategie.zeitskala_profil, strategie.gate_handelt
        )

    def optimiere_x_falls_faellig(self, paar: str, jetzt: pd.Timestamp) -> None:
        """Fuehrt die X-Optimierung aus, wenn das Intervall abgelaufen ist (Default 6 h)."""
        letzte = self._letzte_x_optimierung.get(paar)
        if letzte is not None:
            stunden_seit = (jetzt - letzte).total_seconds() / 3600
            if stunden_seit < self.config.optimizer.neuoptimierung_stunden:
                return

        strategie = self.strategien[paar]
        kraken_name = self.paar_infos[paar].kraken_name
        try:
            df = hole_ohlc(
                self.api, kraken_name, strategie.zeitskala_profil.intervall_minuten, self.config.pfade.daten_verzeichnis
            )
            ergebnis = walk_forward_optimize(
                df, self.config.optimizer, self.config.gebuehren,
                y_eur=self.config.y.y_max_eur,
                start_kapital=self.bot_zustand.start_kapital_eur,
                vorheriger_x=strategie.x_prozent,
            )
        except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
            logger.warning("X-Optimierung fuer %s uebersprungen (Datenfehler): %s", paar, exc)
            return
        except ValueError as exc:
            logger.warning("X-Optimierung fuer %s nicht moeglich: %s", paar, exc)
            return

        self._letzte_x_optimierung[paar] = jetzt

        if not ergebnis.hat_belastbare_evidenz:
            logger.warning(
                "X fuer %s bleibt bei %.4f %%: die Optimierung hatte keinen einzigen "
                "Out-of-Sample-Round-Trip und ist damit nicht belastbar.",
                paar, strategie.x_prozent,
            )
            return

        logger.info(
            "X fuer %s aktualisiert: %.4f %% -> %.4f %% (Out-of-Sample-Score %.2f EUR, %.1f Round-Trips)",
            paar, strategie.x_prozent, ergebnis.x_prozent, ergebnis.mittlerer_out_of_sample_score, ergebnis.mittlere_roundtrips,
        )
        strategie.x_prozent = ergebnis.x_prozent

    def bewerte_zeitskala_falls_faellig(self, paar: str, jetzt: pd.Timestamp) -> None:
        """Bewertet die paarspezifische Zeitskala neu (Default: hoechstens alle 24 h).

        Ein Wechsel erfolgt nur, wenn der Kandidat den Out-of-Sample-Nettogewinn des
        aktuellen Profils um mindestens `wechsel_vorteil_prozent` uebertrifft.
        Offene Lots bleiben beim Wechsel unberuehrt.
        """
        auswahl_config = self.config.zeitskala_auswahl
        letzte = self._letzte_zeitskala_bewertung.get(paar)
        if letzte is not None:
            stunden_seit = (jetzt - letzte).total_seconds() / 3600
            if stunden_seit < auswahl_config.neubewertung_stunden:
                return

        strategie = self.strategien[paar]
        kraken_name = self.paar_infos[paar].kraken_name

        kandidaten_daten: dict[str, pd.DataFrame] = {}
        for profil in auswahl_config.kandidaten:
            try:
                kandidaten_daten[profil.name] = hole_ohlc(
                    self.api, kraken_name, profil.intervall_minuten, self.config.pfade.daten_verzeichnis
                )
            except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
                logger.warning("Zeitskala-Kandidat %s fuer %s nicht ladbar: %s", profil.name, paar, exc)

        if not kandidaten_daten:
            return

        fallback = profil_nach_name(self.config, auswahl_config.fallback_profil)
        bestes_profil, grund, alle_ergebnisse = waehle_zeitskala_profil(
            kandidaten_daten, auswahl_config.kandidaten, self.config.optimizer, self.config.gebuehren,
            y_eur=self.config.y.y_max_eur, start_kapital=self.bot_zustand.start_kapital_eur,
            mindest_roundtrips=auswahl_config.mindest_roundtrips, fallback_profil=fallback,
        )
        self._letzte_zeitskala_bewertung[paar] = jetzt

        aktuelles_profil = strategie.zeitskala_profil
        if bestes_profil.name == aktuelles_profil.name:
            logger.info("Zeitskala fuer %s bleibt bei %s (%s)", paar, aktuelles_profil.name, grund)
            return

        aktuelles_ergebnis = next(
            (e for e in alle_ergebnisse if e.profil.name == aktuelles_profil.name and e.optimierung), None
        )
        bestes_ergebnis = next((e for e in alle_ergebnisse if e.profil.name == bestes_profil.name and e.optimierung), None)

        if aktuelles_ergebnis and bestes_ergebnis:
            alt_score = aktuelles_ergebnis.optimierung.mittlerer_out_of_sample_score
            neu_score = bestes_ergebnis.optimierung.mittlerer_out_of_sample_score
            noetige_schwelle = abs(alt_score) * (1 + auswahl_config.wechsel_vorteil_prozent / 100)
            if alt_score > 0 and neu_score < noetige_schwelle:
                logger.info(
                    "Zeitskala fuer %s bleibt bei %s: Kandidat %s uebertrifft nur %.2f statt noetiger %.2f EUR",
                    paar, aktuelles_profil.name, bestes_profil.name, neu_score, noetige_schwelle,
                )
                return

        logger.warning(
            "Zeitskala-Wechsel fuer %s: %s -> %s. Grund: %s. Offene Lots (%s) bleiben erhalten.",
            paar, aktuelles_profil.name, bestes_profil.name, grund, len(strategie.lots),
        )
        strategie.zeitskala_profil = bestes_profil

    # ---------- Ein Zyklus ----------

    def fuehre_zyklus_aus(self, jetzt: pd.Timestamp | None = None) -> None:
        """Fuehrt einen kompletten Scan-und-Handel-Zyklus ueber alle Paare aus."""
        jetzt = jetzt or pd.Timestamp.now(tz="UTC")
        if self.bot_zustand.global_zustand == PaarZustand.GESTOPPT.value:
            return

        bewertungen: dict[str, RegimeErgebnis] = {}
        for paar in self.strategien:
            ergebnis = self.bewerte_paar(paar)
            if ergebnis is not None:
                bewertungen[paar] = ergebnis

        if not bewertungen:
            logger.warning("Kein Paar konnte in diesem Zyklus bewertet werden - warte auf naechsten Durchlauf")
            return

        ranking = ranke_paare(bewertungen)
        logger.info("Aktuelles Ranking:\n%s", formatiere_ranking(ranking))

        paare_mit_lots = {paar for paar, s in self.strategien.items() if s.lots}
        zuteilung = verteile_kapital(ranking, self.config, self.bot_zustand.kapital_frei_eur, paare_mit_lots)

        for paar in zuteilung.aktive_paare:
            strategie = self.strategien[paar]
            zugeteilt = zuteilung.kapital_je_paar.get(paar, 0.0)
            if zugeteilt > strategie.kapital_eur:
                nachschub = min(zugeteilt - strategie.kapital_eur, self.bot_zustand.kapital_frei_eur)
                strategie.kapital_eur += nachschub
                self.bot_zustand.kapital_frei_eur -= nachschub

            self.optimiere_x_falls_faellig(paar, jetzt)
            self.bewerte_zeitskala_falls_faellig(paar, jetzt)

        for paar in list(zuteilung.aktive_paare) + list(zuteilung.nur_verkaufen_paare):
            self._verarbeite_paar(paar, bewertungen.get(paar), jetzt, nur_verkaufen=paar in zuteilung.nur_verkaufen_paare)

        self._pruefe_kill_switch(jetzt)
        self.speichere()

    def _verarbeite_paar(
        self, paar: str, regime_ergebnis: RegimeErgebnis | None, jetzt: pd.Timestamp, nur_verkaufen: bool
    ) -> None:
        if regime_ergebnis is None:
            return
        strategie = self.strategien[paar]
        kraken_name = self.paar_infos[paar].kraken_name

        try:
            ticker = self.api.ticker([kraken_name])
            schluessel = next(iter(ticker))
            aktueller_preis = float(ticker[schluessel]["c"][0])
        except (KrakenAPIError, KrakenVerbindungsFehler, KeyError, StopIteration) as exc:
            logger.warning("Konnte Ticker fuer %s nicht lesen: %s", paar, exc)
            return

        try:
            kerzen = hole_ohlc(
                self.api, kraken_name, strategie.zeitskala_profil.intervall_minuten, self.config.pfade.daten_verzeichnis
            )
        except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
            logger.warning("Konnte Kerzen fuer %s nicht laden: %s", paar, exc)
            return

        atr_wert = None
        if len(kerzen) > self.config.ausbruch.atr_periode:
            atr_serie = berechne_atr(kerzen, self.config.ausbruch.atr_periode)
            letzter_atr = atr_serie.iloc[-1]
            atr_wert = float(letzter_atr) if pd.notna(letzter_atr) else None

        letzte_preise = kerzen["close"].tail(self.config.ausbruch.ausbruch_zeitfenster_kerzen)

        # Ein Paar in Rotation darf nur noch verkaufen: das Gate wird dafuer
        # kuenstlich auf "nicht erlaubt" gesetzt, Lots bleiben aber handelbar.
        effektives_regime = regime_ergebnis
        if nur_verkaufen and regime_ergebnis.handel_erlaubt:
            effektives_regime = RegimeErgebnis(
                kurz=regime_ergebnis.kurz, mittel=regime_ergebnis.mittel, lang=regime_ergebnis.lang,
                oszillations_score=regime_ergebnis.oszillations_score, handel_erlaubt=False,
                grund="Paar-Rotation: nur verkaufen, keine neuen Kaeufe",
            )

        strategie.verarbeite_tick(
            aktueller_preis=aktueller_preis,
            aktuelle_zeit=jetzt,
            regime_ergebnis=effektives_regime,
            letzte_preise=letzte_preise,
            atr_wert=atr_wert,
            gesamtkapital_bot=self.gesamtwert_bot(),
        )

    def gesamtwert_bot(self) -> float:
        """Freies Kapital plus Kapital und Bestandswert aller Paare."""
        gesamt = self.bot_zustand.kapital_frei_eur
        for strategie in self.strategien.values():
            gesamt += strategie.kapital_eur
            gesamt += sum(lot.menge * lot.fill_preis_kauf for lot in strategie.lots)
        return gesamt

    def _pruefe_kill_switch(self, jetzt: pd.Timestamp) -> None:
        gesamtwert = self.gesamtwert_bot()
        if not ist_kill_switch_ausgeloest(gesamtwert, self.bot_zustand.start_kapital_eur, self.config.kill_switch):
            return

        grund = (
            f"Kill-Switch: Gesamtwert {gesamtwert:.2f} EUR liegt mehr als "
            f"{self.config.kill_switch.max_verlust_prozent} % unter dem Startkapital "
            f"{self.bot_zustand.start_kapital_eur:.2f} EUR"
        )
        logger.critical(grund)
        for paar, strategie in self.strategien.items():
            if strategie.zustand != PaarZustand.GESTOPPT:
                protokolliere_zustandswechsel(
                    self.zustand_log_pfad, paar, strategie.zustand.value, PaarZustand.GESTOPPT.value, grund, jetzt
                )
                strategie.zustand = PaarZustand.GESTOPPT
        self.bot_zustand.global_zustand = PaarZustand.GESTOPPT.value

    # ---------- Dauerbetrieb ----------

    def laufe(self, max_zyklen: int | None = None, schlafen=None) -> None:
        """Dauerbetrieb bis Strg + C, Kill-Switch oder `max_zyklen` erreicht sind."""
        import time

        schlafen = schlafen or time.sleep
        zyklus = 0
        try:
            while max_zyklen is None or zyklus < max_zyklen:
                if self.bot_zustand.global_zustand == PaarZustand.GESTOPPT.value:
                    logger.critical("Bot ist GESTOPPT (Kill-Switch) - keine weiteren Zyklen.")
                    break
                self.fuehre_zyklus_aus()
                zyklus += 1
                if max_zyklen is None or zyklus < max_zyklen:
                    schlafen(self.config.abfrage_sekunden)
        except KeyboardInterrupt:
            logger.info("Abbruch durch Nutzer (Strg + C) - beende sauber.")
        finally:
            self.beende_sauber()

    def beende_sauber(self) -> None:
        """Storniert offene Orders (nur im Live-Modus moeglich) und speichert den Zustand."""
        from .broker import KrakenLiveBroker

        if isinstance(self.broker, KrakenLiveBroker):
            try:
                offene = self.api.open_orders()
                for order_id in offene.get("open", {}):
                    logger.info("Storniere offene Order %s", order_id)
                    self.api.cancel_order(order_id)
            except (KrakenAPIError, KrakenVerbindungsFehler) as exc:
                logger.warning(
                    "Konnte offene Orders beim Beenden nicht pruefen - bitte im Kraken-Konto "
                    "manuell nachsehen: %s", exc
                )

        self.speichere()
        logger.info("Zustand gespeichert unter %s", self.zustand_pfad)
