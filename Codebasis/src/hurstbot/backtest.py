"""Lot-basierte Grid-Simulation mit Gebuehren und Spread.

`LotBuch` ist die zentrale Buchfuehrungs-Klasse: Sie verwaltet offene Lots
(jeder Kauf ist ein eigenes Lot mit Einstandspreis und Menge) und verkauft nur
Lots, deren Nettoerloes nach Gebuehren und Spread echt ueber dem Einstandspreis
liegt. Diese Klasse wird sowohl vom Backtest (`simulate`) als auch von der
Live-/Paper-Strategie (`strategy.py`) verwendet, damit die Buchfuehrung
unabhaengig von der Kerzenlaenge und vom Ausfuehrungsmodus identisch ist.

Es gibt KEINEN Blick in die Zukunft: `simulate` iteriert streng sequenziell
ueber die Kerzen und verwendet pro Schritt ausschliesslich den aktuellen und
vergangene Werte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Lot:
    """Ein einzelner Kauf: Menge, Einstandspreis (inkl. Gebuehr) und Zeitpunkt."""

    einstandspreis: float  # Break-even-Preis pro Einheit, bereits inkl. Kauf-Gebuehr
    fill_preis_kauf: float  # tatsaechlicher Ausfuehrungspreis (vor Gebuehr), fuer Reporting
    menge: float
    kauf_zeit: pd.Timestamp


@dataclass
class Trade:
    """Ein ausgefuehrter Trade (Kauf oder Verkauf) im Backtest/Paper-Betrieb."""

    zeit: pd.Timestamp
    art: str  # "KAUF" oder "VERKAUF"
    preis: float  # Fill-Preis (nach Spread, vor Gebuehr)
    menge: float
    gebuehr_eur: float
    lot_einstandspreis: float | None = None  # nur bei VERKAUF: Einstandspreis des verkauften Lots
    lot_kauf_zeit: pd.Timestamp | None = None  # nur bei VERKAUF: wann das Lot eroeffnet wurde


@dataclass
class BacktestResult:
    """Kennzahlen einer Backtest-Simulation."""

    trades: list[Trade]
    end_kapital_eur: float
    end_bestand_wert_eur: float
    netto_gewinn_eur: float
    max_drawdown_prozent: float
    anzahl_roundtrips: int
    mittlere_amplitude_prozent: float
    mittlere_haltedauer_stunden: float
    offene_lots_anzahl: int
    offene_lots_mittlere_dauer_stunden: float
    equity_kurve: pd.Series = field(repr=False)


class LotBuch:
    """Lot-basierte Buchfuehrung: Kauf erzeugt ein Lot, Verkauf schliesst ein Lot nur,
    wenn der Nettoerloes nach Gebuehren und Spread echt ueber dem Einstandspreis liegt."""

    def __init__(self, start_kapital_eur: float, gebuehr_prozent: float, spread_prozent: float) -> None:
        self.kapital_eur = start_kapital_eur
        self.lots: list[Lot] = []
        self.trades: list[Trade] = []
        self._fee = gebuehr_prozent / 100
        self._halber_spread = spread_prozent / 100 / 2

    def investierter_wert(self, aktueller_preis: float) -> float:
        return sum(lot.menge for lot in self.lots) * aktueller_preis

    def gesamtwert(self, aktueller_preis: float) -> float:
        return self.kapital_eur + self.investierter_wert(aktueller_preis)

    def kaufen(self, preis: float, zeit: pd.Timestamp, y_eur: float) -> bool:
        """Kauft fuer y_eur, sofern genug Kapital vorhanden ist. Gibt zurueck, ob gekauft wurde."""
        if y_eur <= 0:
            return False
        fill_preis = preis * (1 + self._halber_spread)
        kosten_mit_gebuehr = y_eur * (1 + self._fee)
        if kosten_mit_gebuehr > self.kapital_eur:
            return False

        menge = y_eur / fill_preis
        einstandspreis = fill_preis * (1 + self._fee)
        self.kapital_eur -= kosten_mit_gebuehr
        self.lots.append(Lot(einstandspreis=einstandspreis, fill_preis_kauf=fill_preis, menge=menge, kauf_zeit=zeit))
        self.trades.append(
            Trade(zeit=zeit, art="KAUF", preis=fill_preis, menge=menge, gebuehr_eur=kosten_mit_gebuehr - y_eur)
        )
        return True

    def verkaufe_faellige_lots(self, preis: float, zeit: pd.Timestamp, x_prozent: float) -> list[Trade]:
        """Verkauft alle Lots, deren Trigger erreicht ist UND die netto profitabel waeren."""
        fill_preis = preis * (1 - self._halber_spread)
        netto_verkaufspreis_pro_einheit = fill_preis * (1 - self._fee)

        verkaufte_trades: list[Trade] = []
        verbleibende_lots: list[Lot] = []
        for lot in self.lots:
            trigger_preis = lot.einstandspreis * (1 + x_prozent / 100)
            ist_profitabel = netto_verkaufspreis_pro_einheit > lot.einstandspreis
            if preis >= trigger_preis and ist_profitabel:
                erloes = lot.menge * netto_verkaufspreis_pro_einheit
                self.kapital_eur += erloes
                trade = Trade(
                    zeit=zeit,
                    art="VERKAUF",
                    preis=fill_preis,
                    menge=lot.menge,
                    gebuehr_eur=lot.menge * fill_preis * self._fee,
                    lot_einstandspreis=lot.einstandspreis,
                    lot_kauf_zeit=lot.kauf_zeit,
                )
                self.trades.append(trade)
                verkaufte_trades.append(trade)
            else:
                verbleibende_lots.append(lot)
        self.lots = verbleibende_lots
        return verkaufte_trades


def simulate(
    df: pd.DataFrame,
    x_prozent: float,
    y_eur: float,
    gebuehr_prozent: float,
    spread_prozent: float,
    start_kapital_eur: float,
    preis_spalte: str = "close",
    zeit_spalte: str = "zeit",
) -> BacktestResult:
    """Simuliert die Grid-Strategie streng sequenziell (kein Look-Ahead-Bias).

    Anker-Logik: Faellt der Preis um x_prozent unter den aktuellen Referenzpreis
    (Anker), wird ein neues Lot fuer y_eur gekauft. Nach jedem Trade (Kauf oder
    Verkauf) wird der Anker auf den aktuellen Preis zurueckgesetzt, sodass sich
    bei anhaltender Bewegung eine Leiter von Trades ergibt. Verkauft wird je
    Preisschritt unabhaengig vom Anker jedes Lot, dessen individueller
    Einstandspreis plus x_prozent erreicht ist UND das netto profitabel ist.
    """
    buch = LotBuch(start_kapital_eur, gebuehr_prozent, spread_prozent)
    anker: float | None = None
    equity_werte: list[float] = []
    zeit_index: list[pd.Timestamp] = []

    for _, zeile in df.iterrows():
        preis = float(zeile[preis_spalte])
        zeit = zeile[zeit_spalte]

        if anker is None:
            anker = preis
            equity_werte.append(buch.gesamtwert(preis))
            zeit_index.append(zeit)
            continue

        gehandelt = bool(buch.verkaufe_faellige_lots(preis, zeit, x_prozent))

        veraenderung_prozent = (preis - anker) / anker * 100
        if veraenderung_prozent <= -x_prozent:
            if buch.kaufen(preis, zeit, y_eur):
                gehandelt = True

        if gehandelt:
            anker = preis

        equity_werte.append(buch.gesamtwert(preis))
        zeit_index.append(zeit)

    return _ergebnis_aus_buch(buch, equity_werte, zeit_index, df, preis_spalte, zeit_spalte)


def _ergebnis_aus_buch(
    buch: LotBuch,
    equity_werte: list[float],
    zeit_index: list[pd.Timestamp],
    df: pd.DataFrame,
    preis_spalte: str,
    zeit_spalte: str,
) -> BacktestResult:
    equity_kurve = pd.Series(equity_werte, index=zeit_index)

    letzter_preis = float(df[preis_spalte].iloc[-1]) if len(df) else 0.0
    end_bestand_wert = buch.investierter_wert(letzter_preis)
    start_kapital = equity_kurve.iloc[0] if len(equity_kurve) else buch.kapital_eur
    netto_gewinn = buch.gesamtwert(letzter_preis) - start_kapital

    if len(equity_kurve) > 1:
        laufendes_maximum = equity_kurve.cummax()
        drawdown = (laufendes_maximum - equity_kurve) / laufendes_maximum.replace(0, np.nan)
        max_drawdown_prozent = float(drawdown.max(skipna=True) * 100) if drawdown.notna().any() else 0.0
    else:
        max_drawdown_prozent = 0.0

    verkaeufe = [t for t in buch.trades if t.art == "VERKAUF"]
    anzahl_roundtrips = len(verkaeufe)

    if verkaeufe:
        amplituden = [
            (t.preis - (t.lot_einstandspreis or t.preis)) / t.lot_einstandspreis * 100
            for t in verkaeufe
            if t.lot_einstandspreis
        ]
        mittlere_amplitude = float(np.mean(amplituden)) if amplituden else 0.0
        haltedauern = [
            (t.zeit - t.lot_kauf_zeit).total_seconds() / 3600
            for t in verkaeufe
            if t.lot_kauf_zeit is not None
        ]
        mittlere_haltedauer = float(np.mean(haltedauern)) if haltedauern else 0.0
    else:
        mittlere_amplitude = 0.0
        mittlere_haltedauer = 0.0

    letzte_zeit = zeit_index[-1] if zeit_index else None
    offene_dauern = (
        [(letzte_zeit - lot.kauf_zeit).total_seconds() / 3600 for lot in buch.lots]
        if letzte_zeit is not None
        else []
    )
    offene_lots_mittlere_dauer = float(np.mean(offene_dauern)) if offene_dauern else 0.0

    return BacktestResult(
        trades=buch.trades,
        end_kapital_eur=buch.kapital_eur,
        end_bestand_wert_eur=end_bestand_wert,
        netto_gewinn_eur=netto_gewinn,
        max_drawdown_prozent=max_drawdown_prozent,
        anzahl_roundtrips=anzahl_roundtrips,
        mittlere_amplitude_prozent=mittlere_amplitude,
        mittlere_haltedauer_stunden=mittlere_haltedauer,
        offene_lots_anzahl=len(buch.lots),
        offene_lots_mittlere_dauer_stunden=offene_lots_mittlere_dauer,
        equity_kurve=equity_kurve,
    )
