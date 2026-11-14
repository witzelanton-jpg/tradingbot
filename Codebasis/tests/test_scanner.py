"""Tests fuer den Watchlist-Scanner: Paar-Pruefung, Ranking, Kapitalverteilung.

Alle Kraken-Zugriffe laufen gegen gemockte Antworten, nie gegen die echte API.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hurstbot.config import AppConfig, ScannerConfig
from hurstbot.hurst import HurstErgebnis
from hurstbot.kraken_api import KrakenVerbindungsFehler
from hurstbot.regime import RegimeErgebnis, SkalenErgebnis
from hurstbot.scanner import formatiere_ranking, pruefe_watchlist, ranke_paare, verteile_kapital

# Gemockte AssetPairs-Antwort, im Format der echten Kraken-API.
ASSET_PAIRS_ANTWORT = {
    "XXBTZEUR": {"altname": "XBTEUR", "pair_decimals": 1, "lot_decimals": 8, "ordermin": "0.0001"},
    "XDGEUR": {"altname": "XDGEUR", "pair_decimals": 7, "lot_decimals": 8, "ordermin": "50"},
    "XETHZEUR": {"altname": "ETHEUR", "pair_decimals": 2, "lot_decimals": 8, "ordermin": "0.01"},
}


def _regime(handel_erlaubt=True, score=0.5, h=0.3, konfidenz=0.9, grund="test"):
    hurst = HurstErgebnis(h=h, ci_untere_grenze=h - 0.05, ci_obere_grenze=h + 0.05, konfidenz=konfidenz)
    skala = SkalenErgebnis(hurst=hurst, monotonie={"monotonie_score": 0.1})
    return RegimeErgebnis(
        kurz=skala, mittel=skala, lang=None, oszillations_score=score, handel_erlaubt=handel_erlaubt, grund=grund
    )


class TestPruefeWatchlist:
    def test_liest_praezision_und_mindestvolumen(self):
        api = MagicMock()
        api.asset_pairs.return_value = ASSET_PAIRS_ANTWORT

        infos = pruefe_watchlist(api, ["XXBTZEUR", "XDGEUR"])

        assert set(infos) == {"XXBTZEUR", "XDGEUR"}
        assert infos["XXBTZEUR"].praezision.preis_dezimalstellen == 1
        assert infos["XXBTZEUR"].praezision.mindest_ordervolumen == pytest.approx(0.0001)
        assert infos["XDGEUR"].praezision.mindest_ordervolumen == pytest.approx(50.0)

    def test_nicht_existierendes_paar_wird_uebersprungen(self):
        api = MagicMock()
        api.asset_pairs.return_value = ASSET_PAIRS_ANTWORT

        infos = pruefe_watchlist(api, ["XXBTZEUR", "GIBTESNICHTEUR"])

        assert "XXBTZEUR" in infos
        assert "GIBTESNICHTEUR" not in infos

    def test_zuordnung_ueber_altname(self):
        api = MagicMock()
        api.asset_pairs.return_value = ASSET_PAIRS_ANTWORT

        # Nutzer schreibt "ETHEUR" in die Watchlist, Kraken antwortet mit "XETHZEUR".
        infos = pruefe_watchlist(api, ["ETHEUR"])

        assert "ETHEUR" in infos
        assert infos["ETHEUR"].kraken_name == "XETHZEUR"

    def test_kein_gueltiges_paar_wirft_fehler(self):
        api = MagicMock()
        api.asset_pairs.return_value = ASSET_PAIRS_ANTWORT

        with pytest.raises(RuntimeError, match="Kein einziges Paar"):
            pruefe_watchlist(api, ["QUATSCHEUR"])

    def test_netzwerkfehler_wird_als_runtimeerror_gemeldet(self):
        api = MagicMock()
        api.asset_pairs.side_effect = KrakenVerbindungsFehler("keine Verbindung")

        with pytest.raises(RuntimeError, match="AssetPairs"):
            pruefe_watchlist(api, ["XXBTZEUR"])


class TestRankePaare:
    def test_sortiert_nach_score_absteigend(self):
        bewertungen = {
            "A": _regime(score=0.2),
            "B": _regime(score=0.9),
            "C": _regime(score=0.5),
        }
        ranking = ranke_paare(bewertungen)
        assert [e.paar for e in ranking] == ["B", "C", "A"]

    def test_gate_bestandene_paare_stehen_vor_abgelehnten(self):
        bewertungen = {
            "ohne_gate_hoher_score": _regime(handel_erlaubt=False, score=0.99),
            "mit_gate_kleiner_score": _regime(handel_erlaubt=True, score=0.1),
        }
        ranking = ranke_paare(bewertungen)
        assert ranking[0].paar == "mit_gate_kleiner_score"


class TestVerteileKapital:
    def _config(self, max_parallel=2) -> AppConfig:
        return AppConfig(scanner=ScannerConfig(max_parallele_paare=max_parallel))

    def test_waehlt_nur_top_n_paare(self):
        ranking = ranke_paare({"A": _regime(score=0.9), "B": _regime(score=0.6), "C": _regime(score=0.3)})
        zuteilung = verteile_kapital(ranking, self._config(max_parallel=2), 100.0, paare_mit_offenen_lots=set())

        assert zuteilung.aktive_paare == ["A", "B"]
        assert "C" not in zuteilung.kapital_je_paar

    def test_kapital_wird_nach_score_gewichtet(self):
        ranking = ranke_paare({"A": _regime(score=0.75), "B": _regime(score=0.25)})
        zuteilung = verteile_kapital(ranking, self._config(), 100.0, paare_mit_offenen_lots=set())

        assert zuteilung.kapital_je_paar["A"] == pytest.approx(75.0)
        assert zuteilung.kapital_je_paar["B"] == pytest.approx(25.0)
        assert sum(zuteilung.kapital_je_paar.values()) == pytest.approx(100.0)

    def test_paare_ohne_gate_erhalten_kein_kapital(self):
        ranking = ranke_paare({"A": _regime(handel_erlaubt=False, score=0.9)})
        zuteilung = verteile_kapital(ranking, self._config(), 100.0, paare_mit_offenen_lots=set())

        assert zuteilung.aktive_paare == []
        assert zuteilung.kapital_je_paar == {}

    def test_rotation_laesst_offene_lots_nicht_verwaisen(self):
        """Ein Paar mit offenen Lots, das aus den Top-N faellt, muss in
        'nur verkaufen' wechseln statt vergessen zu werden."""
        ranking = ranke_paare({"A": _regime(score=0.9), "B": _regime(score=0.6), "ALT": _regime(score=0.1)})
        zuteilung = verteile_kapital(
            ranking, self._config(max_parallel=2), 100.0, paare_mit_offenen_lots={"ALT"}
        )

        assert zuteilung.aktive_paare == ["A", "B"]
        assert zuteilung.nur_verkaufen_paare == ["ALT"]

    def test_paar_das_gate_verliert_aber_lots_haelt_wird_nur_verkaufen(self):
        ranking = ranke_paare({"A": _regime(score=0.9), "VERLOREN": _regime(handel_erlaubt=False, score=0.0)})
        zuteilung = verteile_kapital(
            ranking, self._config(), 100.0, paare_mit_offenen_lots={"VERLOREN"}
        )

        assert "VERLOREN" not in zuteilung.aktive_paare
        assert zuteilung.nur_verkaufen_paare == ["VERLOREN"]

    def test_aktives_paar_steht_nicht_zugleich_in_nur_verkaufen(self):
        ranking = ranke_paare({"A": _regime(score=0.9)})
        zuteilung = verteile_kapital(ranking, self._config(), 100.0, paare_mit_offenen_lots={"A"})

        assert zuteilung.aktive_paare == ["A"]
        assert zuteilung.nur_verkaufen_paare == []


class TestFormatiereRanking:
    def test_leeres_ranking_liefert_hinweis(self):
        assert "Kein Paar" in formatiere_ranking([])

    def test_enthaelt_paarnamen_und_gate_status(self):
        ranking = ranke_paare({"XBTEUR": _regime(score=0.42, handel_erlaubt=True)})
        text = formatiere_ranking(ranking)
        assert "XBTEUR" in text
        assert "JA" in text
