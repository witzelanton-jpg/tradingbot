"""Tests fuer Gate-Hysterese, Oszillations-Score und Y-Betragsberechnung."""

from __future__ import annotations

import pytest

from hurstbot.config import GateConfig, YConfig
from hurstbot.hurst import HurstErgebnis
from hurstbot.regime import SkalenErgebnis, aktualisiere_gate, berechne_oszillations_score, berechne_y_betrag

GATE = GateConfig(h_start=0.45, h_stop=0.50)


def _skala(h: float, ci_oben: float, konfidenz: float = 0.8, monotonie: float = 0.1) -> SkalenErgebnis:
    hurst = HurstErgebnis(h=h, ci_untere_grenze=h - 0.05, ci_obere_grenze=ci_oben, konfidenz=konfidenz)
    return SkalenErgebnis(hurst=hurst, monotonie={"monotonie_score": monotonie})


class TestGateHysterese:
    def test_handel_startet_unter_h_start(self):
        kurz = _skala(0.40, 0.44)
        mittel = _skala(0.42, 0.46)
        erlaubt, grund = aktualisiere_gate(False, kurz, mittel, GATE)
        assert erlaubt is True

    def test_absolute_sperre_bei_h_ueber_0_5(self):
        kurz = _skala(0.55, 0.60)
        mittel = _skala(0.40, 0.44)
        erlaubt, grund = aktualisiere_gate(True, kurz, mittel, GATE)
        assert erlaubt is False
        assert "0.5" in grund

    def test_absolute_sperre_bei_oberer_ci_ueber_0_5(self):
        # H selbst ist < 0.5, aber die obere Konfidenzgrenze erreicht 0.5 -> trotzdem sperren.
        kurz = _skala(0.42, 0.51)
        mittel = _skala(0.40, 0.44)
        erlaubt, grund = aktualisiere_gate(True, kurz, mittel, GATE)
        assert erlaubt is False

    def test_hysterese_haelt_laufenden_handel(self):
        # H liegt in der Zone [h_start, h_stop) -> vorheriger Zustand bleibt erhalten.
        kurz = _skala(0.47, 0.49)
        mittel = _skala(0.46, 0.48)
        erlaubt, _ = aktualisiere_gate(True, kurz, mittel, GATE)
        assert erlaubt is True

    def test_hysterese_haelt_pause(self):
        kurz = _skala(0.47, 0.49)
        mittel = _skala(0.46, 0.48)
        erlaubt, _ = aktualisiere_gate(False, kurz, mittel, GATE)
        assert erlaubt is False

    def test_stop_bei_h_ueber_h_stop_unabhaengig_von_vorherigem_zustand(self):
        kurz = _skala(0.52, 0.56)
        mittel = _skala(0.51, 0.55)
        erlaubt, _ = aktualisiere_gate(True, kurz, mittel, GATE)
        assert erlaubt is False

    def test_schlechtere_von_kurz_und_mittel_entscheidet(self):
        # Kurzfristig gut (< h_start), aber mittelfristig schlecht (>= h_stop) -> kein Handel.
        kurz = _skala(0.30, 0.35)
        mittel = _skala(0.55, 0.58)
        erlaubt, _ = aktualisiere_gate(True, kurz, mittel, GATE)
        assert erlaubt is False


class TestOszillationsScore:
    def test_score_liegt_immer_in_0_1(self):
        for h in [0.0, 0.1, 0.3, 0.45, 0.49, 0.5, 0.6]:
            for konfidenz in [0.0, 0.5, 1.0]:
                for monotonie in [0.0, 0.5, 1.0]:
                    score = berechne_oszillations_score(h, GATE, konfidenz, monotonie, skalen_abweichung=0.1, langfrist_faktor=1.0)
                    assert 0.0 <= score <= 1.0

    def test_score_ist_null_ab_h_stop(self):
        assert berechne_oszillations_score(0.50, GATE, 1.0, 0.0, 0.0, 1.0) == 0.0
        assert berechne_oszillations_score(0.60, GATE, 1.0, 0.0, 0.0, 1.0) == 0.0

    def test_kleineres_h_erhoeht_score(self):
        score_hoch = berechne_oszillations_score(0.40, GATE, 0.8, 0.1, 0.0, 1.0)
        score_niedrig = berechne_oszillations_score(0.10, GATE, 0.8, 0.1, 0.0, 1.0)
        assert score_niedrig > score_hoch

    def test_hohe_monotonie_senkt_score(self):
        score_ohne_monotonie = berechne_oszillations_score(0.2, GATE, 0.8, 0.0, 0.0, 1.0)
        score_mit_monotonie = berechne_oszillations_score(0.2, GATE, 0.8, 0.9, 0.0, 1.0)
        assert score_mit_monotonie < score_ohne_monotonie


class TestBerechneYBetrag:
    Y_CONFIG = YConfig(y_min_eur=5.0, y_max_eur=50.0)

    def test_hoher_score_und_konfidenz_naehert_sich_y_max(self):
        y = berechne_y_betrag(1.0, 1.0, self.Y_CONFIG, verfuegbares_kapital_eur=1000, exposure_limit_eur=1000, mindest_betrag_eur=0.0)
        assert y == pytest.approx(50.0)

    def test_niedriger_score_bleibt_nahe_y_min(self):
        y = berechne_y_betrag(0.0, 0.0, self.Y_CONFIG, verfuegbares_kapital_eur=1000, exposure_limit_eur=1000, mindest_betrag_eur=0.0)
        assert y == pytest.approx(5.0)

    def test_y_immer_in_bereich_wenn_nicht_extern_begrenzt(self):
        for score in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for konfidenz in [0.0, 0.3, 0.7, 1.0]:
                y = berechne_y_betrag(score, konfidenz, self.Y_CONFIG, verfuegbares_kapital_eur=1000, exposure_limit_eur=1000, mindest_betrag_eur=0.0)
                assert 5.0 <= y <= 50.0

    def test_kapitalgrenze_wird_respektiert(self):
        y = berechne_y_betrag(1.0, 1.0, self.Y_CONFIG, verfuegbares_kapital_eur=3.0, exposure_limit_eur=1000, mindest_betrag_eur=0.0)
        assert y == pytest.approx(3.0)

    def test_exposure_limit_wird_respektiert(self):
        y = berechne_y_betrag(1.0, 1.0, self.Y_CONFIG, verfuegbares_kapital_eur=1000, exposure_limit_eur=10.0, mindest_betrag_eur=0.0)
        assert y == pytest.approx(10.0)

    def test_unter_mindestordergroesse_liefert_null(self):
        y = berechne_y_betrag(0.0, 0.0, self.Y_CONFIG, verfuegbares_kapital_eur=1000, exposure_limit_eur=1000, mindest_betrag_eur=6.0)
        assert y == 0.0
