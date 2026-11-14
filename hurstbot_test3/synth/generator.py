"""Erzeugt idealisierte Kursdaten fuer den Hurstbot und legt dafuer einen Szenario-Ordner an.

    y(t) = P0 + A * sin(2*pi*f*t + phi) + Stoerung(t)

Drei Arten der Eingabe:

  1) Kommandozeile
     python synth/generator.py --name mein_test --amplitude 1.4 --frequenz 40 \\
            --stoerung weiss=0.2 --stoerung trend=-2@0 --stunden 12

  2) Vordefinierte Szenarien aus szenarien.yaml
     python synth/generator.py --szenarien szenarien.yaml            (alle)
     python synth/generator.py --szenarien szenarien.yaml --nur F1    (einzelne)

  3) Interaktiv (ohne Argumente): fragt Amplitude, Frequenz und Stoerungen ab.

Ergebnis je Szenario: laeufe/<name>/ mit
  markt.npz      Minutenreihe (Kurs + Basispreis ohne Sinus)
  szenario.json  Parameter, Laufbeginn/-ende, Formel
  config.yaml    Bot-Konfiguration (Kopie von ../config.yaml, ggf. mit Abweichungen)
  data/ logs/ state/  leer - hier schreibt der Bot

Mit --echtzeit liegt der Laufbeginn auf "jetzt" (fuer den echten CLI-Run gegen den
lokalen Server, siehe fake_kraken.py), sonst auf dem in szenarien.yaml bzw. per
--laufbeginn angegebenen Zeitpunkt in der Vergangenheit.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kursmodell import ARTEN, KursModell, Stoerung, SynthetischerMarkt, erzeuge_minutenreihe, modell_als_dict, modell_aus_dict  # noqa: E402

TESTORDNER = Path(__file__).resolve().parent.parent
STANDARD_LAUFBEGINN = "2026-11-05T00:00:00Z"
STANDARD_PAAR = "SINUSEUR"


def _config_fuer_szenario(ziel: Path, config_abweichungen: dict, port: int | None) -> None:
    """Kopiert die Basis-Config und setzt einzelne Top-Level-Werte bzw. den Server-Port."""
    text = (TESTORDNER / "config.yaml").read_text(encoding="utf-8")
    for schluessel, wert in (config_abweichungen or {}).items():
        muster = re.compile(rf"^{re.escape(schluessel)}:.*$", re.MULTILINE)
        if not muster.search(text):
            raise ValueError(f"config.yaml hat keinen Top-Level-Schluessel '{schluessel}'")
        text = muster.sub(f"{schluessel}: {wert}   # gesetzt vom Generator (Szenario)", text, count=1)
    if port is not None:
        text = text.replace("http://127.0.0.1:8765", f"http://127.0.0.1:{port}")
    ziel.write_text(text, encoding="utf-8")


def erzeuge_szenario(
    name: str,
    modell: KursModell,
    lauf_stunden: float,
    laufbeginn: pd.Timestamp,
    beschreibung: str = "",
    historie_tage: int = 400,
    paar: str = STANDARD_PAAR,
    config_abweichungen: dict | None = None,
    port: int | None = None,
    ueberschreiben: bool = False,
) -> Path:
    ordner = TESTORDNER / "laeufe" / name
    if ordner.exists():
        if not ueberschreiben:
            raise SystemExit(f"{ordner} existiert schon - mit --ueberschreiben neu anlegen")
        shutil.rmtree(ordner)
    for unter in ("data", "logs", "state"):
        (ordner / unter).mkdir(parents=True, exist_ok=True)

    t0, i_lauf, kurs, basis = erzeuge_minutenreihe(modell, laufbeginn, historie_tage, lauf_stunden)
    markt = SynthetischerMarkt(t0, kurs, basis)
    markt.speichere(ordner / "markt.npz")
    _config_fuer_szenario(ordner / "config.yaml", config_abweichungen or {}, port)

    lauf = kurs[i_lauf : i_lauf + int(lauf_stunden * 60)]
    laufbeginn_utc = pd.Timestamp(laufbeginn).tz_convert("UTC").floor("min")
    info = {
        "name": name,
        "beschreibung": beschreibung,
        "paar": paar,
        "modell": modell_als_dict(modell),
        "formel": modell.formel(),
        "periode_minuten": modell.periode_minuten,
        "amplitude_prozent_von_p0": 100 * modell.amplitude / modell.basispreis,
        "laufbeginn": laufbeginn_utc.isoformat(),
        "laufende": (laufbeginn_utc + pd.Timedelta(hours=lauf_stunden)).isoformat(),
        "lauf_stunden": lauf_stunden,
        "historie_tage": historie_tage,
        "historie_start": pd.Timestamp(t0, unit="s", tz="UTC").isoformat(),
        "config_abweichungen": config_abweichungen or {},
        "kurs_im_lauf": {"min": float(lauf.min()), "max": float(lauf.max()), "start": float(lauf[0]), "ende": float(lauf[-1])},
        "kurs_gesamt": {"min": float(kurs.min()), "max": float(kurs.max())},
    }
    (ordner / "szenario.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{name}] {modell.formel()}")
    print(f"    Periode {modell.periode_minuten:.1f} min, Amplitude {info['amplitude_prozent_von_p0']:.2f} % von P0, "
          f"Lauf {lauf_stunden:g} h ab {laufbeginn_utc:%Y-%m-%d %H:%M} UTC, Historie {historie_tage} Tage")
    print(f"    Kurs im Lauf: {lauf.min():.2f} .. {lauf.max():.2f} EUR (Start {lauf[0]:.2f}, Ende {lauf[-1]:.2f})")
    print(f"    -> {ordner}")
    return ordner


def _frage(text: str, standard: str) -> str:
    antwort = input(f"{text} [{standard}]: ").strip()
    return antwort or standard


def interaktiv() -> None:
    print("Idealisierte Kursdaten: y(t) = P0 + A*sin(2*pi*f*t) + Stoerung(t)")
    name = _frage("Name des Szenarios", "eigenes_szenario")
    p0 = float(_frage("Basispreis P0 in EUR", "100"))
    a = float(_frage("Amplitude A in EUR", "1.4"))
    f = float(_frage("Frequenz f in Schwingungen pro Tag (40 = Periode 36 min)", "40"))
    print("Stoerungen als art=staerke[@ab_stunden], mehrere mit Komma. Arten:")
    for art, einheit in ARTEN.items():
        print(f"  {art:<12} {einheit}")
    stoer_text = _frage("Stoerungen (leer = keine)", "weiss=0.2")
    stoerungen = [Stoerung.aus_text(s) for s in stoer_text.split(",") if s.strip()]
    stunden = float(_frage("Laufdauer in Stunden", "12"))
    modell = KursModell(basispreis=p0, amplitude=a, frequenz_pro_tag=f, stoerungen=stoerungen)
    erzeuge_szenario(name, modell, stunden, pd.Timestamp(STANDARD_LAUFBEGINN))


def main() -> None:
    parser = argparse.ArgumentParser(description="Idealisierte Kursdaten fuer den Hurstbot erzeugen")
    parser.add_argument("--szenarien", type=Path, help="YAML-Datei mit vordefinierten Szenarien")
    parser.add_argument("--nur", nargs="*", help="nur diese Szenarien aus der YAML-Datei")
    parser.add_argument("--name", help="Name des Szenarios (Ordner unter laeufe/)")
    parser.add_argument("--basispreis", type=float, default=100.0, help="P0 in EUR (Default 100)")
    parser.add_argument("--amplitude", type=float, help="A in EUR")
    parser.add_argument("--frequenz", type=float, help="f in Schwingungen pro Tag (40 = Periode 36 min)")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase in Grad")
    parser.add_argument("--stoerung", action="append", default=[], help="art=staerke[@ab_stunden], mehrfach moeglich")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stunden", type=float, default=12.0, help="Laufdauer in Stunden")
    parser.add_argument("--laufbeginn", default=STANDARD_LAUFBEGINN, help="Laufbeginn (UTC, ISO-Format)")
    parser.add_argument("--echtzeit", action="store_true", help="Laufbeginn = jetzt (fuer den Server/CLI-Run)")
    parser.add_argument("--port", type=int, help="Port des lokalen Servers (setzt basis_url in config.yaml)")
    parser.add_argument("--abfrage-sekunden", type=int, help="abfrage_sekunden in der Szenario-Config")
    parser.add_argument("--ueberschreiben", action="store_true")
    args = parser.parse_args()

    if args.szenarien:
        daten = yaml.safe_load(args.szenarien.read_text(encoding="utf-8"))
        standard = daten.get("standard", {})
        for name, s in daten["szenarien"].items():
            if args.nur and name not in args.nur:
                continue
            modell = modell_aus_dict({k: s[k] for k in s if k in {"basispreis", "amplitude", "frequenz_pro_tag", "phase_grad", "stoerungen", "seed"}}
                                     | {k: v for k, v in standard.get("modell", {}).items() if k not in s})
            beginn = pd.Timestamp.now(tz="UTC") if args.echtzeit else pd.Timestamp(s.get("laufbeginn", standard.get("laufbeginn", STANDARD_LAUFBEGINN)))
            erzeuge_szenario(
                name, modell, float(s.get("lauf_stunden", 12)), beginn,
                beschreibung=s.get("beschreibung", ""), historie_tage=int(standard.get("historie_tage", 400)),
                paar=standard.get("paar", STANDARD_PAAR), config_abweichungen=s.get("config", {}),
                port=args.port, ueberschreiben=args.ueberschreiben,
            )
        return

    if args.amplitude is None or args.frequenz is None or not args.name:
        interaktiv()
        return

    modell = KursModell(
        basispreis=args.basispreis, amplitude=args.amplitude, frequenz_pro_tag=args.frequenz,
        phase_grad=args.phase, stoerungen=[Stoerung.aus_text(s) for s in args.stoerung], seed=args.seed,
    )
    beginn = pd.Timestamp.now(tz="UTC") if args.echtzeit else pd.Timestamp(args.laufbeginn)
    abweichungen = {"abfrage_sekunden": args.abfrage_sekunden} if args.abfrage_sekunden else {}
    erzeuge_szenario(args.name, modell, args.stunden, beginn, config_abweichungen=abweichungen,
                     port=args.port, ueberschreiben=args.ueberschreiben)


if __name__ == "__main__":
    main()
