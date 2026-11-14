"""Wertet die Laeufe unter laeufe/ aus und schreibt auswertung/zusammenfassung.{csv,md}.

Quellen je Szenario (alles vom Bot selbst geschrieben, ausser zyklen.csv/lauf.json):
  logs/trade_log.csv        Kaeufe/Verkaeufe (Bot)
  logs/zustands_log.jsonl   Zustandswechsel mit Grund (Bot)
  state/bot_zustand.json    Endzustand: Kapital, offene Lots, X (Bot)
  data/SINUSEUR_*.csv       OHLC-Cache des Bots - wird mit den wahren Kerzen verglichen
  zyklen.csv                je Zyklus: was der Bot gesehen/entschieden hat (sim_run.py)
  konsole.log               Konsolenausgabe (fuer den CLI-Run: Ranking je Zyklus)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kursmodell import SynthetischerMarkt  # noqa: E402

TESTORDNER = Path(__file__).resolve().parent.parent
START_KAPITAL = 100.0

RANKING_ZEILE = re.compile(
    r"^SINUSEUR\s+(?P<score>[\d.]+)\s+(?P<hk>-?[\d.]+|n/a)\s+(?P<hm>-?[\d.]+|n/a)\s+(?P<konf>[\d.]+|n/a)\s+(?P<gate>JA|NEIN)\s+(?P<grund>.*)$"
)


def trades_auswerten(pfad: Path, bis: pd.Timestamp | None = None) -> dict:
    if not pfad.exists():
        return {"kaeufe": 0, "verkaeufe": 0, "realisiert_eur": 0.0, "gebuehren_eur": 0.0, "trades": pd.DataFrame()}
    t = pd.read_csv(pfad)
    t["zeitpunkt"] = pd.to_datetime(t["zeitpunkt"], utc=True, format="ISO8601")
    if bis is not None:  # abgebrochener Lauf: nur bis zum letzten mitgeschriebenen Zyklus
        t = t[t["zeitpunkt"] <= bis]
    v = t[t["art"] == "VERKAUF"]
    realisiert = float((v["menge"] * v["preis_eur"] - v["gebuehr_eur"] - v["menge"] * v["lot_einstandspreis"]).sum())
    return {
        "kaeufe": int((t["art"] == "KAUF").sum()),
        "verkaeufe": int(len(v)),
        "realisiert_eur": realisiert,
        "gebuehren_eur": float(t["gebuehr_eur"].sum()),
        "trades": t,
    }


def zustandswechsel(pfad: Path) -> pd.DataFrame:
    if not pfad.exists():
        return pd.DataFrame(columns=["zeitpunkt", "alter_zustand", "neuer_zustand", "grund"])
    zeilen = [json.loads(z) for z in pfad.read_text(encoding="utf-8").splitlines() if z.strip()]
    return pd.DataFrame(zeilen)


def cache_gegen_wahrheit(ordner: Path, markt: SynthetischerMarkt, intervall: int, von: pd.Timestamp, bis: pd.Timestamp) -> dict | None:
    """Vergleicht den OHLC-Cache des Bots mit den wahren, vollstaendigen Kerzen im Laufzeitraum."""
    pfad = ordner / "data" / f"SINUSEUR_{intervall}m.csv"
    if not pfad.exists():
        return None
    cache = pd.read_csv(pfad)
    cache["zeit"] = pd.to_datetime(cache["zeit"], utc=True)
    wahr = markt.wahre_kerzen(intervall, von.timestamp(), bis.timestamp())
    m = cache.merge(wahr, on="zeit", suffixes=("_cache", "_wahr"))
    if m.empty:
        return None
    abw = (m["close_cache"] - m["close_wahr"]).abs()
    spanne_cache = (m["high_cache"] - m["low_cache"]).mean()
    spanne_wahr = (m["high_wahr"] - m["low_wahr"]).mean()
    return {
        "kerzen_verglichen": int(len(m)),
        "anteil_close_falsch": float((abw > 1e-4).mean()),
        "mittlere_close_abweichung_eur": float(abw.mean()),
        "spanne_cache_zu_wahr": float(spanne_cache / spanne_wahr) if spanne_wahr > 0 else None,
    }


def max_drawdown(werte: pd.Series) -> float:
    if werte.empty:
        return 0.0
    hoch = werte.cummax()
    return float(((hoch - werte) / hoch).max() * 100)


def ranking_aus_konsole(pfad: Path) -> pd.DataFrame:
    zeilen = []
    for z in pfad.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RANKING_ZEILE.match(z.strip())
        if m:
            d = m.groupdict()
            zeilen.append({
                "score": float(d["score"]),
                "h_kurz": None if d["hk"] == "n/a" else float(d["hk"]),
                "h_mittel": None if d["hm"] == "n/a" else float(d["hm"]),
                "gate": d["gate"] == "JA",
                "grund": d["grund"],
            })
    return pd.DataFrame(zeilen)


def werte_aus(ordner: Path) -> dict:
    info = json.loads((ordner / "szenario.json").read_text(encoding="utf-8"))
    markt = SynthetischerMarkt.lade(ordner / "markt.npz")
    lauf = json.loads((ordner / "lauf.json").read_text(encoding="utf-8")) if (ordner / "lauf.json").exists() else {}
    zustand = json.loads((ordner / "state" / "bot_zustand.json").read_text(encoding="utf-8"))
    paar = zustand["paare"].get(info["paar"], {})
    zw = zustandswechsel(ordner / "logs" / "zustands_log.jsonl")

    zyklen_pfad = ordner / "zyklen.csv"
    z = None
    if zyklen_pfad.exists():
        z = pd.read_csv(zyklen_pfad)
        z["zeit"] = pd.to_datetime(z["zeit"], utc=True)
    abgebrochen = z is not None and not (ordner / "lauf.json").exists()
    tr = trades_auswerten(ordner / "logs" / "trade_log.csv", bis=z["zeit"].iloc[-1] if abgebrochen else None)
    if abgebrochen and not zw.empty:
        zw = zw[pd.to_datetime(zw["zeitpunkt"], utc=True) <= z["zeit"].iloc[-1]]

    if z is not None:
        von, bis = z["zeit"].iloc[0], z["zeit"].iloc[-1]
        end_preis = float(z["preis"].iloc[-1])
        gate = z["gate"].astype(bool)
        aktiv = gate & (z["score"] > 0)
        h_kurz, h_mittel, ci_kurz, ci_mittel = z["h_kurz"], z["h_mittel"], z["ci_kurz_oben"], z["ci_mittel_oben"]
        dd = max_drawdown(z["wert_marktwert"])
        tiefster_wert = float(z["wert_marktwert"].min())
        x_verlauf = z["x_prozent"].round(4).drop_duplicates().tolist()
        profile = z["profil"].drop_duplicates().tolist()
        zyklen_anzahl = len(z)
        rechenzeit = lauf.get("rechenzeit_s")
    else:  # CLI-Run: Ranking aus der Konsolenausgabe, Zeiten aus Logs/Dateien
        r = ranking_aus_konsole(ordner / "konsole.log")
        von = pd.Timestamp(info["laufbeginn"])
        bis = pd.Timestamp((ordner / "state" / "bot_zustand.json").stat().st_mtime, unit="s", tz="UTC")
        end_preis = markt.preis(bis.timestamp())
        gate = r["gate"] if not r.empty else pd.Series(dtype=bool)
        aktiv = gate & (r["score"] > 0)
        h_kurz, h_mittel = r["h_kurz"], r["h_mittel"]
        ci_kurz = ci_mittel = pd.Series(dtype=float)
        dd, tiefster_wert = None, None
        x_verlauf = [paar.get("x_prozent")]
        profile = [paar.get("zeitskala_profil_name")]
        zyklen_anzahl = len(r)
        rechenzeit = (bis - von).total_seconds()

    if abgebrochen:  # Endstand = letzter mitgeschriebener Zyklus (Zustandsdatei ist juenger)
        letzte = z.iloc[-1]
        anzahl_lots = int(letzte["lots"])
        marktwert = float(letzte["wert_marktwert"])
        buchwert = float(letzte["wert_bot_buchwert"])
        unrealisiert = marktwert - buchwert  # Lots zum Marktpreis gegen Kaufpreis (ohne Kaufgebuehr)
        x_verlauf = z["x_prozent"].round(4).drop_duplicates().tolist()
    else:
        lots = paar.get("lots", [])
        anzahl_lots = len(lots)
        lots_menge = sum(l["menge"] for l in lots)
        marktwert = zustand["kapital_frei_eur"] + paar.get("kapital_eur", 0.0) + lots_menge * end_preis
        buchwert = zustand["kapital_frei_eur"] + paar.get("kapital_eur", 0.0) + sum(l["menge"] * l["fill_preis_kauf"] for l in lots)
        unrealisiert = sum(l["menge"] * (end_preis - l["einstandspreis"]) for l in lots)

    neue_zustaende = zw["neuer_zustand"].value_counts().to_dict() if not zw.empty else {}
    pausen_gruende = zw.loc[zw["neuer_zustand"] == "PAUSIERT", "grund"].str.replace(r"[\d.]+", "#", regex=True).value_counts().to_dict() if not zw.empty else {}

    ergebnis = {
        "szenario": info["name"],
        "beschreibung": info.get("beschreibung", ""),
        "periode_min": info["periode_minuten"],
        "amplitude_eur": info["modell"]["amplitude"],
        "stoerungen": "; ".join(f"{s['art']} {s['staerke']:g}" + (f"@{s['ab_stunden']:g}h" if s["art"] in ("trend", "sprung") else "") for s in info["modell"]["stoerungen"]) or "keine",
        "zyklen": zyklen_anzahl,
        "sim_von": str(von), "sim_bis": str(bis),
        "rechenzeit_s": rechenzeit,
        "gate_offen_anteil": float(gate.mean()) if len(gate) else None,
        "aktiv_anteil": float(aktiv.mean()) if len(aktiv) else None,
        "h_kurz_median": float(h_kurz.median()) if len(h_kurz) else None,
        "h_mittel_median": float(h_mittel.median()) if len(h_mittel) else None,
        "ci_kurz_oben_median": float(ci_kurz.median()) if len(ci_kurz) else None,
        "ci_mittel_oben_median": float(ci_mittel.median()) if len(ci_mittel) else None,
        "kaeufe": tr["kaeufe"], "verkaeufe": tr["verkaeufe"],
        "realisiert_eur": round(tr["realisiert_eur"], 4),
        "gebuehren_eur": round(tr["gebuehren_eur"], 4),
        "abgebrochen": abgebrochen,
        "offene_lots": anzahl_lots,
        "unrealisiert_eur": round(unrealisiert, 4),
        "endwert_markt_eur": round(marktwert, 4),
        "netto_markt_eur": round(marktwert - START_KAPITAL, 4),
        "endwert_buch_eur": round(buchwert, 4),
        "max_drawdown_markt_prozent": None if dd is None else round(dd, 3),
        "tiefster_marktwert_eur": tiefster_wert,
        "x_verlauf": x_verlauf,
        "profile": profile,
        "zustandswechsel": neue_zustaende,
        "pausen_gruende": pausen_gruende,
        "global_zustand": zustand.get("global_zustand"),
        "kurs_start": float(markt.preis(von.timestamp())), "kurs_ende": end_preis,
    }
    for k in (5, 60):
        ergebnis[f"cache_{k}m"] = cache_gegen_wahrheit(ordner, markt, k, von, bis)
    return ergebnis


def main() -> None:
    namen = sys.argv[1:] or sorted(p.name for p in (TESTORDNER / "laeufe").iterdir() if (p / "state" / "bot_zustand.json").exists())
    ergebnisse = []
    for name in namen:
        try:
            ergebnisse.append(werte_aus(TESTORDNER / "laeufe" / name))
        except Exception as exc:  # ein unfertiger Lauf soll die anderen nicht blockieren
            print(f"{name}: nicht auswertbar ({exc})")
    ziel = TESTORDNER / "auswertung"
    ziel.mkdir(exist_ok=True)
    (ziel / "ergebnisse.json").write_text(json.dumps(ergebnisse, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    flach = pd.DataFrame([{k: v for k, v in e.items() if not isinstance(v, (dict, list))} for e in ergebnisse])
    flach.to_csv(ziel / "zusammenfassung.csv", index=False)
    spalten = ["szenario", "zyklen", "gate_offen_anteil", "aktiv_anteil", "h_kurz_median", "h_mittel_median",
               "kaeufe", "verkaeufe", "realisiert_eur", "offene_lots", "unrealisiert_eur", "netto_markt_eur",
               "max_drawdown_markt_prozent"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(flach[[s for s in spalten if s in flach]].to_string(index=False))
    for e in ergebnisse:
        print(f"\n[{e['szenario']}] X: {e['x_verlauf']} | Profile: {e['profile']} | Zustaende: {e['zustandswechsel']}")
        if e["pausen_gruende"]:
            print(f"    Pausen: {e['pausen_gruende']}")
        print(f"    Cache 5m: {e['cache_5m']} | Cache 60m: {e['cache_60m']}")


if __name__ == "__main__":
    main()
