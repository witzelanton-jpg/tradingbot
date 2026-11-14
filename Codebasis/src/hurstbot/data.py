"""OHLC-Daten laden, lokal cachen und ueber mehrere Zeitskalen bereitstellen.

Kraken liefert pro Abfrage maximal 720 Kerzen. Der lokale CSV-Cache haelt die
Historie laenger vor und wird bei jedem Aufruf nur um die neuen Kerzen (ueber
den `since`-Parameter) ergaenzt, um Rate-Limits zu schonen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import MultiSkalenConfig
from .kraken_api import KrakenAPI

logger = logging.getLogger("hurstbot.data")

OHLC_SPALTEN = ["zeit", "open", "high", "low", "close", "vwap", "volumen", "trades"]


def _parse_ohlc_antwort(ergebnis: dict) -> pd.DataFrame:
    """Wandelt die rohe Kraken-OHLC-Antwort in ein sauberes DataFrame um."""
    schluessel = next(k for k in ergebnis if k != "last")
    rohdaten = ergebnis[schluessel]
    if not rohdaten:
        return pd.DataFrame(columns=OHLC_SPALTEN).astype({"zeit": "datetime64[ns, UTC]"})
    df = pd.DataFrame(rohdaten, columns=OHLC_SPALTEN)
    df["zeit"] = pd.to_datetime(df["zeit"].astype("int64"), unit="s", utc=True)
    for spalte in ["open", "high", "low", "close", "vwap", "volumen"]:
        df[spalte] = df[spalte].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.drop_duplicates(subset="zeit").sort_values("zeit").reset_index(drop=True)
    return df


def cache_pfad(daten_verzeichnis: Path, paar: str, intervall_minuten: int) -> Path:
    daten_verzeichnis = Path(daten_verzeichnis)
    daten_verzeichnis.mkdir(parents=True, exist_ok=True)
    return daten_verzeichnis / f"{paar}_{intervall_minuten}m.csv"


def lade_ohlc_cache(pfad: Path) -> pd.DataFrame | None:
    if not pfad.exists():
        return None
    df = pd.read_csv(pfad)
    if df.empty:
        return None
    df["zeit"] = pd.to_datetime(df["zeit"], utc=True)
    return df


def speichere_ohlc_cache(df: pd.DataFrame, pfad: Path) -> None:
    """Atomares Schreiben: erst in Temp-Datei, dann umbenennen."""
    tmp = pfad.with_suffix(pfad.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(pfad)


def hole_ohlc(
    api: KrakenAPI,
    paar: str,
    intervall_minuten: int,
    daten_verzeichnis: Path,
    max_alter_sekunden: float | None = None,
    benutze_cache: bool = True,
) -> pd.DataFrame:
    """Holt OHLC-Daten fuer ein Paar/Intervall, kombiniert mit lokalem Cache."""
    if max_alter_sekunden is None:
        max_alter_sekunden = max(intervall_minuten * 60 / 2, 30.0)

    pfad = cache_pfad(daten_verzeichnis, paar, intervall_minuten)
    cache_df = lade_ohlc_cache(pfad) if benutze_cache else None

    since = None
    if cache_df is not None and not cache_df.empty:
        letzte_zeit = cache_df["zeit"].max()
        alter_sekunden = (pd.Timestamp.now(tz="UTC") - letzte_zeit).total_seconds()
        if alter_sekunden < max_alter_sekunden:
            return cache_df
        since = int(letzte_zeit.timestamp())

    ergebnis = api.ohlc(paar, intervall_minuten, since=since)
    neue_daten = _parse_ohlc_antwort(ergebnis)

    if cache_df is not None and not cache_df.empty and not neue_daten.empty:
        kombiniert = pd.concat([cache_df, neue_daten], ignore_index=True)
        kombiniert = kombiniert.drop_duplicates(subset="zeit").sort_values("zeit").reset_index(drop=True)
    elif cache_df is not None and not cache_df.empty:
        kombiniert = cache_df
    else:
        kombiniert = neue_daten

    if benutze_cache and not kombiniert.empty:
        speichere_ohlc_cache(kombiniert, pfad)
    return kombiniert


def hole_multiskalen_ohlc(
    api: KrakenAPI,
    paar: str,
    multiskalen_config: MultiSkalenConfig,
    daten_verzeichnis: Path,
) -> dict[str, pd.DataFrame]:
    """Laedt kurz-, mittel- und langfristige OHLC-Reihen fuer die Hurst-Analyse."""
    kurz = hole_ohlc(api, paar, multiskalen_config.kurz_intervall_minuten, daten_verzeichnis)
    kurz = kurz.tail(multiskalen_config.kurz_anzahl_kerzen).reset_index(drop=True)

    mittel = hole_ohlc(api, paar, multiskalen_config.mittel_intervall_minuten, daten_verzeichnis)
    mittel = mittel.tail(multiskalen_config.mittel_anzahl_kerzen).reset_index(drop=True)

    try:
        lang = hole_ohlc(api, paar, multiskalen_config.lang_intervall_minuten, daten_verzeichnis)
        lang = lang.tail(multiskalen_config.lang_anzahl_kerzen).reset_index(drop=True)
    except Exception as exc:  # Langzeit-Skala ist ein Zusatzscore, kein Hartausschluss
        logger.warning("Langzeit-OHLC fuer %s nicht verfuegbar, fahre ohne fort: %s", paar, exc)
        lang = pd.DataFrame(columns=OHLC_SPALTEN)

    return {"kurz": kurz, "mittel": mittel, "lang": lang}
