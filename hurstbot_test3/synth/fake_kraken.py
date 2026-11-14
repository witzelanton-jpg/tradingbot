"""Synthetische Kraken-Schnittstelle fuer hurstbot_test3.

Zwei Wege, ueber die der unveraenderte Bot an die idealisierten Kurse kommt:

1. In-Prozess (`SynthetischeKrakenAPI`): dieselben oeffentlichen Methoden wie
   `hurstbot.kraken_api.KrakenAPI` (asset_pairs, ohlc, ticker) mit denselben
   Antwortformaten. Die Zeit kommt von einer Uhr, die der Simulationstreiber
   (sim_run.py) weiterstellt.

2. HTTP-Server (`python synth/fake_kraken.py serve <ordner>`): beantwortet
   /0/public/AssetPairs, /0/public/OHLC und /0/public/Ticker auf localhost in
   ECHTZEIT. Mit `kraken_api.basis_url: http://127.0.0.1:<port>` in der
   config.yaml laeuft dann der echte Befehl `hurstbot --config config.yaml run`
   gegen die synthetischen Kurse - inklusive HTTP, Rate-Limit und Retry-Logik
   des Bots.

Private Endpunkte gibt es bewusst nicht: Paper braucht keine, und ein Aufruf
waere ein Fehler, der auffallen soll.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from hurstbot.kraken_api import KrakenAPIError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kursmodell import SynthetischerMarkt  # noqa: E402

# Handelsregeln des synthetischen Paares (aehnlich einem liquiden Kraken-EUR-Paar)
PAAR_REGELN = {"pair_decimals": 2, "lot_decimals": 8, "ordermin": "0.01", "cost_decimals": 5}


class SimUhr:
    """Simulierte Uhr: wird vom Treiber weitergestellt."""

    def __init__(self, start: pd.Timestamp) -> None:
        self.jetzt = pd.Timestamp(start)

    def sekunden(self) -> float:
        return self.jetzt.timestamp()


class EchtzeitUhr:
    def sekunden(self) -> float:
        return time.time()


class SynthetischeKrakenAPI:
    """Ersetzt `KrakenAPI` fuer die oeffentlichen Endpunkte, mit synthetischen Maerkten."""

    def __init__(self, maerkte: dict[str, SynthetischerMarkt], uhr) -> None:
        self.maerkte = maerkte
        self.uhr = uhr
        self.anfragen: dict[str, int] = {"AssetPairs": 0, "OHLC": 0, "Ticker": 0}
        self._sperre = threading.Lock()

    def _zaehle(self, methode: str) -> None:
        with self._sperre:
            self.anfragen[methode] = self.anfragen.get(methode, 0) + 1

    def _markt(self, paar: str) -> SynthetischerMarkt:
        if paar not in self.maerkte:
            raise KrakenAPIError(f"Kraken-Fehler: ['EQuery:Unknown asset pair'] ({paar})")
        return self.maerkte[paar]

    # ---------- oeffentliche Endpunkte (gleiche Signaturen wie KrakenAPI) ----------

    def asset_pairs(self, paare: list[str] | None = None) -> dict:
        self._zaehle("AssetPairs")
        namen = paare or list(self.maerkte)
        return {
            p: {"altname": p, "wsname": p, "base": p[:-3], "quote": p[-3:], **PAAR_REGELN}
            for p in namen
            if p in self.maerkte
        }

    def ohlc(self, paar: str, intervall_minuten: int, since: int | None = None) -> dict:
        self._zaehle("OHLC")
        markt = self._markt(paar)
        zeilen, last = markt.kerzen(int(intervall_minuten), self.uhr.sekunden(), since_s=since)
        return {paar: zeilen, "last": last}

    def ticker(self, paare: list[str]) -> dict:
        self._zaehle("Ticker")
        antwort = {}
        for p in paare:
            preis = self._markt(p).preis(self.uhr.sekunden())
            antwort[p] = {
                "a": [f"{preis * 1.00025:.5f}", "1", "1.000"],
                "b": [f"{preis * 0.99975:.5f}", "1", "1.000"],
                "c": [f"{preis:.5f}", "0.10000000"],
            }
        return antwort

    # ---------- private Endpunkte: im Paper-Test nie benutzt ----------

    def _privat(self, *_a, **_k) -> dict:
        raise KrakenAPIError("Synthetische API: private Endpunkte gibt es im Paper-Test nicht")

    balance = add_order = open_orders = cancel_order = _privat


# ---------- HTTP-Server fuer den echten CLI-Run ----------


def _baue_handler(api: SynthetischeKrakenAPI, log_pfad: Path):
    log_sperre = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # Standard-Log nach stderr unterdruecken
            pass

        def _antworte(self, status: int, nutzlast: dict) -> None:
            body = json.dumps(nutzlast).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            methode = url.path.rsplit("/", 1)[-1]
            try:
                if url.path.startswith("/0/public/AssetPairs"):
                    paare = params["pair"].split(",") if "pair" in params else None
                    ergebnis = api.asset_pairs(paare)
                elif url.path.startswith("/0/public/OHLC"):
                    since = int(params["since"]) if "since" in params else None
                    ergebnis = api.ohlc(params["pair"], int(params.get("interval", 1)), since=since)
                elif url.path.startswith("/0/public/Ticker"):
                    ergebnis = api.ticker(params["pair"].split(","))
                else:
                    self._antworte(404, {"error": [f"EGeneral:Unknown method {url.path}"], "result": {}})
                    return
                self._antworte(200, {"error": [], "result": ergebnis})
                status = "ok"
            except KrakenAPIError as exc:
                self._antworte(200, {"error": [str(exc)], "result": {}})
                status = f"fehler: {exc}"
            with log_sperre, open(log_pfad, "a", encoding="utf-8") as f:
                f.write(f"{pd.Timestamp.now(tz='UTC').isoformat()} {methode} {params} -> {status}\n")

        def do_POST(self):  # noqa: N802
            self._antworte(200, {"error": ["EGeneral:Permission denied (synthetischer Server, nur oeffentliche Endpunkte)"], "result": {}})

    return Handler


def serve(ordner: Path, port: int) -> None:
    info = json.loads((ordner / "szenario.json").read_text(encoding="utf-8"))
    markt = SynthetischerMarkt.lade(ordner / "markt.npz")
    api = SynthetischeKrakenAPI({info["paar"]: markt}, EchtzeitUhr())
    server = ThreadingHTTPServer(("127.0.0.1", port), _baue_handler(api, ordner / "server_anfragen.log"))
    print(f"Synthetischer Kraken-Server fuer {info['paar']} ({info['name']}) auf http://127.0.0.1:{port}", flush=True)
    print(f"Modell: {info['formel']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetischer Kraken-Server (nur oeffentliche Endpunkte)")
    sub = parser.add_subparsers(dest="befehl", required=True)
    p = sub.add_parser("serve", help="Server in Echtzeit fuer einen Szenario-Ordner starten")
    p.add_argument("ordner", type=Path)
    p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.ordner, args.port)


if __name__ == "__main__":
    main()
