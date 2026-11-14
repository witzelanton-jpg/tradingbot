"""Zugriff auf die Kraken-API (oeffentlich und privat) mit Rate-Limit und Retries.

Oeffentliche Endpunkte (Ticker, OHLC, AssetPairs) benoetigen keinen API-Schluessel.
Private Endpunkte (Balance, AddOrder, OpenOrders, CancelOrder) muessen signiert
werden (HMAC-SHA512 nach Kraken-Spezifikation) und benoetigen `KRAKEN_API_KEY`
und `KRAKEN_API_SECRET`.

Netzwerkfehler fuehren nie zum Absturz: Nach konfigurierbaren Wiederholungen mit
exponentiellem Backoff wird eine `KrakenVerbindungsFehler`-Ausnahme geworfen, die
von den Aufrufern (z. B. der Scan-Schleife) abgefangen werden kann.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any

import requests

from .config import KrakenApiConfig

logger = logging.getLogger("hurstbot.kraken_api")

# Fehlertexte, bei denen eine Wiederholung sinnvoll ist (typischerweise voruebergehend).
_TRANSIENTE_FEHLERMUSTER = (
    "EAPI:Rate limit exceeded",
    "EService:Unavailable",
    "EService:Busy",
    "EGeneral:Temporary lockout",
    "EOrder:Rate limit exceeded",
)


class KrakenAPIError(Exception):
    """Von Kraken zurueckgemeldeter Anwendungsfehler (z. B. ungueltiges Paar)."""


class KrakenVerbindungsFehler(Exception):
    """Netzwerk- oder Verbindungsfehler nach Ausschoepfen aller Wiederholungen."""


class _RateLimiter:
    """Einfacher Mindestabstand zwischen Anfragen, um Kraken-Limits einzuhalten."""

    def __init__(self, anfragen_pro_sekunde: float) -> None:
        self._mindestabstand = 1.0 / anfragen_pro_sekunde if anfragen_pro_sekunde > 0 else 0.0
        self._letzter_aufruf = 0.0

    def warte(self) -> None:
        jetzt = time.monotonic()
        wartezeit = self._mindestabstand - (jetzt - self._letzter_aufruf)
        if wartezeit > 0:
            time.sleep(wartezeit)
        self._letzter_aufruf = time.monotonic()


def _ist_transienter_fehler(fehlerliste: list[str]) -> bool:
    return any(muster in fehler for fehler in fehlerliste for muster in _TRANSIENTE_FEHLERMUSTER)


class KrakenAPI:
    """Duenne Huelle um die Kraken-REST-API mit Retry/Backoff und Signierung."""

    def __init__(
        self,
        config: KrakenApiConfig,
        api_key: str | None = None,
        api_secret: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._api_secret = api_secret
        self._session = session or requests.Session()
        self._rate_limiter = _RateLimiter(config.anfragen_pro_sekunde)

    # ---------- Interna ----------

    def _public_url(self, methode: str) -> str:
        return f"{self._config.basis_url}/{self._config.api_version}/public/{methode}"

    def _mit_wiederholung(self, func, *args, **kwargs) -> Any:
        letzter_fehler: Exception | None = None
        for versuch in range(self._config.max_wiederholungen):
            self._rate_limiter.warte()
            try:
                return func(*args, **kwargs)
            except KrakenAPIError as exc:
                letzter_fehler = exc
                if not _ist_transienter_fehler([str(exc)]):
                    raise
                logger.warning("Transienter Kraken-Fehler (Versuch %s): %s", versuch + 1, exc)
            except (requests.RequestException, requests.Timeout) as exc:
                letzter_fehler = exc
                logger.warning("Netzwerkfehler bei Kraken-Anfrage (Versuch %s): %s", versuch + 1, exc)
            wartezeit = self._config.backoff_basis_sekunden * (2**versuch)
            time.sleep(wartezeit)
        raise KrakenVerbindungsFehler(
            f"Kraken-API nach {self._config.max_wiederholungen} Versuchen nicht erreichbar"
        ) from letzter_fehler

    def _get_public(self, methode: str, params: dict | None = None) -> dict:
        def _ausfuehren() -> dict:
            antwort = self._session.get(
                self._public_url(methode), params=params or {}, timeout=self._config.timeout_sekunden
            )
            antwort.raise_for_status()
            daten = antwort.json()
            if daten.get("error"):
                raise KrakenAPIError(f"Kraken-Fehler bei {methode}: {daten['error']}")
            return daten["result"]

        return self._mit_wiederholung(_ausfuehren)

    # ---------- Oeffentliche Endpunkte ----------

    def ticker(self, paare: list[str]) -> dict:
        """Aktueller Ticker (u. a. letzter Preis, Geld-/Briefkurs) fuer eine Liste von Paaren."""
        return self._get_public("Ticker", {"pair": ",".join(paare)})

    def ohlc(self, paar: str, intervall_minuten: int, since: int | None = None) -> dict:
        """OHLC-Kerzen. Kraken liefert maximal 720 Kerzen pro Abfrage."""
        params: dict[str, Any] = {"pair": paar, "interval": intervall_minuten}
        if since is not None:
            params["since"] = since
        return self._get_public("OHLC", params)

    def asset_pairs(self, paare: list[str] | None = None) -> dict:
        """Handelsregeln je Paar (u. a. Mindestordervolumen `ordermin`, Praezision)."""
        params = {"pair": ",".join(paare)} if paare else {}
        return self._get_public("AssetPairs", params)

    # ---------- Private (signierte) Endpunkte ----------

    def _signiere(self, urlpfad: str, daten: dict) -> str:
        if not self._api_secret:
            raise KrakenAPIError("Kein KRAKEN_API_SECRET gesetzt")
        postdata = urllib.parse.urlencode(daten)
        nonce_und_postdata = (str(daten["nonce"]) + postdata).encode()
        sha256_digest = hashlib.sha256(nonce_und_postdata).digest()
        nachricht = urlpfad.encode() + sha256_digest
        secret_decoded = base64.b64decode(self._api_secret)
        signatur = hmac.new(secret_decoded, nachricht, hashlib.sha512)
        return base64.b64encode(signatur.digest()).decode()

    def _post_private(self, methode: str, daten: dict | None = None) -> dict:
        if not self._api_key or not self._api_secret:
            raise KrakenAPIError(
                "Privater Endpunkt benoetigt KRAKEN_API_KEY und KRAKEN_API_SECRET als Umgebungsvariablen"
            )
        daten = dict(daten or {})
        daten["nonce"] = int(time.time() * 1000)
        urlpfad = f"/{self._config.api_version}/private/{methode}"

        def _ausfuehren() -> dict:
            headers = {
                "API-Key": self._api_key,
                "API-Sign": self._signiere(urlpfad, daten),
            }
            antwort = self._session.post(
                f"{self._config.basis_url}{urlpfad}",
                data=daten,
                headers=headers,
                timeout=self._config.timeout_sekunden,
            )
            antwort.raise_for_status()
            ergebnis = antwort.json()
            if ergebnis.get("error"):
                raise KrakenAPIError(f"Kraken-Fehler bei {methode}: {ergebnis['error']}")
            return ergebnis["result"]

        return self._mit_wiederholung(_ausfuehren)

    def balance(self) -> dict:
        """Verfuegbares Guthaben je Asset (privater Endpunkt)."""
        return self._post_private("Balance")

    def add_order(
        self,
        paar: str,
        typ: str,
        order_art: str,
        menge: float,
        preis: float | None = None,
        validate: bool = False,
        zeitlimit: str | None = None,
    ) -> dict:
        """Order anlegen. Mit `validate=True` wird nur geprueft, nichts ausgefuehrt."""
        daten: dict[str, Any] = {
            "pair": paar,
            "type": typ,
            "ordertype": order_art,
            "volume": f"{menge}",
        }
        if preis is not None:
            daten["price"] = f"{preis}"
        if validate:
            daten["validate"] = "true"
        if zeitlimit:
            daten["expiretm"] = zeitlimit
        return self._post_private("AddOrder", daten)

    def open_orders(self) -> dict:
        return self._post_private("OpenOrders")

    def cancel_order(self, order_id: str) -> dict:
        return self._post_private("CancelOrder", {"txid": order_id})
