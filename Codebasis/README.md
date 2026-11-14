# Hurstbot — Hurst-basierter Grid-Trading-Bot für Kraken

Ein Grid-Trading-Bot, der pro Paar nach einer Schwellenstrategie handelt (bei −X % kaufen,
bei +X % über dem Einstandspreis verkaufen) und **nur dann handelt, wenn der Kurs statistisch
als oszillierend (mean-reverting) eingeschätzt wird**. Die Einschätzung erfolgt über den
Hurst-Exponenten, geschätzt über mehrere Verfahren und Zeitskalen, mit Konfidenzintervall.

Die Schwelle X wird per Walk-Forward-Backtest optimiert, der Betrag Y ergibt sich aus der
Stärke und Sicherheit des Signals. Mehrere Schutzmechanismen (Trend-/Ausbruchswächter,
Exposure-Limit, Kill-Switch) sollen verhindern, dass der Bot in einen Trend hineinhandelt.

> **Bitte zuerst den Abschnitt [Risiken und Grenzen](#risiken-und-grenzen) lesen.**
> Der Hurst-Exponent ist eine statistische Schätzung, keine Prognose.

---

## Inhalt

- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [API-Schlüssel einrichten](#api-schlüssel-einrichten)
- [Betriebsmodi](#betriebsmodi)
- [Befehle](#befehle)
- [Wie der Bot entscheidet](#wie-der-bot-entscheidet)
- [Schutzmechanismen](#schutzmechanismen)
- [Dateien und Zustand](#dateien-und-zustand)
- [Tests](#tests)
- [Risiken und Grenzen](#risiken-und-grenzen)
- [Annahmen und offene Punkte](#annahmen-und-offene-punkte)

---

## Installation

Voraussetzung: **Python 3.11 oder neuer**.

```bash
# Virtuelle Umgebung anlegen und aktivieren
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Abhängigkeiten installieren
pip install -r requirements.txt

# Paket installieren (macht den Befehl "hurstbot" verfügbar)
pip install -e .
```

Optional, für die Bayes'sche Verfeinerung der X-Optimierung:

```bash
pip install optuna
```

Danach in der Konfiguration `optimizer.mit_optuna: true` setzen.

---

## Konfiguration

```bash
cp config.example.yaml config.yaml
```

`config.example.yaml` ist vollständig kommentiert. Die wichtigsten Parameter:

| Parameter | Default | Bedeutung |
|---|---|---|
| `start_kapital_eur` | 100 | Kapitalgrenze und Bezugsgröße für Exposure/Kill-Switch |
| `gate.h_start` / `gate.h_stop` | 0.45 / 0.50 | Hysterese: Handel beginnt unter `h_start`, endet ab `h_stop` |
| `y.y_min_eur` / `y.y_max_eur` | 5 / 50 | Spanne für den Betrag pro Kauf |
| `gebuehren.taker_gebuehr_prozent` | 0.26 | **Auf die eigene Kraken-Gebührenstufe anpassen** |
| `exposure.max_exposure_prozent` | 40 | Ab diesem Anteil in einem Paar keine weiteren Käufe |
| `kill_switch.max_verlust_prozent` | 10 | Gesamtverlust, bei dem der Bot vollständig stoppt |
| `scanner.watchlist` | 5 Paare | Welche Paare überhaupt betrachtet werden |
| `scanner.max_parallele_paare` | 2 | Wie viele Paare gleichzeitig gehandelt werden |
| `optimizer.neuoptimierung_stunden` | 6 | Wie oft X neu optimiert wird |
| `optimizer.glaettung_alpha` | 0.3 | Wie schnell X neuen Werten folgen darf |

Alle Pfade (`data/`, `logs/`, `state/`) lassen sich unter `pfade:` umstellen.

> **Hinweis zu den Gebühren:** Zu niedrig angesetzte Gebühren machen den Backtest
> systematisch zu optimistisch und können eine in Wirklichkeit verlustreiche Strategie
> profitabel erscheinen lassen. Die tatsächliche Gebührenstufe steht im Kraken-Konto.

---

## API-Schlüssel einrichten

Für `run` (Paper) wird **kein** API-Schlüssel benötigt — nur öffentliche Kursdaten.
Für `--validate` und `--live` ist ein Schlüssel nötig.

```bash
cp .env.example .env
# .env öffnen und KRAKEN_API_KEY sowie KRAKEN_API_SECRET eintragen
```

**Berechtigungen des Schlüssels (wichtig):** Der Schlüssel darf in den Kraken-Konto­einstellungen
ausschließlich diese Rechte haben:

- ✅ **Orders anlegen und abbrechen** (*Create & cancel orders*)
- ✅ **Guthaben abfragen** (*Query funds*, *Query open orders & trades*)
- ❌ **Auszahlungsrechte** (*Withdraw funds*) — **niemals aktivieren**

Ein Schlüssel ohne Auszahlungsrecht kann im Schadensfall kein Geld aus dem Konto bewegen.

Weitere Regeln, die im Code durchgesetzt werden:

- Schlüssel werden **nur** aus Umgebungsvariablen bzw. `.env` gelesen, nie aus dem Code.
- `.env` steht in `.gitignore` und darf nicht committet werden.
- Schlüssel und Signaturen werden **nie** geloggt.

---

## Betriebsmodi

Der Bot hat drei Ausführungsstufen. Sicherheit hat Vorrang: der Default ist immer Paper.

### 1. Paper (Default)

```bash
hurstbot run
```

Virtuelle Ausführung mit realistischem Gebühren- und Spread-Modell. Es werden echte
Kursdaten gelesen, aber **keine Orders gesendet**. Kein API-Schlüssel nötig.

### 2. Validate

```bash
hurstbot run --validate
```

Orders werden an die private Kraken-API gesendet, aber mit `validate=true`: Kraken prüft
Format, Mindestgröße und Guthaben und meldet zurück, ob die Order angenommen **würde** —
ausgeführt wird **nichts**. Das ist der empfohlene letzte Test vor Live.

### 3. Live — echte Orders

```bash
hurstbot run --live
```

Der Live-Modus kommt nur zustande, wenn **alle drei** Freigaben vorliegen:

1. die Programmoption `--live` ist gesetzt,
2. die Umgebungsvariable `KRAKEN_LIVE_CONFIRM=JA` existiert (exakt `JA`),
3. die interaktive Bestätigung beim Start wird mit `JA` beantwortet.

Vor der Abfrage zeigt der Bot Watchlist, Betragsspanne, Kapitalgrenze, Exposure-Limit und
Kill-Switch-Schwelle an. Fehlt eine der drei Freigaben, bricht der Start ab, ohne dass eine
einzige Order gesendet wird. Die Prüfung sitzt im Konstruktor des Live-Brokers, nicht nur
in der CLI — es gibt keinen Codepfad, der sie umgeht.

**Empfohlene Reihenfolge:** `backtest` → `optimize` → `run` (Paper, mehrere Tage) →
`run --validate` → erst dann, mit kleinem Kapital, `run --live`.

---

## Befehle

```bash
# Watchlist bewerten und Ranking ausgeben (handelt nicht)
hurstbot scan

# Einmalige Simulation mit festem X
hurstbot backtest --pair XDGEUR --x 1.5 --y 20

# Walk-Forward-Optimierung von X, schreibt CSV + Textbericht nach logs/
hurstbot optimize --pair XDGEUR --y 20

# Betrieb
hurstbot run                    # Paper (Default)
hurstbot run --validate         # Orders nur prüfen lassen
hurstbot run --live             # echte Orders (drei Freigaben nötig)

# Nützliche Zusätze
hurstbot --config andere.yaml run
hurstbot --debug scan
hurstbot run --max-zyklen 5     # nach 5 Zyklen beenden
hurstbot backtest --pair XDGEUR --profil 1h
```

Beenden im Dauerbetrieb mit **Strg + C**: der Bot speichert den Zustand und behandelt
offene Orders (im Live-Modus werden sie storniert).

---

## Wie der Bot entscheidet

### Hurst-Exponent (robust)

- Gerechnet wird auf **Log-Preisen**, nicht auf Rohpreisen.
- Mindestens zwei unabhängige Schätzer werden kombiniert (Median): **Lag-Std** (Steigung
  im Log-Log-Raum) und **R/S** (Rescaled Range); optional **DFA**.
- Die Unsicherheit wird per **Block-Bootstrap** geschätzt (Blöcke erhalten die
  Autokorrelationsstruktur). Ergebnis: H, 90-%-Konfidenzintervall und eine
  Konfidenz-Kennzahl in [0, 1] (schmales Intervall = hohe Konfidenz).

### Drei Zeitskalen

| Skala | Kerzen | Abdeckung | Rolle |
|---|---|---|---|
| kurzfristig | 5 min | ca. 60 h | muss die Kriterien erfüllen |
| mittelfristig | 1 h | ca. 30 Tage | muss die Kriterien erfüllen |
| langfristig | 1 Tag | soweit verfügbar | Zusatzscore, verhindert Handel im Dauertrend |

Gehandelt wird nur, wenn **kurz- und mittelfristig** die Kriterien erfüllt sind; dabei
zählt jeweils der schlechtere (höhere) H-Wert.

### Monotonie-Prüfung

Zusätzlich zu H wird gemessen, wie geradlinig der Kurs läuft: Anteil gleichgerichteter
Kerzen, Rang-Korrelation (Spearman) zwischen Zeit und Kurs und das Verhältnis von
Gesamtbewegung zu Pfadlänge. Starke Monotonie senkt den Score.

### Handels-Gate mit Hysterese

- **H ≥ 0.5 oder obere Konfidenzgrenze ≥ 0.5 → kein Handel**, unabhängig von allem anderen.
- Handel **beginnt** erst bei H < `h_start` (0.45) und **endet** ab H ≥ `h_stop` (0.50).
- Dazwischen bleibt der laufende Zustand erhalten, damit der Bot nicht flackert.

### Betrag Y (5–50 EUR)

`Y = Y_MIN + (Y_MAX − Y_MIN) · Score · Konfidenz`, zusätzlich begrenzt durch verfügbares
Kapital, Exposure-Limit und das Kraken-Mindestordervolumen des Paares (`ordermin` aus
`AssetPairs`). Unsicheres oder nur knapp niedriges H hält Y nahe 5 EUR.

### Optimierung von X

- **Walk-Forward:** mehrere rollierende Fenster, pro Fenster Training und anschließender
  Out-of-Sample-Test. Das finale X ist der Median der pro Fenster gewählten Werte.
- **Zielfunktion:** maximaler Netto-Gewinn nach Gebühren, mit Strafterm für Max-Drawdown
  und für lange Haltedauer offener Lots. Round-Trips und mittlere Amplitude werden
  zusätzlich ausgegeben.
- **Untergrenze:** X muss über dem Gebühren-Break-even liegen
  (`2 × Gebühr + Spread + Puffer`), sonst kann ein Round-Trip nie profitabel sein.
- **Glättung:** neue X-Werte werden exponentiell mit dem alten gemischt
  (`glaettung_alpha`), damit der Bot sein Raster nicht ständig verschiebt.
- **Belastbarkeit:** X gilt nur als validiert, wenn es out-of-sample überhaupt gehandelt
  hat (≥ 1 Round-Trip) **und** dabei im Mittel Gewinn erzielte. Ist eine der beiden
  Bedingungen verletzt, behält der Bot sein bisheriges X und warnt — ein X ohne
  Out-of-Sample-Beleg oder mit negativem Erwartungswert wird nie übernommen. (Dasselbe
  Kriterium gilt bei der Zeitskalen-Auswahl.)

### Paarspezifische Zeitskala

Jedes Paar kann ein eigenes Profil (Kerzenlänge + Hurst-Fenster) aus einer festen
Kandidatenliste bekommen (Default: 5m/200, 15m/200, 1h/200). Die Auswahl erfolgt
**ausschließlich nach Out-of-Sample-Nettogewinn** mit Mindestanzahl Round-Trips —
**nicht** nach dem kleinsten H (das wäre Selection Bias; ein Test auf reinem Rauschen
stellt sicher, dass der Selektor dort kein „gutes" Profil findet). Gewechselt wird
höchstens alle 24 h und nur bei mindestens 20 % Vorteil. Offene Lots bleiben beim
Wechsel erhalten; die Buchführung ist unabhängig von der Kerzenlänge.

### Lot-Buchführung

Jeder Kauf ist ein eigenes **Lot** mit Einstandspreis (inkl. Gebühr), Menge und Zeitpunkt.
Verkauft wird nur Bestand, der vorher gekauft wurde, und ein Lot nur dann, wenn der
Nettoerlös nach Gebühren und Spread **echt über** dem Einstandspreis liegt. Ein Verkauf
mit Netto-Verlust ist strukturell ausgeschlossen.

---

## Schutzmechanismen

| Mechanismus | Auslöser | Wirkung |
|---|---|---|
| **Exposure-Limit** | > 40 % des Gesamtkapitals in einem Paar | keine weiteren Käufe; Verkäufe weiter erlaubt |
| **Trend-/Ausbruchswächter** | Abweichung > 3 × ATR vom Anker, **oder** > 3 % im Zeitfenster, **oder** Einzelkerze > 6 × robuste (MAD-)Sigma | Zustand `PAUSIERT`, keine neuen Orders; offene Lots bleiben unberührt |
| **Wiedereinstieg** | Kurs zurück im Toleranzband (±1 %) um den Ankerpreis | Handel wird fortgesetzt |
| **Wiedereinstieg (Fallback)** | 12 stabile Kerzen in neuer Spanne **und** H < `h_start` | Ankerpreis wird neu gesetzt (per Config abschaltbar) |
| **Wiedereinstieg (Timeout)** | 48 h pausiert ohne Rückkehr | deutliche Warnung im Log |
| **Hurst-Wächter** | H steigt auf ≥ `h_stop` | Paar geht in „nur verkaufen", keine neuen Käufe (realisiert über das Gate) |
| **Kill-Switch** | Gesamtwert > 10 % unter Startkapital | **gesamter Bot stoppt**, Eintrag im Log |
| **Paar-Rotation** | Paar verliert Gate oder fällt aus den Top-N | Modus „nur verkaufen" — offene Lots verwaisen nie |

Jeder Zustandswechsel (`HANDELT` / `PAUSIERT` / `NUR_VERKAUFEN` / `GESTOPPT`) wird mit
Grund und Zeitstempel protokolliert.

---

## Dateien und Zustand

```
data/      OHLC-Cache (CSV), wächst über die Zeit und schont das API-Rate-Limit
logs/      trade_log.csv        alle Käufe/Verkäufe
           zustands_log.jsonl   alle Zustandswechsel mit Grund
           backtest_*.csv       Berichte aus "backtest"
           optimierung_*.csv/txt Berichte aus "optimize"
state/     bot_zustand.json     Portfolio, Lots, Anker, X je Paar
```

**Idempotenz:** Der Zustand wird nach jeder Orderausführung geschrieben — erst in eine
temporäre Datei, dann per atomarem `os.replace` umbenannt. Ein Absturz mitten im Schreiben
kann daher keine halb geschriebene Zustandsdatei hinterlassen. Eine beschädigte Datei
führt nicht zum Absturz: der Bot meldet sie und startet bewusst frisch.

**Rate-Limits:** Alle Anfragen laufen über einen gemeinsamen Mindestabstand
(`anfragen_pro_sekunde`), mit exponentiellem Backoff und Wiederholungen bei
`EAPI:Rate limit exceeded` oder Netzwerkfehlern. OHLC-Daten werden lokal gecacht und nur
inkrementell (über `since`) nachgeladen.

---

## Tests

```bash
pytest                    # alle Tests
pytest -v                 # mit Einzelnamen
pytest tests/test_hurst.py
```

168 Tests, Laufzeit ca. 30 s. Die Kraken-API ist durchgehend **gemockt** — die Tests
kontaktieren nie die echte API. Abgedeckt sind unter anderem:

- **Hurst:** künstliche trendende / zufällige / oszillierende Reihen liefern
  H > 0.5 / ≈ 0.5 / < 0.5 (AR(1)-Modell auf Log-Renditen)
- **Konfidenz und Y-Mapping:** Y bleibt immer in [5, 50] bzw. unter allen Limits
- **Gate-Hysterese** und alle Wächter (Exposure, Ausbruch, Wiedereinstieg, Kill-Switch)
- **Lot-Buchführung:** kein Verkauf mit Netto-Verlust, auch nicht bei gestiegenem Rohpreis
- **Broker-Sicherheit:** Live-Modus ist ohne alle drei Freigaben nicht möglich
- **Kein Look-Ahead-Bias:** das Anhängen zukünftiger Kerzen ändert keinen Trade im
  vorherigen Zeitraum
- **Selection Bias:** auf reinem Rauschen findet der Zeitskala-Selektor kein „gutes" Profil
- **Robustheit:** Netzwerk- und Ticker-Fehler führen nie zum Absturz

---

## Risiken und Grenzen

**Bitte vollständig lesen, bevor echtes Geld eingesetzt wird.**

- **Der Hurst-Exponent ist eine statistische Schätzung, keine Garantie.** Er beschreibt
  die *Vergangenheit* eines Kursfensters. Ein niedriger H-Wert bedeutet nicht, dass der
  Kurs weiter oszilliert — das Regime kann jederzeit kippen, und zwar genau dann, wenn
  der Bot gerade investiert ist.

- **Vergangene Oszillation sagt keine zukünftige voraus.** Der Backtest zeigt, was mit den
  gewählten Parametern *gewesen wäre*. Walk-Forward-Validierung reduziert Overfitting,
  beseitigt es aber nicht.

- **Gebühren können die Strategie unprofitabel machen.** Bei engen Schwellen frisst die
  Gebühr (2 × ca. 0.26 % plus Spread) den Gewinn vollständig auf. Deshalb liegt X zwingend
  über dem Break-even — das begrenzt aber auch, wie viele Oszillationen überhaupt
  abgefangen werden können.

- **Die Strategie ist strukturell long-biased.** Gekauft wird in fallende Kurse. Fällt ein
  Kurs dauerhaft, sammelt der Bot Lots an, die den Zielpreis nie erreichen. Der
  Ausbruchswächter begrenzt das, verhindert es aber nicht vollständig — offene Lots bleiben
  bewusst unberührt und können lange im Minus stehen.

- **Verkauft wird nur mit Netto-Gewinn.** Das schützt vor Verlustverkäufen, bedeutet aber:
  Lots können unbegrenzt lange offen bleiben. Es gibt bewusst **kein Stop-Loss** auf
  Lot-Ebene. Der Kill-Switch greift erst auf Portfolio-Ebene.

- **Der Kill-Switch stoppt, er verkauft nicht.** Bei Auslösung werden keine neuen Orders
  mehr gesendet, bestehende Positionen bleiben aber im Markt. Das Schließen bleibt eine
  bewusste Entscheidung des Nutzers.

- **Krypto-Märkte sind hochvolatil und laufen durch.** Zwischen zwei Zyklen (Default 60 s)
  kann sich der Kurs beliebig weit bewegen. Der Bot sieht keine Zwischenbewegung.

- **Limit-Orders werden nicht garantiert ausgeführt.** Der Live-Broker nutzt Limit-Orders
  mit Zeitlimit und storniert nicht ausgeführte Orders. Ein Lot kann dadurch länger
  offen bleiben als geplant.

- **Technische Risiken bleiben:** API-Ausfälle, Netzwerkprobleme, Delisting eines Paares,
  Änderungen an Krakens Gebühren oder Mindestordergrößen. Der Bot fängt Fehler ab und
  stürzt nicht ab, kann aber in solchen Phasen nicht handeln.

- **Diese Software ist keine Finanzberatung.** Setze nur Kapital ein, dessen vollständigen
  Verlust du verkraften kannst. Teste ausführlich in Paper und Validate, bevor du Live
  gehst, und beginne dann mit dem kleinstmöglichen Betrag.

---

## Annahmen und offene Punkte

### Getroffene Annahmen

1. **Gebührenmodell:** Es wird durchgehend mit der **Taker**-Gebühr gerechnet, auch wenn
   der Live-Broker Limit-Orders (Maker) bevorzugt. Das ist die konservative Richtung —
   die Realität sollte günstiger ausfallen.
2. **Spread:** Als fester Prozentwert aus der Config angesetzt (Default 0.05 %), nicht
   live aus dem Orderbuch gelesen. Bei illiquiden Paaren kann der echte Spread deutlich
   größer sein.
3. **Anker-Logik:** Nach jedem Trade wird der Anker auf den aktuellen Preis zurückgesetzt,
   sodass sich bei anhaltender Bewegung eine Leiter von Käufen ergibt. Verkauft wird
   unabhängig vom Anker je Lot anhand seines individuellen Einstandspreises.
4. **Gate-Kombination:** Für „kurz UND mittel" wird jeweils der schlechtere (höhere)
   H-Wert und die niedrigere Konfidenz verwendet.
5. **Konfidenz-Kennzahl:** Normiert auf eine Referenz-Intervallbreite von 0.5 — ein
   Intervall so breit wie der gesamte sinnvolle H-Bereich ergibt Konfidenz 0.
6. **Paper- und Backtest-Fills** nutzen dasselbe Gebühren-/Spread-Modell, damit beide
   Ergebnisse vergleichbar sind.
7. **Kapitalverteilung** erfolgt proportional zum Score der aktiven Paare; freies Kapital
   wird beim Wechsel nachgeschoben, aber nicht aus laufenden Paaren abgezogen.
8. **Zeitzonen:** Intern wird durchgehend UTC verwendet (Kraken-Zeitstempel).

### Offene Punkte

1. **Zu kurze Out-of-Sample-Fenster bei 5-Minuten-Kerzen.** Kraken liefert max. 720 Kerzen
   pro Abfrage. Bei 5 Walk-Forward-Fenstern bleiben pro Testfenster nur ca. 43 Kerzen
   (ca. 3.5 h) — darin kommt oft kein einziger Round-Trip zustande. Der Bot erkennt das
   und übernimmt ein solches X **nicht** (siehe Warnung im Optimierungsbericht), aber
   bis der lokale Cache über mehrere Tage gewachsen ist, liefert die X-Optimierung auf
   5m-Kerzen kaum belastbare Werte. **Empfehlung:** zunächst mit `--profil 1h` optimieren
   oder `walk_forward_fenster_anzahl` auf 2–3 senken.

   Im Test am 04.10.2026 ergab `optimize --pair XDGEUR --profil 1h` ein X von 2.27 % mit
   einem mittleren Out-of-Sample-Ergebnis von **−0.46 EUR** — also verlustbringend. Der
   Bot wertet das korrekt als nicht belastbar und handelt damit nicht. Das ist kein
   Fehler, sondern das erwartete Verhalten, wenn ein Paar im betrachteten Zeitraum
   schlicht nicht oszilliert hat.

2. **Das Gate ist in der Praxis sehr streng.** Beim Test gegen die echte API (04.10.2026)
   bestand **kein einziges** der fünf Watchlist-Paare das Gate — die obere Konfidenzgrenze
   lag überall ≥ 0.5, obwohl H selbst teils bei 0.44 lag. Das ist exakt das spezifizierte
   Verhalten („H ≥ 0,5 oder obere Konfidenzgrenze ≥ 0,5: kein Handel, auf keinen Fall"),
   bedeutet aber: mit Bootstrap-Intervallbreiten von ca. 0.07–0.08 handelt der Bot sehr
   selten. Wer häufiger handeln will, müsste bewusst entscheiden, das Kriterium auf H
   allein zu stützen — **das ist eine Risikoentscheidung und wurde hier nicht getroffen.**

3. **`hurst_waechter.bestand_abbauen` ist als Konfigurationsschalter vorhanden, aber noch
   nicht implementiert** (Default `false`, wie spezifiziert). Bei Umschalten auf `true`
   passiert derzeit nichts; der aktive Bestandsabbau bei steigendem H fehlt noch.
   Der *passive* Teil des Hurst-Wächters (keine neuen Käufe ab H ≥ `h_stop`) ist
   implementiert — er wird über das Gate realisiert, das in diesem Fall
   `NUR_VERKAUFEN` erzwingt.

4. **`zeitskala_auswahl.dominante_zyklusdauer_vorschlag` ist noch nicht implementiert**
   (laut Spezifikation optional). Die Kandidaten kommen aktuell immer aus der festen
   Liste in der Config.

5. **Kein Zustandsabgleich mit Kraken beim Live-Neustart.** Die Spezifikation verlangt,
   dass der Live-Broker beim Neustart den lokalen Zustand mit den echten Beständen und
   Orders bei Kraken abgleicht. Implementiert ist bisher nur das Stornieren offener
   Orders beim Beenden. **Vor dem ersten echten Live-Einsatz sollte dieser Abgleich
   ergänzt werden** — sonst kann der lokale Lot-Bestand von der Realität abweichen,
   z. B. nach einem Absturz zwischen Orderausführung und Zustandsspeicherung.

6. **Maker-Orders werden nicht aktiv bevorzugt platziert.** Der Live-Broker nutzt
   Limit-Orders, setzt das Limit aber nahe am Marktpreis (Spread-Hälfte), statt sich
   bewusst in das Orderbuch zu legen. Echte Maker-Rabatte werden so nicht zuverlässig
   erreicht.

7. **Der Scanner bewertet alle Paare sequenziell.** Bei 5 Paaren dauert ein Zyklus ca.
   90 s (dominiert vom Bootstrap und dem Rate-Limit). Bei deutlich größeren Watchlists
   sollte der Hurst-Bootstrap parallelisiert oder seine Wiederholungszahl gesenkt werden.

8. **DFA ist implementiert, aber nicht im Default aktiv** (`schaetzer: [lag_std, rs]`),
   da es die Laufzeit pro Bewertung merklich erhöht. Für genauere Schätzungen kann es
   in der Config ergänzt werden.

---

## Projektstruktur

```
src/hurstbot/
  config.py          Konfiguration laden und validieren (pydantic)
  kraken_api.py      Public + Private Endpunkte, Rate-Limit, Retries, Signierung
  data.py            OHLC laden und cachen, Mehrskalen-Abruf
  hurst.py           Schätzer (Lag-Std, R/S, DFA), Bootstrap-Konfidenz, Monotonie
  regime.py          Score, Gate mit Hysterese, Y-Berechnung
  backtest.py        Lot-Buchführung und Simulation mit Gebühren/Spread
  optimizer.py       X-Optimierung (Walk-Forward), Zeitskalen-Auswahl
  guards.py          Trendwächter, Exposure, Wiedereinstieg, Kill-Switch
  strategy.py        Grid-Logik pro Paar (Lots, Anker, Zustandsautomat)
  scanner.py         Watchlist-Prüfung, Ranking, Kapitalverteilung
  broker.py          Paper / Validate / Live
  runner.py          Hauptschleife (Orchestrierung)
  state.py           atomare Persistenz
  logging_setup.py   Konsole + CSV-Trade-Log + Zustands-Log
  cli.py             Befehle: scan, backtest, optimize, run
tests/               168 Tests, Kraken-API durchgehend gemockt
```

## Lizenz

MIT
