# hurstbot_test3 – Test mit idealisierten Kursdaten

Protokoll: Claude-Doc „Hurstbot – Testprotokoll 3: Idealisierte Daten (November 2026)“
https://claude.ai/artifact/8Xb9WPp5kmRhV7AMjRKTRq

Kursmodell: `y(t) = P0 + A*sin(2*pi*f*t) + Stoerung(t)`, ein Wert je Minute, daraus Kraken-Kerzen.
Die Codebasis wird nicht verändert (Prüfsummen in `kontrolle/`).

## Umgebung

```
python -m venv .venv
.venv\Scripts\python -m pip install -r ..\Codebasis\requirements.txt
.venv\Scripts\python -m pip install .\werkzeuge\hurstbot_starter     # nur der Befehl "hurstbot"
```

Zusätzlich liegt in `.venv\Lib\site-packages\hurstbot_codebasis.pth`:

```
D:/Monatsprojekt/09-26_Traidingbot/Codebasis/src
import sys; sys.dont_write_bytecode = True
```

(statt `pip install -e Codebasis`, das in der Codebasis schreiben würde).

## Daten erzeugen (Amplitude, Frequenz, Störung)

```
.venv\Scripts\python synth\generator.py --szenarien szenarien.yaml          # alle Szenarien
.venv\Scripts\python synth\generator.py --name mein_test --amplitude 1.4 --frequenz 40 ^
        --stoerung weiss=0.2 --stoerung trend=-5@0 --stunden 8                # eigenes Szenario
.venv\Scripts\python synth\generator.py                                      # fragt alles ab
```

Störungen: `weiss=σ` (EUR je Minute), `random_walk=σ` (EUR pro Tag), `trend=EUR_pro_Tag@ab_stunden`,
`sprung=EUR@ab_stunden`. Frequenz in Schwingungen pro Tag (40 = Periode 36 min).
Jedes Szenario landet in `laeufe\<name>\` mit eigener `config.yaml` und `data\ logs\ state\`.

## Bot laufen lassen (Paper)

Simuliert (schnell, simulierte Uhr, gleicher Bot-Aufbau wie `hurstbot run`):

```
.venv\Scripts\python synth\sim_run.py laeufe\mein_test
bash starte_simulationen.sh            # alle Szenarien parallel
```

Echter Befehl gegen den lokalen Fake-Kraken-Server (Echtzeit):

```
.venv\Scripts\python synth\generator.py --name echt --amplitude 1.4 --frequenz 40 --stoerung weiss=0.2 --stunden 3 --echtzeit --port 8765
.venv\Scripts\python synth\fake_kraken.py serve laeufe\echt --port 8765
cd laeufe\echt
..\..\.venv\Scripts\hurstbot --config config.yaml run --max-zyklen 100
```

Hinweis: Ein Zyklus kostet auf dem Test-Laptop 10–38 s (Bootstrap). Lange Läufe brauchen
einen wachen Rechner (`werkzeuge\wach_halten.py`) und genug freien Speicher.

## Auswerten

```
.venv\Scripts\python synth\auswertung.py            # -> auswertung\zusammenfassung.csv, ergebnisse.json
```

## Ordner

| Ordner/Datei | Inhalt |
| --- | --- |
| `synth/kursmodell.py` | Kursmodell, Kerzenbildung im Kraken-Format |
| `synth/generator.py` | Eingabe von Amplitude, Frequenz, Störungen; legt Szenario-Ordner an |
| `synth/fake_kraken.py` | synthetische Kraken-API (im Prozess) und lokaler HTTP-Server |
| `synth/sim_run.py` | simulierter Paper-Run |
| `synth/auswertung.py` | Auswertung inkl. Vergleich OHLC-Cache gegen wahre Kerzen |
| `laeufe/` | ein Ordner je Lauf (T0 Kurztest, F0–F5, F4b, L1–L2, C0/C1 echter Befehl) |
| `kontrolle/` | SHA-256 der Codebasis vor und nach dem Test |
