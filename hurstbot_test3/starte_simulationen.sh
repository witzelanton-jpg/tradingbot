#!/usr/bin/env bash
# Startet alle simulierten Paper-Runs parallel (je ein Prozess pro Szenario).
# Konsolenausgabe des Bots je Szenario: laeufe/<name>/konsole.log
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SZENARIEN="${@:-F0_rein F1_weiss F2_grosse_amplitude F3_random_walk F4_trend F5_sprung L1_weiss L2_starkes_rauschen}"
for s in $SZENARIEN; do
  ./.venv/Scripts/python.exe synth/sim_run.py "laeufe/$s" > "laeufe/$s/konsole.log" 2>&1 &
  echo "gestartet: $s (PID $!)"
done
wait
echo "alle Simulationen beendet: $(date)"
