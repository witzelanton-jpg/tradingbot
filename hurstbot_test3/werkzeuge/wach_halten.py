"""Haelt Windows waehrend der Testlaeufe wach (kein Standby bei Leerlauf).

Nutzt SetThreadExecutionState: wirkt nur, solange dieser Prozess laeuft, und
aendert keine Energieeinstellungen dauerhaft. Endet nach --stunden oder mit Strg+C.
(Zugeklappter Deckel kann trotzdem in den Standby fuehren.)
"""

import argparse
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

parser = argparse.ArgumentParser()
parser.add_argument("--stunden", type=float, default=4.0)
args = parser.parse_args()

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print(f"System wird {args.stunden:g} h wach gehalten.", flush=True)
try:
    time.sleep(args.stunden * 3600)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("Wachhalten beendet.", flush=True)
