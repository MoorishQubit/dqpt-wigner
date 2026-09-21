import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=str(ROOT/"src")
if src not in sys.path:
    sys.path.insert(0,src)
