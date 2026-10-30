import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:   # allows `pytest` without `pip install -e .`
    sys.path.insert(0, SRC)
