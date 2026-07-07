import sys
from pathlib import Path

# Make the project root importable so `from src.xxx import ...` resolves in
# tests. `src/` is a real package (src/__init__.py); the single canonical
# import style is `from src.X import ...`.
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
