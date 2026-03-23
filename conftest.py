import sys
from pathlib import Path

# Make both the project root (for `from src.xxx import`) and src/ (for bare
# `from xxx import`) importable in tests.
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
