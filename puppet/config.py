from pathlib import Path

HEADLESS = False
ROOT = Path(__file__).parents[1]
SESSION_DIR = ROOT / "data" / "session"
UBLOCK_PATH = ROOT / "puppet" / "extensions" / "ublock"