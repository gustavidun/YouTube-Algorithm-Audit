from pathlib import Path
import os

HEADLESS = False
ROOT = Path(__file__).parents[1]
SESSION_DIR = ROOT / "data" / "session"
LOG_DIR = ROOT / "data" / "logs"
UBLOCK_PATH = ROOT / "puppet" / "extensions" / "ublock"

if not SESSION_DIR.exists():
    os.mkdir(SESSION_DIR)

if not LOG_DIR.exists():
    os.mkdir(LOG_DIR)