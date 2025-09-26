from pathlib import Path
import os

HEADLESS = False

ROOT = Path(__file__).parents[1]

SESSION_DIR = ROOT / "data" / "session"
LOG_DIR = ROOT / "data" / "logs"
PUPPETS_DIR = ROOT / "data" / "puppets"

UBLOCK_PATH = ROOT / "src" / "puppet" / "extensions"

SLANT_DIR = ROOT / "data" / "slant"
SLANT_ESTIMATIONS = SLANT_DIR / "slant_estimations.csv"
BLACKLIST = SLANT_DIR / "blacklist.pkl"

if not SESSION_DIR.exists():
    os.mkdir(SESSION_DIR)

if not LOG_DIR.exists():
    os.mkdir(LOG_DIR)

if not UBLOCK_PATH.exists():
    os.mkdir(UBLOCK_PATH)