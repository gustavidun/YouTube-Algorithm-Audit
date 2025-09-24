import pandas as pd
from pathlib import Path
from functools import lru_cache
import pickle

ROOT = Path(__file__).parents[1]
SLANT_DIR = ROOT / "data" / "slant"
SLANT_ESTIMATIONS = SLANT_DIR / "slant_estimations.csv"
BLACKLIST = SLANT_DIR / "blacklist.pkl"


@lru_cache(maxsize=1)
def load_slants():
    """ Return and cache slant df """
    return pd.read_csv(SLANT_ESTIMATIONS)


def load_blacklist() -> list[str]:
    if not BLACKLIST.exists():
        return []
    with open(BLACKLIST, "rb") as f:
        bl = pickle.load(f)
    return bl
    

def get_slant(video_id : str) -> float | None:
    """ Get slant from dataframe for given YT ID. Returns None if not found. """
    df = load_slants()
    video_row = df.loc[df["video_id"] == video_id]
    if not video_row.empty:
        return video_row["slant"].iloc[0]
    else:
        return None


def get_video_ids(slant_range : tuple, exclude : list[str] = []) -> list[str]:
    """ Return video IDs in slant range. Optionally exclude list of ids """
    df = load_slants()
    mask = df["slant"].between(*slant_range,inclusive="both")
    ids = df[mask]["video_id"].tolist()
    ids_exc = [id for id in ids if id not in exclude]
    return ids_exc


def write_blacklist(ids : list[str]):
    """ Write ids to blacklist pickle """
    blacklist : list = load_blacklist()
    blacklist.extend(ids)

    with open(BLACKLIST, "wb") as f:
        pickle.dump(blacklist, f)

