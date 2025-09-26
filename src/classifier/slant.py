import pandas as pd
from pathlib import Path
from functools import lru_cache
import pickle
import random

import config
from models import Video

@lru_cache(maxsize=1)
def load_slants():
    """ Return and cache slant df """
    return pd.read_csv(config.SLANT_ESTIMATIONS)


def load_blacklist() -> list[str]:
    if not config.BLACKLIST.exists():
        return []
    with open(config.BLACKLIST, "rb") as f:
        bl = pickle.load(f)
    return bl
    

def get_slant(id : str) -> float | None:
    """ Get slant from dataframe for given YT ID. Returns None if not found. """
    df = load_slants()
    video_row = df.loc[df["video_id"] == id]
    if not video_row.empty:
        return video_row["slant"].iloc[0]
    else:
        return None


def get_videos(slant_range : tuple, exclude : list[str] = [], n = 0) -> list[Video]:
    """ Return videos in slant range. Optionally exclude list of ids. Optionally define n videos to randomly sample """
    df = load_slants()
    mask = df["slant"].between(*slant_range,inclusive="both")

    vids = [Video(vid, slant) for vid, slant in zip(df[mask]["video_id"], df[mask]["slant"])]
    vids = [vid for vid in vids if vid.id not in exclude] # exclude

    if n > 0: vids = random.sample(vids, k=n)

    return vids


def write_blacklist(ids : list[str]):
    """ Write ids to blacklist pickle """
    blacklist : list = load_blacklist()
    blacklist.extend(ids)

    with open(config.BLACKLIST, "wb") as f:
        pickle.dump(blacklist, f)

