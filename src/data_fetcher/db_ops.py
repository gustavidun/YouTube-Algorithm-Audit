from .youtube_api import get_videos_metadata, get_comments
from .db import get_videos, update_videos, save_snapshot, get_connection
from config import SLANT_DIR
import random
import pandas as pd
from models import Video
from config import SLANTS_TRAIN

def build_metadata():
    """Fills db with metadata from the YouTube Data API"""
    vids = get_videos((-1,1))
    random.shuffle(vids)

    print(f"Building metadata for {len(vids)} videos")

    chunks = [vids[i:i+49] for i in range(0, len(vids), 49)]

    for i, chunk in enumerate(chunks):
        vids_u = get_videos_metadata(chunk)

        for vid in vids_u: #blacklist videos with no metadata
            if vid.title is None: vid.blacklist = True

        update_videos(vids_u)
        print(f"Updated chunk {i}/{len(chunks)}")


def blacklist_empty(): 
    """Blacklist videos with no metadata."""
    vids = get_videos((-1,1))
    for vid in vids:
        if vid.title is None: vid.blacklist = True
    update_videos(vids)


def trim_train(landmark_threshold : int = 12): 
    """Removes train label from videos with landmark count below threshold"""
    vids = get_videos((-1,1))
    for vid in vids:
        if vid.L + vid.R < landmark_threshold:
            vid.train = False
    update_videos(vids)


def fill_comments():
    vids = get_videos((-1,1))
    vids = [vid for vid in vids if vid.train and not vid.comments and not vid.blacklist] # filter out non-train and already filled
    chunks = [vids[i:i+49] for i in range(0, len(vids), 49)]
    for i, chunk in enumerate(chunks):
        for vid in chunk:
            u_vid = get_comments(vid)
            if u_vid: vid.comments = u_vid.comments
        update_videos(chunk)
        print(f"Updated chunk {i}/{len(chunks)}")


def fill_landmarks():
    df = pd.read_csv(SLANT_DIR / "slant_estimations.csv")
    vids = get_videos((-1,1))

    L_map = dict(zip(df["video_id"], df["liberal_landmark_follows"]))
    R_map = dict(zip(df["video_id"], df["conservative_landmark_follows"]))

    for i, vid in enumerate(vids):
        print(f"{i}/{len(vids)}")
        if vid.id in L_map and vid.id in R_map:
            vid.L = L_map.get(vid.id)
            vid.R = R_map.get(vid.id)
    update_videos(vids)

if __name__ == "__main__":
    pass