from .youtube_api import get_videos_metadata
from .db import get_videos, update_videos
import random

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


if __name__ == "__main__":
    pass