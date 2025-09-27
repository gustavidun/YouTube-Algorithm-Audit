from itertools import islice
from youtube_comment_downloader import *
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time

import config
from .slant import get_videos

def get_comments(id : str, n = 10, wait = 0):
    """Get top n comments from video ID"""
    if wait > 0: time.sleep(wait)

    downloader = YoutubeCommentDownloader()
    try: 
        comments = downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={id}", sort_by=SORT_BY_POPULAR)
        print(f"Fetched comments from video id {id}")
        return [comment["text"] for comment in list(islice(comments, n))]
    except Exception as e:
        print(f"Couldn't fetch comments. Error: {e}. Skipping.")
        return None


def get_transcript(id : str, wait = 0):
    """Get transcript from video ID"""
    ytt_api = YouTubeTranscriptApi()
    if wait > 0: time.sleep(wait)

    try:
        t_dict = ytt_api.fetch(id).to_raw_data()
        print(f"Fetched transcript from video id {id}")
        transcript = [section["text"] for section in t_dict]
        return " ".join(transcript)
    except Exception as e:
        print(f"Transcript unavailable. Error {e}. Skipping.")
        return None


if __name__ == "__main__":
    vids = get_videos((-1,1), n=100)

    rows = [
        {
            "id": vid.id,
            "slant": vid.slant,
            "top_comments": get_comments(vid.id, wait = 1),
            "transcript": get_transcript(vid.id, wait = 1)
        }
        for vid in vids
    ]

    df = pd.DataFrame.from_records(rows)
    df.to_pickle(config.SLANT_DIR / "slant_metadata.pkl")
    