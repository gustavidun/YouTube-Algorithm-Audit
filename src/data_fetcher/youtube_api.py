import requests
from time import sleep

import config
from models import Video

from itertools import islice
from youtube_comment_downloader import *

from youtube_transcript_api import YouTubeTranscriptApi
import time
import json

VIDEOS_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/videos"
COMMENTS_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/commentThreads"

api_index : int = 0
error_counter : int = 0 


async def request(url : str, params : dict):
    global api_index, error_counter
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        if resp:
            error_counter = 0
            return resp.json()
        else:
            print("No response.")
            return
    except requests.HTTPError as e:
        status = e.response.status_code
        err = json.loads(e.response.content)
        reason = err["error"]["errors"][0]["reason"]
        print(f"HTTP error... {reason}")
        if status == 403 and reason == "quotaExceeded":
            print(f"API index {api_index} reached quota.")
            if (api_index + 1 < len(config.API_KEY)):
                api_index += 1
                params.update({"key": config.API_KEY[api_index]})
                request(url, params)
            else:
                raise Exception("Quota exceeded.")
        if status == 400:
            print("Bad request.")
            return
    except (requests.Timeout, requests.ConnectionError):
        print("Connection error. Trying again in 5 seconds...")
        sleep(5)
        if error_counter <= config.MAX_API_ERRORS:
            request(url, params)
            error_counter += 1
        else:
            raise Exception("Max errors reached.")


async def get_videos_metadata(vids : list[Video]) -> list[Video]:
    if len(vids) > 50: #split list 
        raise Exception("API can only process 50 videos at a time.")

    video_resp = await request(VIDEOS_ENDPOINT, params = {
        "key": config.API_KEY[api_index],
        "part": "contentDetails, snippet, statistics",
        "id": ",".join([vid.id for vid in vids])
    })

    if not "items" in video_resp:
        print("No items in response")
        return
    
    vids = [
        Video(
            id=item["id"],
            slant=next((v for v in vids if v.id == item["id"]), None).slant,
            channel=item["snippet"]["channelTitle"],
            title=item["snippet"]["title"],
            description=item["snippet"].get("description", None),
            category=item["snippet"].get("categoryId", None),
            tags=item["snippet"].get("tags", [])
        ) 
        for item in video_resp["items"]
    ]

    return vids


async def get_comments(vid : Video):
    """ Get comments using YouTube Data API """    
    video_resp = await request(COMMENTS_ENDPOINT, params = {
        "key": config.API_KEY[api_index],
        "part": "snippet",
        "videoId": vid.id,
        "order": "relevance",
        "textFormat": "plainText"
    })

    if not video_resp:
        print("No response")
        return

    if not "items" in video_resp:
        print("No items in response")
        return

    vid.comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in video_resp["items"]] 

    return vid


def scrape_comments(vid : Video, n = 20, wait = 0.05):
    """Scrape comments from unofficial YouTube API"""
    downloader = YoutubeCommentDownloader()
    comments = list(islice(
        downloader.get_comments_from_url(
            f'https://www.youtube.com/watch?v={vid.id}',
            sort_by=SORT_BY_POPULAR,
            sleep=wait
        ),
        n
    ))
    vid.comments = [comment["text"] for comment in comments]
    print("scraped comments")
    return vid 


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

