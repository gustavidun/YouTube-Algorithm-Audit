import requests
from time import sleep

from config import API_KEY, MAX_API_ERRORS
from models import Video

VIDEOS_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/videos"

api_index : int = 0
error_counter : int = 0 


def request(url : str, params : dict):
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
        print("HTTP error...")
        status = e.response.status_code
        if status == 403:
            print(f"API index {api_index} reached quota.")
            if (api_index + 1 < len(API_KEY)):
                api_index += 1
                params.update({"key": API_KEY[api_index]})
                request(url, params)
            else:
                raise Exception("Quota exceeded.")
        if status == 400:
            print("Bad request.")
            return
    except (requests.Timeout, requests.ConnectionError):
        print("Connection error. Trying again in 5 seconds...")
        sleep(5)
        if error_counter <= MAX_API_ERRORS:
            request(url, params)
            error_counter += 1
        else:
            raise Exception("Max errors reached.")


def get_videos_metadata(*id : str) -> list[Video]:
    if len(id) > 50: #split list 
        [id[i:i + 50] for i in range(0, len(id), 50)]

    video_resp = request(VIDEOS_ENDPOINT, params = {
        "key": API_KEY[api_index],
        "part": "contentDetails, snippet, statistics",
        "id": ",".join(id)
    })

    if not "items" in video_resp:
        print("No items in response")
        return
    
    vids = [
        Video(
            id=item["id"],
            channel=item["snippet"]["channelTitle"],
            title=item["snippet"]["title"],
            description=item["snippet"]["description"],
            category=item["snippet"]["categoryId"],
            tags=item["snippet"]["tags"]
        ) 
        for item in video_resp["items"]
    ]

    return vids

