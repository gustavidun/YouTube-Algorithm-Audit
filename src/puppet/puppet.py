import logging
from typing import Literal
import pandas as pd

from .youtube_driver import YouTubeDriver, VideoUnavailableException
import config

from data_fetcher import get_video, get_videos, update_videos
from models import Watch, Video

PuppetState = Literal["init", "training", "drifting", "closed"]

class YTPuppet():
    def __init__(self, id : str, slant : float, target_slant : float, headless : bool = True):
        self.ID = id
        self.cur_slant = slant
        self.target_slant = target_slant

        self.cur_state : PuppetState = "init"
        self.history : list[Watch] = []

        self.setup_logger()

        self.driver = YouTubeDriver
        self.driver_args = {
            "session_dir": config.SESSION_DIR / id,
            "headless": headless,
            "ublock_path": config.UBLOCK_PATH,
            "logger": self.logger
        }


    def setup_logger(self):
        self.logger = logging.getLogger(self.ID)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(config.LOG_DIR / self.ID, encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


    async def watch(self, driver : YouTubeDriver, vid : Video, wt : int) -> Watch:
        vid, recs = await driver.watch(vid, wt)

        for rec in recs:
            rec_vid = get_video(rec.id)
            if rec_vid is not None: rec.slant = rec_vid.slant

        watch = Watch(self.cur_state, self, self.cur_slant, len(self.history) + 1, vid, recs)
        self.history.append(watch)

        self.logger.info(f"Finished watch. {watch}")

        return watch


    async def train(self, driver : YouTubeDriver, slant_margin = 0.2, depth = 100, wt = 30):
        self.cur_state = "training"

        slant_range = (self.cur_slant-slant_margin, self.cur_slant+slant_margin)

        self.logger.info(f"Fetching train videos in slant range: {slant_range}")
        train_vids = get_videos(
            slant_range=slant_range,
            n=depth
        )

        blacklist = []
        for vid in train_vids:
            try:
                await self.watch(driver, vid, wt)
            except VideoUnavailableException: #add to blacklist if unavailable
                vid.blacklist = True
                blacklist.append(vid)

        update_videos(blacklist)


    async def drift(self, driver : YouTubeDriver, seed_margin = 0.2, depth = 200, wt = 30):
        self.cur_state = "drifting"

        slant_range = (self.cur_slant-seed_margin, self.cur_slant+seed_margin)
        self.logger.info(f"Fetching drift seed video in slant range: {slant_range}")

        next_vid = get_videos(
            slant_range=slant_range,
            exclude=[watch.video.id for watch in self.history],
            n=1
        )[0]

        for i in range(depth):
            try:
                watch = await self.watch(driver, next_vid, wt)
            except VideoUnavailableException: #skip if unavailable
                continue
            
            self.cur_slant = (self.target_slant - self.cur_slant) / depth # drift term

            next_vid = min( # select closest slant
                watch.recs,
                key=lambda rec: abs(rec.slant - self.cur_slant) if rec.slant is not None else float("inf")
            )
            self.logger.info(f"Up next video slant: {next_vid.slant}")


    async def serialize(self):
        rows = [
            {
                "puppet_id": self.ID,
                "puppet_state": w.state,
                "puppet_slant": w.puppet_slant,
                "depth": w.depth,
                "video_id": w.video.id,
                "video_slant": w.video.slant,
                "recs_id": [r.id for r in w.recs],
                "recs_slant": [r.slant for r in w.recs],
            }
            for w in self.history
        ]
        return pd.DataFrame.from_records(rows)


    async def run(self):
        self.logger.info(f"Running sock-puppet, {self.ID}")

        async with self.driver(**self.driver_args) as driver:
            await driver.consent_check()

            self.logger.info("Initialising training...")
            await self.train(driver, wt=5, depth=10)

            self.logger.info("Initialising drifting...")
            await self.drift(driver, wt=5, depth=10)

            self.logger.info("Finished run. Saving...")
            df = await self.serialize()
            path = config.PUPPETS_DIR / f"{self.ID}.csv"
            df.to_csv(path)

            self.logger.info(f"Puppet data saved to {path}. Closing...")
            self.cur_state = "closed"




