import logging
from typing import Literal
import pandas as pd

from .youtube_driver import YouTubeDriver, VideoUnavailableException, PlaybackException
import config
import classifier
from logger import setup_logger
from dataclasses import replace

from data_fetcher import get_videos_by_id, get_videos, get_videos_metadata, scrape_comments, insert_videos
from models import Watch, Video

PuppetState = Literal["init", "training", "drifting", "closed"]

class YTPuppet():
    def __init__(self, id : str, slant : float, target_slant : float, headless : bool = True, train_depth = 100, drift_depth = 200, wt = 30):
        self.ID = id
        self.cur_slant = slant
        self.init_slant = slant
        self.target_slant = target_slant

        self.train_depth = train_depth
        self.drift_depth = drift_depth
        self.wt = wt

        self.cur_state : PuppetState = "init"
        self.history : list[Watch] = []

        self.logger = setup_logger(
            self.ID,
            level=logging.DEBUG,
            file_out=config.LOG_DIR / self.ID
        )

        self.driver = YouTubeDriver
        self.driver_args = {
            "session_dir": config.SESSION_DIR / id,
            "headless": headless,
            "ublock_path": config.UBLOCK_PATH,
            "logger": self.logger
        }

    async def _get_closest_slant(self, recs : list[Video]):
        return min(
            recs,
            key=lambda rec: abs(rec.slant - self.cur_slant) if rec.slant is not None else float("inf")
        )

    async def _get_slants(self, recs : list[Video]):
        # filter out already seen videos
        vid_hist = [w for w in self.history if w.source == "video"] # filter out homepage watches
        rec_old = recs
        recs = [rec for rec in recs if rec.id not in [w.video.id for w in vid_hist]]
        self.logger.info(f"Filtered out {len(rec_old) - len(recs)} already watched videos.")

        fetch = get_videos_by_id([rec.id for rec in recs])
        db_map = {v.id: v for v in fetch}
        missing = [r for r in recs if r.id not in db_map] # not in db
        match = [db_map[r.id] for r in recs if r.id in db_map and db_map[r.id].slant is not None] # in db with slant
        missing_slant = [db_map[r.id] for r in recs if r.id in db_map and db_map[r.id].slant is None] # in db but no slant

        # fill nans with metadata and slant prediction
        missing = await get_videos_metadata(missing)
        missing = [scrape_comments(rec) for rec in missing]
        missing = classifier.predict(missing)
        missing_slant = classifier.predict(missing_slant)

        # insert missing videos in db without slant
        if missing: insert_videos([replace(vid, slant=None) for vid in missing])

        return match + missing + missing_slant


    async def _watch(self, driver : YouTubeDriver, vid : Video, depth, get_slants : bool) -> Watch:
        error_log = []

        try:
            vid, recs = await driver.play_video(vid)
        except VideoUnavailableException as e:
            error_log.append(e)
            self.logger.info(f"Video {vid.id} unavailable.")
            raise
        except Exception as e:
            error_log.append(e)
            raise

        if get_slants: recs = await self._get_slants(recs)

        try: 
            wt = await driver.wait_wt(self.wt)
        except PlaybackException as e:
            error_log.append(e)
            self.logger.info(f"Can't play video {vid.id}.")
        except Exception as e:
            error_log.append(e)
            raise 

        watch = Watch(self.cur_state, self, self.cur_slant, (self.init_slant, self.target_slant), depth, vid, recs, error_log, "video", wt)

        self.history.append(watch)
        if not error_log: self.logger.info(f"Finished watch without error. {watch}")
        else: self.logger.warning(f"Recorded watch with error. {watch}") 

        return watch


    async def train(self, driver : YouTubeDriver, slant_margin = 0.05):
        self.cur_state = "training"

        slant_range = (self.cur_slant-slant_margin, self.cur_slant+slant_margin)

        self.logger.info(f"Fetching train videos in slant range: {slant_range}")
        train_vids = get_videos(
            slant_range=slant_range,
            n=self.train_depth,
            train=True
        )

        for i, vid in enumerate(train_vids):
            try: await self._watch(driver, vid, i + 1, get_slants=False)
            except: continue


    async def drift(self, driver : YouTubeDriver, homepage_freq = 5):
        self.cur_state = "drifting"
        next_vid = await self._get_closest_slant(await driver.get_homepage_recs())

        for i in range(1, self.drift_depth + 1):
            self.cur_slant += (self.target_slant - self.init_slant) / self.drift_depth # drift term

            watch = await self._watch(driver, next_vid, i, get_slants=True)

            if i % homepage_freq == 0:
                recs = await driver.get_homepage_recs()
                recs = await self._get_slants(recs)
                self.history.append(
                    Watch(self.cur_state, self, self.cur_slant, (self.init_slant, self.target_slant), i, None, recs, [], "homepage", None)
                )
                next_vid = await self._get_closest_slant(recs)
            else: 
                next_vid = await self._get_closest_slant(watch.recs)


    async def serialize(self):
        rows = [
            {
                "puppet_id": self.ID,
                "puppet_state": w.state,
                "puppet_slant": w.puppet_slant,
                "puppet_cond": w.puppet_cond,
                "depth": w.depth,
                "video_id": getattr(w.video, "id", None),
                "video_slant": getattr(w.video, "slant", None),
                "recs_id": [r.id for r in w.recs],
                "recs_slant": [r.slant for r in w.recs],
                "errors": w.errors,
                "source": w.source,
                "watch_time": w.watch_time
            }
            for w in self.history
        ]
        return pd.DataFrame.from_records(rows)


    async def run(self):
        self.logger.info(f"Running sock-puppet, {self.ID}")

        async with self.driver(**self.driver_args) as driver:
            await driver.consent_check()

            self.logger.info("Initialising training...")
            await self.train(driver)

            self.logger.info("Initialising drifting...")
            await self.drift(driver)

            self.logger.info("Finished run. Saving...")
            df = await self.serialize()
            path = config.PUPPETS_DIR / f"{self.ID}.pkl"
            df.to_pickle(path)

            self.logger.info(f"Puppet data saved to {path}. Closing...")
            self.cur_state = "closed"
