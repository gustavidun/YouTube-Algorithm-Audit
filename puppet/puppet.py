import asyncio
import random
import logging

from .youtube_driver import YouTubeDriver, VideoUnavailableException
from . import config

from classifier import get_slant, get_video_ids, write_blacklist, load_blacklist


class YTPuppet():
    def __init__(self, id : str, slant : float, headless : bool = True, stream_output = True):
        self.ID = id
        self.cur_slant = slant

        self.logger = logging.getLogger(self.ID)
        self.logger.setLevel(logging.DEBUG)
        log_path = config.LOG_DIR / self.ID

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.history = []

        self.driver = YouTubeDriver
        self.driver_args = {
            "session_dir": config.SESSION_DIR / id,
            "headless": headless,
            "ublock_path": config.UBLOCK_PATH,
            "logger": self.logger
        }


    async def train(self, driver : YouTubeDriver, slant_margin = 0.2, depth = 100, wt = 30):
        slant_range = (self.cur_slant-slant_margin, self.cur_slant+slant_margin)
        self.logger.info(f"Fetching train videos in slant range: {slant_range}")
        train_pool = get_video_ids(
            slant_range=slant_range,
            exclude=load_blacklist()
        )
        train_ids = random.sample(train_pool, k=depth)
        blacklist = []

        for id in train_ids:
            try:
                await driver.watch(id, wt)
                self.history.append(id)
                print(f"Video slant: {get_slant(id)}")
            except VideoUnavailableException: #add to blacklist if unavailable
                blacklist.append(id)

        write_blacklist(blacklist)


    async def drift(self, driver : YouTubeDriver, seed_margin = 0.2, depth = 200, wt = 30):
        slant_range = (self.cur_slant-seed_margin, self.cur_slant+seed_margin)
        self.logger.info(f"Fetching drift seed video in slant range: {slant_range}")
        seed_pool = get_video_ids(
            slant_range=slant_range,
            exclude=self.history + load_blacklist()
        )
        next_vid = random.sample(seed_pool, k=1)[0]

        for i in range(depth):
            try:
                recs = await driver.watch(next_vid, wt)
            except VideoUnavailableException: #skip if unavailable
                continue

            self.history.append(next_vid)
            slants = [get_slant(x) for x in recs]
            closest = min( # select closest slant
                slants,
                key=lambda x: abs(x - self.cur_slant) if x is not None else float("inf")
            )
            next_vid = recs[slants.index(closest)]
            self.logger.info(f"Up next video slant: {closest}")


    async def run(self):
        self.logger.info(f"Running sock-puppet, {self.ID}")

        async with self.driver(**self.driver_args) as driver:
            await driver.consent_check()

            self.logger.info("Initialising training...")
            await self.train(driver, wt=5, depth=10)

            self.logger.info("Initialising drifting...")
            await self.drift(driver, wt=5, depth=10)


async def main():
    agent = YTPuppet("bob", 0.7)
    agent2 = YTPuppet("charlotte", 0.5)    

    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(agent.run())
        task2 = tg.create_task(agent2.run())    


if __name__ == "__main__":
    asyncio.run(main())

