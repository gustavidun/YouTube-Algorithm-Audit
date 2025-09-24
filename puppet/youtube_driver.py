from patchright.async_api import Page, Playwright, BrowserContext, async_playwright
from pathlib import Path
import re 
import asyncio
import logging

class VideoUnavailableException(Exception):
    pass

class YouTubeDriver():
    """Async context manager for interacting with YouTube using Playwright"""
    def __init__(self, session_dir : Path, headless : bool, ublock_path : Path, logger : logging.Logger):
        self.session_dir = session_dir
        self.headless = headless
        self.ublock_path = ublock_path
        self.logger = logger

        self._page : Page | None = None
        self._context : BrowserContext | None = None
        self._playwright : Playwright | None = None


    async def __aenter__(self):
        """ Launch session with persistent context """
        self._playwright = await async_playwright().start()
        self.session_dir.mkdir(exist_ok=True) #create dir if not exists

        self._context = await self._playwright.chromium.launch_persistent_context(
            self.session_dir,
            headless=self.headless,
            channel="chrome",
            no_viewport=True,
            args=[
                f"--disable-extensions-except={self.ublock_path.absolute()}",
                f"--load-extension={self.ublock_path.absolute()}",
            ]
        )
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

        session = await self._context.new_cdp_session(self._page)
        info = await session.send("Browser.getVersion")
        self.logger.info("Initialising YouTube Driver. Chrome version:", info["product"])

        await self._page.goto("https://www.youtube.com")

        return self


    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._context: await self._context.close()
        finally:
            if self._playwright: await self._playwright.stop()


    async def consent_check(self):
        """ Accept cookie consent prompt """
        assert self._page

        consent_btn = self._page.locator('button[aria-label*="Accept the use of cookies"]')
        if await consent_btn.count() > 0:
            self.logger.info("Accepting cookies...")
            await consent_btn.press("Enter")
            await asyncio.sleep(1)
            await self._page.reload()


    async def watch(self, id : str, time : float):
        assert self._page

        url = f"https://www.youtube.com/watch?v={id}"
        await self._page.goto(url)
        await self._page.wait_for_load_state("domcontentloaded")

        # check for video errors
        error = self._page.locator("yt-player-error-message-renderer")
        if await error.count() > 0:
            self.logger.warning("Video is unavailable.")
            raise VideoUnavailableException()

        # check for title element
        title_elem = self._page.locator("h1.ytd-watch-metadata yt-formatted-string")
        if await title_elem.count() < 1:
            raise VideoUnavailableException()
        
        title = await title_elem.get_attribute("title")
        self.logger.info(f"Playing video: {title}")

        #monitor playback time
        await asyncio.sleep(time)
        wt = 0
        while wt <= time:
            wt = await self._page.evaluate("document.querySelector('video')?.currentTime ?? 0") # get player time

            if wt == 0: # if playback stalled, try hitting play again
                await self._page.keyboard.press("k")

            await asyncio.sleep(1)
            self.logger.debug(f"Playback time: {wt}")

        #get up next video ids
        thumbs = await self._page.locator("a.yt-lockup-metadata-view-model__title").all()
        urls = [await x.get_attribute("href") for x in thumbs]
        ids = [re.search(r"/watch\?v=([a-zA-Z0-9_-]{11})", x).group(1) for x in urls] # extract ids
        return ids    