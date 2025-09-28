import sqlite3
from dataclasses import astuple, asdict
from models import Video
import pandas as pd
import logging

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS video (
  id TEXT PRIMARY KEY,
  slant REAL,
  title TEXT,
  channel TEXT,
  description TEXT,
  tags TAGLIST,
  category TEXT,
  blacklist INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_video_slant ON video(slant);
CREATE INDEX IF NOT EXISTS ix_video_channel ON video(channel);
"""


def get_connection():
    return sqlite3.connect(config.DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)


def build_db():
    logger.info("Building database...")

    with get_connection() as con:
        con.executescript(SCHEMA)

        df = pd.read_csv(config.SLANT_ESTIMATIONS_CSV)
        ids = df["video_id"]
        slants = df["slant"]

        #build from CSV if IDs dont exist
        con.executemany("""
            INSERT OR IGNORE INTO video 
                (id, slant)
            VALUES (?, ?)
            """,
            zip(ids, slants)
        )


def insert_video(vid : Video):
    logger.info(f"Adding video {vid.id}...")

    with get_connection() as con:
        con.execute("""
            INSERT INTO video
            (id, slant, title, channel, description, tags, category, blacklist)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            astuple(vid)
        )    


def update_videos(vids : list[Video]):
    logger.info(f"Updating {len(vids)} videos...")

    with get_connection() as con:
        con.executemany("""
            UPDATE video                   
            SET slant = :slant,
                title = :title,
                channel = :channel,
                description = :description,
                tags = :tags,
                category = :category,
                blacklist = :blacklist
            WHERE id = :id 
            """,
            [asdict(vid) for vid in vids]
        )


def get_video(id : str):
    logger.info(f"Fetching video {id}...")

    with get_connection() as con:
        row = con.execute("SELECT * FROM video WHERE id = ?", (id,)).fetchone()
        if row is not None:
            vid = Video(*row) 
            return vid
        return None


def get_videos(slant_range : tuple[float,float], exclude : list[str] = [], n = 0, exclude_blacklist=True) -> list[Video]:
    """ Return videos in slant range. Optionally exclude list of ids. Optionally define n videos to randomly sample """

    logger.info(f"Fetching videos in slant range {slant_range}...")

    params = list(slant_range)
    sql = "SELECT * FROM video WHERE slant BETWEEN ? AND ?"

    if exclude:
        placeholders = ",".join("?" for x in exclude) # question marks
        sql += f" AND id NOT IN ({placeholders})"
        params.extend(exclude)

    if exclude_blacklist:
        sql += f" AND blacklist != 1"

    if n > 0:
        sql += " ORDER BY RANDOM() LIMIT ?"
        params.append(n)

    with get_connection() as con:
        rows = con.execute(sql, params)

    return [Video(*row) for row in rows]


def list_to_text(lst : list):
    return ",".join(lst)


def text_to_list(data):
    return data.decode().split(",")


def setup_logger():
    logger = logging.getLogger("Database")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)s [%(levelname)s]: %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch) 
    return logger


logger = setup_logger()

# tags adapter
sqlite3.register_adapter(list, list_to_text)
sqlite3.register_converter("TAGLIST", text_to_list)

build_db()


