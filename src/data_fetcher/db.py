import sqlite3
from dataclasses import astuple, asdict
from models import Video
import pandas as pd
from pathlib import Path

import config
from logger import setup_logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS video (
  id TEXT PRIMARY KEY,
  slant REAL,
  title TEXT,
  channel TEXT,
  description TEXT,
  tags TAGLIST,
  category TEXT,
  blacklist INTEGER DEFAULT 0,
  L INTEGER,
  R INTEGER,
  train INTEGER DEFAULT 0,
  comments TAGLIST
);

CREATE INDEX IF NOT EXISTS ix_video_slant ON video(slant);
CREATE INDEX IF NOT EXISTS ix_video_channel ON video(channel);
"""


def get_connection():
    con= sqlite3.connect(config.DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def build_db():
    logger.info("Building database...")

    with get_connection() as con:
        con.executescript(SCHEMA)

        df = pd.read_csv(config.SLANT_ESTIMATIONS_CSV)
        ids = df["video_id"]
        slants = df["slant"]
        L = df["liberal_landmark_follows"]
        R = df["conservative_landmark_follows"]

        #build from CSV if IDs dont exist
        con.executemany("""
            INSERT OR IGNORE INTO video 
                (id, slant, L, R, train)
            VALUES (?, ?, ?, ?, ?)
            """,
            zip(ids, slants, L, R, [1]*len(df))
        )


def insert_video(vid : Video):
    logger.info(f"Adding video {vid.id}...")

    with get_connection() as con:
        con.execute("""
            INSERT INTO video (
                id, slant, title, channel, description, tags, category,
                blacklist, L, R, train, comments
            )
            VALUES (
                :id, :slant, :title, :channel, :description, :tags, :category,
                :blacklist, :L, :R, :train, :comments
            )
            """,
            asdict(vid)
        )


def insert_videos(vids: list[Video]) -> int:
    logger.info(f"Adding {len(vids)} videos...")

    with get_connection() as con:
        con.executemany(
            """
            INSERT OR IGNORE INTO video (
                id, slant, title, channel, description, tags, category,
                blacklist, L, R, train, comments
            )
            VALUES (
                :id, :slant, :title, :channel, :description, :tags, :category,
                :blacklist, :L, :R, :train, :comments
            )
            """,
            [asdict(v) for v in vids]
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
                blacklist = :blacklist,
                L = :L,
                R = :R,
                train = :train,
                comments = :comments
            WHERE id = :id 
            """,
            [asdict(vid) for vid in vids]
        )


def get_videos_by_id(ids: list[str]) -> list[Video]:
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    sql = f"SELECT * FROM video WHERE id IN ({placeholders})"
    with get_connection() as con:
        rows = con.execute(sql, ids).fetchall()
    logger.info(f"Fetched {len(rows)}/{len(ids)} videos...")
    return [Video(*row) for row in rows]


def get_videos(slant_range : tuple[float,float], exclude : list[str] = [], n = 0, exclude_blacklist=True, train=False) -> list[Video]:
    """ Return videos in slant range. Optionally exclude list of ids. Optionally define n videos to randomly sample """

    params = list(slant_range)
    sql = "SELECT * FROM video WHERE slant BETWEEN ? AND ?"

    if exclude:
        placeholders = ",".join("?" for x in exclude) # question marks
        sql += f" AND id NOT IN ({placeholders})"
        params.extend(exclude)

    if exclude_blacklist:
        sql += f" AND blacklist != 1"

    if train:
        sql += f" AND train == 1"

    if n > 0:
        sql += " ORDER BY RANDOM() LIMIT ?"
        params.append(n)

    with get_connection() as con:
        rows = con.execute(sql, params).fetchall()

    logger.info(f"Fetching videos in slant range {slant_range}...")
    return [Video(*row) for row in rows]


def save_snapshot(path : Path):
    vids = get_videos((-1,1))
    df = pd.DataFrame([asdict(vid) for vid in vids])
    df.to_pickle(path)


def taglist_to_text(lst : list):
    return " -|- ".join(lst)


def text_to_taglist(data):
    return data.decode().split(" -|- ")

logger = setup_logger("Database")

# tags adapter
sqlite3.register_adapter(list, taglist_to_text)
sqlite3.register_converter("TAGLIST", text_to_taglist)

build_db()

