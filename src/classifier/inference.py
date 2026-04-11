from . import roberta
from models import Video
from logger import setup_logger
from time import time
from config import MODELS_DIR
from lingua import Language, LanguageDetectorBuilder

MODEL_NAME = "gustavidun/roberta-yt-slant-estimator"

detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH).build()
logger = setup_logger("Inference")

def filter_english(vids : list[Video]):
    return [vid for vid in vids if detector.detect_language_of(vid.title) == Language.ENGLISH]

def predict(vids : list[Video]) -> list[float]:
    """ Prediction wrapper """
    vids = [vid for vid in vids if not vid.is_metadata_empty()]
    if not vids: return []
    time_start = time()
    predicts = roberta.predict_videos(MODEL_NAME, vids)
    time_end = time()
    logger.info(f"Predicted {len(vids)} video slants in {time_end - time_start} seconds.")
    return predicts