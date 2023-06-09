import json
import uvicorn
from fastapi import FastAPI
import logging
from logging.config import dictConfig
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import math

class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "api"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    # Logging config
    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }

dictConfig(LogConfig().dict())
logger = logging.getLogger("api") 

app = FastAPI()
df = pd.read_csv(open("movie_data.csv"))
embeddings = pickle.load(open("movie_embeddings_simple.pickle", "rb"))
nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(embeddings)
embedder = SentenceTransformer('all-mpnet-base-v2')

def movie_series_to_dict(series: pd.Series) -> dict:
    out = series.to_dict()
    out = {k: None if not isinstance(v, str) and math.isnan(v) else v for k, v in out.items()}
    for k in ["genres", "keywords", "countries"]:
        out[k] = json.loads(out[k])
    return out


@app.get("/")
async def root(sentence: str):
    sentence_embedding = embedder.encode(sentence, convert_to_tensor=True)
    dists, indices = nbrs.kneighbors(sentence_embedding.unsqueeze(0))
    movie_output = [
        movie_series_to_dict(df.loc[i])
        for _, i in zip(dists[0], indices[0])
    ]
        
    return movie_output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)