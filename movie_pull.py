import dataclasses
import datetime as dt
import pickle
import random
import time
from typing import List
import requests
import json

@dataclasses.dataclass
class Movie:
    tmdb_id: int
    title: str
    original_language: str
    overview: str
    poster_path: str
    genre_ids: List[int]
    popularity: float
    release_date: dt.date
    vote_average: float
    vote_count: int


def pull_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key=b510c787118d88e3fa5cb66db417ae99"
    json_data = json.loads(requests.get(url).text)
    return {i['id']: i['name'] for i in json_data["genres"]}


def pull_movie_recommendations(id):
    url = f"https://api.themoviedb.org/3/movie/{id}/recommendations?api_key=b510c787118d88e3fa5cb66db417ae99"
    try:
        json_data = json.loads(requests.get(url).text)
    except:
        print(f"Retrying {url}")
        time.sleep(2)
        return pull_movie_recommendations(id)
    movies = {
        r["id"]: Movie(
            r["id"],
            r["title"],
            r["original_language"],
            r["overview"],
            r["poster_path"],
            r["genre_ids"],
            r["popularity"],
            dt.datetime.strptime(r["release_date"], "%Y-%m-%d").date(),
            r["vote_average"],
            r["vote_count"]
        )
        for r in json_data["results"]
        if r["media_type"] == "movie"
        and r["release_date"] != ""
    }
    return movies

def crawl(
        movies = (m := pull_movie_recommendations(674324)),
        recommendations = {674324: set(m.keys())}
    ):
    movies = movies.copy()
    pulled = set(recommendations.keys())
    recommendations = recommendations.copy()
    while len(pulled) < 10000:
        pull_id = random.sample(movies.keys() - pulled, 1)[0]
        pulled_movies = pull_movie_recommendations(pull_id)
        movies.update(pulled_movies)
        recommendations[pull_id] = set(pulled_movies.keys())
        pulled.add(pull_id)
        print(f"{len(pulled)=}")
    return movies, recommendations

def crawl2(
        movies = (m := pull_movie_recommendations(674324)),
        recommendations = {674324: set(m.keys())}
    ):
    movies = movies.copy()
    recommendations = recommendations.copy()
    most_important_to_pull = sorted([movies[i] for i in set(movies.keys()) - set(recommendations.keys())], key=lambda m: m.vote_count, reverse=True)
    for movie in most_important_to_pull[:2000]:
        pulled_movies = pull_movie_recommendations(movie.tmdb_id)
        movies.update(pulled_movies)
        recommendations[movie.tmdb_id] = set(pulled_movies.keys())
        print(f"{len(recommendations)=}, {movie.title=}")
    return movies, recommendations

genres = pull_genres()
pickle.dump(genres, open("genres.pickle", "wb"))

movies, recommendations = pickle.load(open("movies.pickle", "rb"))
movies, recommendations = crawl2(movies, recommendations)
pickle.dump((movies, recommendations), open("movies.pickle", "wb"))
...
    