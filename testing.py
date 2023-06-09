import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import json


def get_knn_genre_score():
    genre_series = pd.read_csv(open("genres.csv", "r"), header=None)[0]
    genre_embs: nn.Embedding = pickle.load(open("genre_embeddings.pickle", "rb"))
    movies = pd.read_csv(open("movie_data.csv", "r"))
    movie_embs: nn.Embedding = pickle.load(open("movie_embeddings.pickle", "rb"))

    nbrs = NearestNeighbors(n_neighbors=6, algorithm="ball_tree").fit(
        genre_embs.weight.data
    )
    distances, indices = nbrs.kneighbors(movie_embs.weight.data)
    scores = []
    genre_score_counter = defaultdict(list)
    for id, knns in enumerate(indices):
        movie: pd.Series = movies.loc[id]
        genres = set(json.loads(movie["genres"]))
        knns = {genre_series[i] for i in knns[:len(genres)]}
        score = len(knns & genres) / len(genres)
        scores.append(score)
        for g in genres:
            genre_score_counter[g].append(score)
    genre_scores = {genre: sum(scores) / len(scores) for genre, scores in genre_score_counter.items()}
    genre_count = {genre: len(scores) for genre, scores in genre_score_counter.items()}
    print(sum(scores)/len(scores))

    nbrs = NearestNeighbors(n_neighbors=6, metric="cosine").fit(
        genre_embs.weight.data
    )
    distances, indices = nbrs.kneighbors(movie_embs.weight.data)
    scores = []
    genre_score_counter = defaultdict(list)
    for id, knns in enumerate(indices):
        movie: pd.Series = movies.loc[id]
        genres = set(json.loads(movie["genres"]))
        knns = {genre_series[i] for i in knns[:len(genres)]}
        score = len(knns & genres) / len(genres)
        scores.append(score)
        for g in genres:
            genre_score_counter[g].append(score)
    genre_scores = {genre: sum(scores) / len(scores) for genre, scores in genre_score_counter.items()}
    genre_count = {genre: len(scores) for genre, scores in genre_score_counter.items()}
    print(sum(scores)/len(scores))
    return sum(scores) / len(scores)

def genre_genre_knn_stuff():
    genres = pd.read_csv(open("genres.csv", "r"), header=None)[0]
    genre_embs: nn.Embedding = pickle.load(open("genre_embeddings.pickle", "rb"))

    nbrs = NearestNeighbors(n_neighbors=21, algorithm='ball_tree').fit(genre_embs.weight.data)
    distances, indices = nbrs.kneighbors(genre_embs.weight.data)
    for id, (knns, dists) in enumerate(zip(indices, distances)):
        genre_dists = list(zip(list(genres[knns]), dists))
        ...

def genre_movie_knn_stuff():
    genres = pd.read_csv(open("genres.csv", "r"), header=None)[0]
    genre_embs: nn.Embedding = pickle.load(open("genre_embeddings.pickle", "rb"))
    movies = pd.read_csv(open("movie_data.csv", "r"))
    movie_embs: nn.Embedding = pickle.load(open("movie_embeddings.pickle", "rb"))

    dis_nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(movie_embs.weight.data)
    cos_nbrs = NearestNeighbors(n_neighbors=8, metric="cosine").fit(movie_embs.weight.data)
    a, dis_knns = dis_nbrs.kneighbors(genre_embs.weight.data)
    b, cos_knns = cos_nbrs.kneighbors(genre_embs.weight.data)
    for id, (d_knns, c_knns) in enumerate(zip(dis_knns, cos_knns)):
        print("\n", genres[id])
        for _, movie in movies.loc[d_knns].iterrows():
            print(movie["primaryTitle"], movie["genres"])
        print('\n\n')
        for _, movie in movies.loc[c_knns].iterrows():
            print(movie["primaryTitle"], movie["genres"])
        

def genre_combo_movie_knn_stuff():
    genres = pd.read_csv(open("genres.csv", "r"), header=None)[0]
    genre_embs: nn.Embedding = pickle.load(open("genre_embeddings.pickle", "rb"))
    genre_ids = {genre: id for id, genre in dict(genres).items()}
    movies = pd.read_csv(open("movie_data.csv", "r"))
    movie_embs: nn.Embedding = pickle.load(open("movie_embeddings.pickle", "rb"))

    genre_combos: pd.Series = movies["genres"].value_counts()
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(movie_embs.weight.data)
    for combo, freq in genre_combos.iteritems():
        combo_emb = torch.mean(genre_embs(torch.tensor([genre_ids[i] for i in json.loads(combo)])), dim=0).detach().unsqueeze(0)
        dists, inds = nbrs.kneighbors(combo_emb)
        dists, inds = dists[0], inds[0]
        print("\n", combo, freq)
        for (_, movie), dist in zip(movies.loc[inds].iterrows(), dists):
            print(f"{dist:.2f}", movie["genres"], movie["primaryTitle"])
        ...

def keyword_movie_knn_stuff():
    keywords = pd.read_csv(open("keywords.csv", "r"), header=None)[0]
    keyword_embs: nn.Embedding = pickle.load(open("keyword_embeddings.pickle", "rb"))
    movies = pd.read_csv(open("movie_data.csv", "r"))
    movie_embs: nn.Embedding = pickle.load(open("movie_embeddings.pickle", "rb"))

    nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(movie_embs.weight.data)
    distances, indices = nbrs.kneighbors(keyword_embs.weight.data)
    for id, (knns, dists) in enumerate(zip(indices, distances)):
        print("\n", keywords[id])
        for _, movie in movies.loc[knns].iterrows():
            print(movie["primaryTitle"], movie["keywords"])
        ...

genre_movie_knn_stuff()