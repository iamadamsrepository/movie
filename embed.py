from collections import Counter, defaultdict
import dataclasses
import random
import time
from typing import List
import torch
import torch.nn.functional as F
import pickle
import datetime as dt
from sklearn.neighbors import NearestNeighbors
import pickle

EMBEDDING_SIZE = 32

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

# class MovieDataset(torch.utils.data.Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.langs = {v: i for i, v in enumerate(sorted({movies[m].original_language for m in filtered_recs.keys()}))}
#         self.one_hots = torch.nn.functional.one_hot(torch.arange(0, 24).to("cuda")).type(torch.FloatTensor)

#     def __len__(self):
#         return len(filtered_recs)

#     def __getitem__(self, id) -> Movie:
#         movie_id = embedding_ids[id]
#         movie = movies[movie_id]
#         return movie

#     def collate(self, in_movies: List[Movie]) -> List[Movie]:
#         return in_movies
    
loaded_movies: List[Movie]
loaded_movies, loaded_recommendations = pickle.load(open("movies.pickle", "rb"))
loaded_genres = pickle.load(open("genres.pickle", "rb"))

for movie in loaded_movies.values():
    movie.genre_ids = [loaded_genres[i] for i in movie.genre_ids]
genre_ids = {k: i for i, k in enumerate(loaded_genres.values())}
id_genres = {i: k for i, k in enumerate(loaded_genres.values())}

l = -1
recommendations = loaded_recommendations.copy()
while l != len(recommendations):
    l = len(recommendations)
    for id, rs in recommendations.items():
        f = {r for r in rs if r in recommendations}
        recommendations[id] = f
    recommendations = {id: f for id, f in recommendations.items() if len(f) >= 4 and loaded_movies[id].vote_count > 1000}
recommendations = {loaded_movies[id].title: {loaded_movies[i].title for i in ids} for id, ids in recommendations.items()}
movies = {v.title: v for v in loaded_movies.values() if v.title in recommendations}
movie_ids = {k: i for i, k in enumerate(movies.keys())}
id_movies = {i: k for i, k in enumerate(movies.keys())}

genre_counter = defaultdict(int)
for movie in movies.values():
    for genre in movie.genre_ids:
        genre_counter[genre] += 1

# loader = torch.utils.data.DataLoader((data := MovieDataset()), batch_size=64, shuffle=True, collate_fn=data.collate)
movie_embeddings = torch.nn.Embedding(len(movies), EMBEDDING_SIZE, max_norm=1, dtype=torch.float64)
genre_embeddings = torch.nn.Embedding(len(genre_ids), EMBEDDING_SIZE, max_norm=1, dtype=torch.float64)

def get_rec_embs(movie: Movie, recommendations: dict, embeddings: dict) -> torch.Tensor:
    return embeddings(torch.Tensor([movie_ids[rec] for rec in recommendations[movie.title]]).to(torch.int))

def get_genre_embs(movie: Movie, genre_embeddings: torch.nn.Embedding, genre_ids: dict) -> torch.Tensor:
    return genre_embeddings(torch.Tensor([genre_ids[genre] for genre in movie.genre_ids]).to(torch.int))

def get_knn_score(recommendations: dict, embeddings: dict, id_movies: dict):
    es = embeddings.weight.data
    nbrs = NearestNeighbors(n_neighbors=24, algorithm='ball_tree').fit(es)
    distances, indices = nbrs.kneighbors(es)
    scores = []
    for id, knns in enumerate(indices):
        title = id_movies[id]
        recs = recommendations[title]
        score = len({id_movies[i] for i in knns} & recs) / len(recs)
        scores.append(score)
    return sum(scores) / len(scores)

def get_knn_genre_score(movie_embeddings: torch.nn.Embedding, genre_embeddings: torch.nn.Embedding):
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(genre_embeddings.weight.data)
    distances, indices = nbrs.kneighbors(movie_embeddings.weight.data)
    scores = []
    for id, knns in enumerate(indices):
        movie = movies[id_movies[id]]
        genres = set(movie.genre_ids)
        knns = {id_genres[i] for i in knns[:len(genres)]}
        score = len(knns & genres) / len(genres)
        scores.append(score)
    return sum(scores) / len(scores)


for epoch in range(600):
    print(f"\nEpoch {epoch}")

    optim = torch.optim.SGD(list(movie_embeddings.parameters()) + list(genre_embeddings.parameters()), lr=0.5*(0.99**epoch))
    avgs = []
    m_norms = []
    if epoch % 10 == 0:
        emb_norms = torch.norm(movie_embeddings.weight, p=2, dim=1).detach().unsqueeze(dim=1)
        movie_embeddings.weight = torch.nn.Parameter(movie_embeddings.weight.div(emb_norms.expand_as(movie_embeddings.weight)))
        emb_norms = torch.norm(genre_embeddings.weight, p=2, dim=1).detach().unsqueeze(dim=1)
        genre_embeddings.weight = torch.nn.Parameter(genre_embeddings.weight.div(emb_norms.expand_as(genre_embeddings.weight)))
    for title, movie in movies.items():
        movie_emb = movie_embeddings(torch.Tensor([movie_ids[title]]).to(torch.int)).squeeze()
        genre_emb = get_genre_embs(movie, genre_embeddings, genre_ids)
        avg_genre = torch.mean(genre_emb, dim=0)
        avgs.append(torch.norm(avg_genre, p=2))
        m_norms.append(torch.norm(movie_emb, p=2))
        loss = F.mse_loss(movie_emb, avg_genre)
        loss.backward()
        optim.step()
        optim.zero_grad()
    avg_genre_norm = torch.mean(torch.norm(genre_embeddings.weight.data, p=2, dim=1)).item()
    test_score = get_knn_genre_score(movie_embeddings, genre_embeddings)
    print(f"  - Avg avg             {sum(avgs)/len(avgs):.3f}")
    print(f"  - Avg movie norm      {sum(m_norms)/len(m_norms):.3f}")
    print(f"  - Avg genre norm      {avg_genre_norm:.3f}")
    print(f"  - Test                {test_score:.3f}")


pickle.dump((movies, movie_ids, movie_embeddings), open("movie_embeddings.pickle", "wb"))
pickle.dump((genre_ids, genre_embeddings), open("genre_embeddings.pickle", "wb"))