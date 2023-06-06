from collections import Counter, defaultdict
import dataclasses
import random
import time
from typing import Callable, List
import torch
import torch.nn as nn 
import torch.nn.functional as F
import datetime as dt
from sklearn.neighbors import NearestNeighbors
import pickle
import pandas as pd
import json
import math

EMBEDDING_SIZE = 256

EPOCHS = 5000
BATCH_SIZE = 64
DEVICE = "cuda"

class YearEncoder(nn.Module):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self.linear = nn.Linear(1, EMBEDDING_SIZE, dtype=torch.float64)
        self.mean = mean
        self.std = std

    def forward(self, movie_batch, as_tensor=False):
        if not as_tensor:
            years = torch.tensor([movie["year"] for movie in movie_batch], dtype=torch.float64).unsqueeze(1)
        else:
            years = movie_batch
        norm_years = (years - self.mean) / self.std
        linear_out = self.linear(norm_years)
        out_norm = torch.norm(linear_out, p=2, dim=1).unsqueeze(dim=1).expand_as(linear_out)
        return linear_out.div(out_norm)


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv(open("movie_data.csv", "r"))
        self.genre_series = pd.read_csv(open("genres.csv", "r"), header=None)[0]
        self.genre_ids = {genre: id for id, genre in dict(self.genre_series).items()}
        self.keyword_series = pd.read_csv(open("keywords.csv", "r"), header=None)[0]
        self.keyword_ids = {kw: id for id, kw in dict(self.keyword_series).items()}
        self.init_embeddings()

    def init_embeddings(self):
        self.movie_embs = torch.nn.Embedding(len(self.df), EMBEDDING_SIZE, max_norm=1, dtype=torch.float64)
        self.genre_embs = torch.nn.Embedding(len(self.genre_series), EMBEDDING_SIZE, max_norm=1, dtype=torch.float64)
        self.keyword_embs = torch.nn.Embedding(len(self.keyword_series), EMBEDDING_SIZE, max_norm=1, dtype=torch.float64)
        self.year_encoder = YearEncoder((years := torch.tensor(self.df["year"], dtype=torch.float)).mean(), years.std())
        print(f"Initialised Embeddings, {EMBEDDING_SIZE=}")

    def write_embeddings(self):
        pickle.dump(self.movie_embs, open("movie_embeddings.pickle", "wb"))
        pickle.dump(self.genre_embs, open("genre_embeddings.pickle", "wb"))
        pickle.dump(self.keyword_embs, open("keyword_embeddings.pickle", "wb"))
        pickle.dump(self.year_encoder, open("year_encoder.pickle", "wb"))
        print("Written Embeddings")

    def normalise_embeddings(self):
        emb: nn.Embedding
        for emb in [self.movie_embs, self.genre_embs, self.keyword_embs]:
            emb_norms = torch.norm(emb.weight, p=2, dim=1).detach().unsqueeze(dim=1)
            emb.weight = torch.nn.Parameter(emb.weight.div(emb_norms.expand_as(emb.weight)))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        movie: pd.Series = self.df.loc[i]
        return movie
    
    def collate(self, items: List[pd.Series]):
        movie_embs: torch.Tensor = self.movie_embs(torch.tensor([movie.name for movie in items]))
        genre_targets = []
        keyword_targets = []
        for movie in items:
            genre_embs = self.genre_embs(torch.tensor([self.genre_ids[i] for i in json.loads(movie["genres"])]))
            genre_targets.append(torch.mean(genre_embs, dim=0))
            keyword_embs = self.keyword_embs(torch.tensor([self.keyword_ids[i] for i in json.loads(movie["keywords"])]))
            keyword_targets.append(torch.mean(keyword_embs, dim=0))
        genre_targets = torch.stack(genre_targets)
        keyword_targets = torch.stack(keyword_targets)
        year_targets = self.year_encoder(items)
        return movie_embs.to(DEVICE), genre_targets.to(DEVICE), keyword_targets.to(DEVICE), year_targets.to(DEVICE)
    
    def get_knn_genre_score(self):
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(self.genre_embs.weight.data)
        distances, indices = nbrs.kneighbors(self.movie_embs.weight.data)
        scores = []
        # genre_score_counter = defaultdict(list)
        for id, knns in enumerate(indices):
            movie: pd.Series = self.df.loc[id]
            genres = set(json.loads(movie["genres"]))
            knns = {self.genre_series[i] for i in knns[:len(genres)]}
            score = len(knns & genres) / len(genres)
            scores.append(score)
            # for g in genres:
            #     genre_score_counter[g].append(score)
        # genre_scores = {genre: sum(scores) / len(scores) for genre, scores in genre_score_counter.items()}
        # genre_count = {genre: len(scores) for genre, scores in genre_score_counter.items()}
        return sum(scores) / len(scores)
    
    def get_knn_keyword_score(self):
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(self.keyword_embs.weight.data)
        distances, indices = nbrs.kneighbors(self.movie_embs.weight.data)
        scores = []
        for id, knns in enumerate(indices):
            movie: pd.Series = self.df.loc[id]
            keywords = set(json.loads(movie["keywords"]))
            knns = {self.keyword_series[i] for i in knns[:len(keywords)]}
            score = len(knns & keywords) / len(keywords)
            scores.append(score)
        return sum(scores) / len(scores)
    
    def get_knn_year_score(self):
        year_embs = self.year_encoder(torch.tensor(range(1930, 2023), dtype=torch.float64).unsqueeze(1), as_tensor=True).detach()
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(year_embs)
        distances, indices = nbrs.kneighbors(self.movie_embs.weight.data)
        scores = []
        for id, knns in enumerate(indices):
            movie: pd.Series = self.df.loc[id]
            year = movie["year"]
            knns = {1930 + i for i in knns}
            if year in knns:
                scores.append(1)
            else:
                scores.append(0)
        return sum(scores) / len(scores)


data = MovieDataset()
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data.collate)
n_batches = math.ceil(len(data) / BATCH_SIZE)

for epoch in range(EPOCHS):
    lr=0.5*(0.992**epoch)
    optim = torch.optim.SGD(
        list(data.movie_embs.parameters()) 
        + list(data.genre_embs.parameters()) 
        + list(data.keyword_embs.parameters()) 
        + list(data.year_encoder.parameters())
        , lr=lr
    )
    print(f"\nEpoch {epoch}, {lr=:.2e}")

    if epoch % 20 == 0:
        data.normalise_embeddings()

    loss_counter = []
    for batch, (movie_embs, genre_targets, keyword_targets, year_targets) in enumerate(loader):
        optim.zero_grad()
        print(f"Batch {batch}/{n_batches}", end="\r")
        genre_loss: torch.Tensor = torch.mean(1 - F.cosine_similarity(movie_embs, genre_targets))
        keyword_loss: torch.Tensor = torch.mean(1 - F.cosine_similarity(movie_embs, keyword_targets))
        year_loss: torch.Tensor = torch.mean(1 - F.cosine_similarity(movie_embs, year_targets))
        loss = genre_loss + keyword_loss + year_loss
        loss.backward()
        optim.step()
        loss_counter.append(loss.item())
    
    avg_loss = sum(loss_counter) / len(loss_counter)
    print(f"  - Avg loss        {avg_loss:.3f}")

    if epoch % 4 == 0:
        genre_test_score = data.get_knn_genre_score()
        keyword_test_score = data.get_knn_keyword_score()
        year_test_score = data.get_knn_year_score()
        avg_movie_norm = torch.mean(torch.norm(data.movie_embs.weight.data, p=2, dim=1)).item()
        avg_genre_norm = torch.mean(torch.norm(data.genre_embs.weight.data, p=2, dim=1)).item()
        avg_keyword_norm = torch.mean(torch.norm(data.keyword_embs.weight.data, p=2, dim=1)).item()
        print(f"  - Genre Test      {genre_test_score:.3f}")
        print(f"  - Keyword Test    {keyword_test_score:.3f}")
        print(f"  - Year Test       {year_test_score:.3f}")
        print(f"  - Avg norms       {avg_movie_norm:.3f}, {avg_genre_norm:.3f}, {avg_keyword_norm:.3f}")


data.write_embeddings()