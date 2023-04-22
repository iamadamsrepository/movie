from collections import Counter, defaultdict
import dataclasses
import random
import time
from typing import Dict, List
import torch
import torch.nn.functional as F
import pickle
import datetime as dt
from sklearn.neighbors import NearestNeighbors
import pickle
from itertools import combinations

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

movies: Dict[str, Movie]
movie_ids: Dict[str, int]
movie_embeddings: torch.nn.Embedding
movies, movie_ids, movie_embeddings = pickle.load(open("movie_embeddings.pickle", "rb"))
id_movies: Dict[int, str] = {v: k for k, v in movie_ids.items()}

genre_ids: Dict[str, int]
genre_embeddings: torch.nn.Embedding
genre_ids, genre_embeddings = pickle.load(open("genre_embeddings.pickle", "rb"))
id_genres: Dict[int, str] = {v: k for k, v in genre_ids.items()}

genre_counter = defaultdict(int)
for movie in movies.values():
    for genre in movie.genre_ids:
        genre_counter[genre] += 1

# nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(movie_embeddings.weight.data)

# distances, indices = nbrs.kneighbors(genre_embeddings.weight.data)
# for id, knns in enumerate(indices):
#     genre = id_genres[id]
#     print("\n", genre)
#     for movie_id in knns:
#         movie = movies[id_movies[movie_id]]
#         print(f"  - {', '.join(movie.genre_ids)}\t\t{movie.title}")
#     ...

# genre_combos = list(combinations(genre_ids.keys(), 2))
# random.shuffle(genre_combos)
# for combo in genre_combos:
#     genre_embs = genre_embeddings(torch.Tensor([genre_ids[genre] for genre in combo]).to(torch.int)).detach()
#     avg_emb = torch.mean(genre_embs, dim=0).unsqueeze(0)
#     distances, indices = nbrs.kneighbors(avg_emb)
#     print("\n", combo)
#     for movie_id in indices.flatten():
#         movie = movies[id_movies[movie_id]]
#         print(f"  - {', '.join(movie.genre_ids)}\t\t{movie.title}")
#     ...

from transformers import BertTokenizer, BertModel

class Bert(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.dropout = torch.nn.Dropout(dropout)
        self.tanh = torch.nn.Tanh()
        self.linear_1 = torch.nn.Linear(768, 64)
        self.linear_2 = torch.nn.Linear(64, 32)

    def trainable_parameters(self):
        return self.linear_1.parameters()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear_1(dropout_output)
        hidden_layer = self.tanh(linear_output)
        dropout_output = self.dropout(hidden_layer)
        linear_output = self.linear_2(dropout_output)
        output = self.tanh(linear_output)
        return output

class Dataset(torch.utils.data.Dataset):
    def __init__(self, genre_ids, genre_embeddings):
        self.genre_ids = genre_ids
        self.id_genres = {v: k for k, v in genre_ids.items()}
        self.genre_embeddings = genre_embeddings
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def __len__(self):
        return len(self.genre_ids)
    
    def __getitem__(self, genre_id):
        return genre_id
    
    def collate(self, genre_ids):
        genres = [self.id_genres[id] for id in genre_ids]
        input_text = " ".join(genres)
        if random.randint(0, 1):
            input_text = input_text.lower()
        input = self.tokenizer(input_text, padding="max_length", max_length=8, truncation=True, return_tensors="pt")
        genre_embs = self.genre_embeddings(torch.Tensor(genre_ids).to(torch.int)).to(torch.float)
        target = torch.mean(genre_embs, dim=0).unsqueeze(0)
        return genres, input, target
    
dataset = Dataset(genre_ids, genre_embeddings)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate)
criterion = F.mse_loss
model = Bert()

for epoch in range(1000):
    optim = torch.optim.SGD(model.trainable_parameters(), lr=0.04*(0.98**epoch), momentum=0.5, weight_decay=0.01)
    loss_counter = []
    model.train()
    for _, input, target in loader:
        output = model(input["input_ids"].squeeze(1), input["attention_mask"])
        loss = criterion(output, target)
        loss_counter.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad()

    print(f"\nEpoch {epoch}")
    print(f"  - Avg Loss {sum(loss_counter)/len(loss_counter):.3f}")
    
    if epoch % 10 == 0:
        model.eval()
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(genre_embeddings.weight.data)
        scores = []
        # for i in range(1, 4):
        #     test_loader = torch.utils.data.DataLoader(dataset, batch_size=i, shuffle=True)
            # for genre, input, target in loader:
            #     output = model(input["input_ids"].squeeze(1), input["attention_mask"])
            #     distances, indices = nbrs.kneighbors(output.detach())
            #     genres = id_genres[indices.squeeze().item()]
        for genre, input, target in test_loader:
            output = model(input["input_ids"].squeeze(1), input["attention_mask"])
            distances, indices = nbrs.kneighbors(output.detach())
            genres = id_genres[indices.squeeze().item()]
            if genre[0] in genres:
                scores.append(1)
            else:
                scores.append(0)
        print(f"  - Test Score {sum(scores)/len(scores):.3f}")