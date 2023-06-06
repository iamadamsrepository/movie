from collections import Counter, defaultdict
import dataclasses
import random
import time
from typing import List
import torch
import pickle
import datetime as dt
from sklearn.neighbors import NearestNeighbors
import pickle

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


# @dataclasses.dataclass
# class BatchedMovieInput:
#     dates: torch.Tensor
#     langs: torch.Tensor
#     deets: torch.Tensor
#     genres: torch.Tensor


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.langs = {v: i for i, v in enumerate(sorted({movies[m].original_language for m in filtered.keys()}))}
        self.one_hots = torch.nn.functional.one_hot(torch.arange(0, 24).to("cuda")).type(torch.FloatTensor)

    def __len__(self):
        return len(filtered)

    def __getitem__(self, id) -> Movie:
        movie_id = embedding_ids[id]
        movie = movies[movie_id]
        return movie

    def collate(self, in_movies: List[Movie]):
        return in_movies
        # dates = torch.Tensor([[time.mktime(m.release_date.timetuple())/1e9] for m in in_movies]).to("cuda")
        # langs = self.one_hots[[self.langs[m.original_language] for m in in_movies]].to("cuda")
        # deets = torch.Tensor([[m.popularity, m.vote_average, m.vote_count] for m in in_movies]).to("cuda")
        # genres = torch.stack([
        #     torch.cat((genre_embeds(torch.tensor([genre_id_embeddings[g] for g in m.genre_ids]).to("cuda")), torch.zeros([12 - len(m.genre_ids),64]).to("cuda")))
        #     for m in in_movies
        # ])
        # batch = BatchedMovieInput(dates, langs, deets, genres)
        # targets = movie_embeds(torch.tensor([id_embeddings[m.tmdb_id] for m in in_movies]).to("cuda"))
        # return batch, in_movies, targets


class Layer(torch.nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_in, n_out)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))

# class DateEncoder(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.linear = Layer(1, 16)

#     def forward(self, date_tensor: torch.Tensor):
#         d_val = time.mktime(d.timetuple())
#         return self.linear(torch.Tensor([d_val]).to("cuda"))


# class LanguageEncoder(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.langs = {}
#         self.one_hots = torch.nn.functional.one_hot(torch.arange(0, 24).to("cuda")).type(torch.FloatTensor)
#         self.linear = Layer(24, 16)

#     def forward(self, lang: str):
#         if lang not in self.langs:
#             lang_num = len(self.langs)
#             self.langs[lang] = lang_num
#         return self.linear(self.one_hots[self.langs[lang]])

# class GenreLayer(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.movie_genre_layers = torch.nn.ParameterDict({
#             str((movie:=movies[movie_id]).tmdb_id): torch.nn.parameter.Parameter(
#                 torch.cat((
#                     torch.rand(len(movie.genre_ids) + 1), 
#                     torch.zeros(13 - len(movie.genre_ids) - 1)
#                 ))
#             )
#             for movie_id in filtered.keys()
#         })
#         self.softmax = torch.nn.Softmax(dim=0)

    # def forward(self, g_embed: torch.Tensor, input: torch.Tensor, in_movies: List[Movie]):
    #     embeds = torch.cat([input.unsqueeze(1), g_embed], dim=1)
    #     weights = self.softmax(torch.stack([self.movie_genre_layers[str(m.tmdb_id)] for m in in_movies]))
    #     return torch.bmm(weights.unsqueeze(1), embeds).squeeze()


# class MovieDataEncoder(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.date_encoder = Layer(1, 16)
#         self.lang_encoder = Layer(24, 16)
#         self.deets_encoder = Layer(3, 32)
#         self.linear_1 = Layer(64, 64)
#         self.linear_2 = Layer(64, 64)
#         self.genre_layer = GenreLayer()

#     def forward(self, batch: BatchedMovieInput, in_movies: List[Movie]):
#         h1 = torch.cat(
#             [
#                 self.date_encoder(batch.dates), 
#                 self.lang_encoder(batch.langs), 
#                 self.deets_encoder(batch.deets)
#             ], 
#             dim = 1
#         )
#         h2 = self.linear_1(h1)
#         h3 = self.linear_2(h2)
#         out = self.genre_layer(batch.genres, h3, in_movies)
#         return out

#     def test(self, movie: Movie) -> List[Movie]:
#         h1 = torch.cat(
#             [
#                 self.date_encoder(movie.release_date), 
#                 self.lang_encoder(movie.original_language), 
#                 self.linear_1(torch.Tensor([movie.popularity, movie.vote_average, movie.vote_count]))
#             ]
#         )
#         h2 = self.linear_2(h1)
#         h3 = self.linear_3(h2)

#         es = movie_embeds.weight.data
#         nbrs = NearestNeighbors(n_neighbors=24, algorithm='ball_tree').fit(es)
#         _, indices = nbrs.kneighbors(torch.unsqueeze(h3.detach(), dim=0))
#         return [movies[embedding_ids[i]]for i in indices.tolist()[0]]

movies: List[Movie]
movies, recommendations = pickle.load(open("movies.pickle", "rb"))
genres = pickle.load(open("genres.pickle", "rb"))
n_genres = len(genres)

l = -1
filtered = recommendations.copy()
while l != len(filtered):
    l = len(filtered)
    for id, rs in filtered.items():
        f = {r for r in rs if r in filtered}
        filtered[id] = f
    filtered = {id: f for id, f in filtered.items() if len(f) >= 4 and movies[id].vote_count > 1000}
    
# def knn_score():
#     es = movie_embeds.weight.data
#     nbrs = NearestNeighbors(n_neighbors=24, algorithm='ball_tree').fit(es)
#     distances, indices = nbrs.kneighbors(es)
#     scores = []
#     for e_id, knns in enumerate(indices):
#         id = embedding_ids[e_id]
#         recs = filtered[id]
#         score = len({embedding_ids[i] for i in knns} & recs) / len(recs)
#         scores.append(score)
#     return sum(scores) / len(scores)


loader = torch.utils.data.DataLoader((data := MovieDataset()), batch_size=64, shuffle=True, collate_fn=data.collate)
movie_embeds = torch.nn.Embedding(len(filtered), 32)
id_embeddings = {id: i for i, id in enumerate(filtered.keys())}
embedding_ids = {i: id for i, id in enumerate(filtered.keys())}
# movie_embeds, id_embeddings = pickle.load(open("movie_embeds.pickle", "rb"))
# embedding_ids = {e_id: id for id, e_id in id_embeddings.items()}
# movie_embeds = movie_embeds.to("cuda")

genre_embeds = torch.nn.Embedding(len(genres), 32)
# genre_id_embeddings = {id: i for i, id in enumerate(genres.keys())}
# genre_embedding_ids = {i: id for i, id in enumerate(genres.keys())}
# genre_embeds, genre_id_embeddings = pickle.load(open("genre_embeds.pickle", "rb"))
# genre_embedding_ids = {e_id: id for id, e_id in genre_id_embeddings.items()}
# genre_embeds = genre_embeds.to("cuda")
genre_movies = defaultdict(set)
for id, movie in movies.items():
    if id in filtered:
        for genre in movie.genre_ids:
            genre_movies[genre].add(id)

loss_func = torch.nn.MSELoss()
for epoch in range(100):
    optimizers = [
        torch.optim.SGD(i, lr=0.1*(0.97**epoch), momentum=0, weight_decay=0.05)
        for i in [movie_embeds.parameters(), genre_embeds.parameters()]
    ]
    in_movies: List[Movie]
    loss_counter = []
    for batch, in_movies in enumerate(loader):
        movie_eids = torch.tensor([id_embeddings[i.tmdb_id] for i in in_movies])
        movie_embs = movie_embeds(movie_eids)
        loss = 0
        for movie, movie_emb in zip(in_movies, movie_embs):
            genre_eids = torch.tensor([list(genres).index(i) for i in movie.genre_ids])
            genre_embs = genre_embeds(genre_eids)
            ...
            movie_embs = movie_emb.repeat(len(genre_eids), 1)
            loss += torch.sigmoid(loss_func(genre_embs, movie_embs))

            # neg_genre_eids = []
            # while len(neg_genre_eids) != len(genre_eids):
            #     if (r := random.randint(0, n_genres-1)) not in genre_eids:
            #         neg_genre_eids.append(r)
            # neg_genre_eids = torch.tensor(neg_genre_eids)
            # neg_genre_embs = genre_embeds(neg_genre_eids)
            # loss -= torch.sigmoid(loss_func(neg_genre_embs, movie_embs))

        loss_counter.append(loss.item())
        loss.backward()
        for o in optimizers:
            o.step()
        for o in optimizers:
            o.zero_grad()

        # norms = torch.norm(genre_embeds.weight, p=2, dim=1).detach().unsqueeze(dim=1)
        # genre_embeds.weight = torch.nn.Parameter(genre_embeds.weight.div(norms.expand_as(genre_embeds.weight)))
        # norms = torch.norm(movie_embeds.weight, p=2, dim=1).detach().unsqueeze(dim=1)
        # movie_embeds.weight = torch.nn.Parameter(movie_embeds.weight.div(norms.expand_as(movie_embeds.weight)))

    print(f"Epoch {epoch}\t\t{sum(loss_counter)/len(loss_counter):.3f}")
    genre_eids = torch.tensor([list(genres).index(i) for i in movie.genre_ids])
    genre_embs = genre_embeds(genre_eids)
    win = []
    for batch, in_movies in enumerate(loader):
        movie_eids = torch.tensor([id_embeddings[i.tmdb_id] for i in in_movies])
        movie_embs = movie_embeds(movie_eids)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(genre_embeds.weight.data)
        distances, indices = nbrs.kneighbors(movie_embs.data)
        for movie, (ds, gs) in zip(in_movies, zip(distances, indices)):
            if not movie.genre_ids:
                ...
            elif len(movie.genre_ids) == 1:
                win.append(len(set(gs) & set(movie.genre_ids)))
            else:
                win.append(len(set(gs) & set(movie.genre_ids))/2)
    print(f"Win Rate\t\t{sum(win)/len(win):.3f}")
    ...
...
# nn = MovieDataEncoder()
# # nn = pickle.load(open("nn_230209_14:16_ep26_0.3341.pickle", "rb"))

# nn = nn.to('cuda')
# loss_func = torch.nn.MSELoss().to('cuda')
# for j in range(1, 301):
#     optimizers = [
#         torch.optim.SGD(i, lr=0.1*(0.97**j))
#         for i in [nn.parameters()]
#     ]
    # optimizer = torch.optim.SGD(nn.parameters(), lr=0.1*(0.97**j))
    # loss_counter = []

    # for i, (batch, in_movies, targets) in enumerate(loader):
    #     output = nn(batch, in_movies)
    #     nn_loss = loss_func(output, targets)
    #     loss = nn_loss
        
    
    # for i, (id, recs) in enumerate(filtered.items()):
    #     movie = movies[id]
    #     # nn.test(movie)

    #     output = nn(movie)
    #     target = movie_embeds(torch.tensor(id_embeddings[id]).to("cuda"))
    #     nn_loss = torch.sum((output - target)**2)
    #     loss = nn_loss

        # target = random.choice(list(recs))
        # t_e = movie_embeds(torch.tensor(id_embeddings[target]))
        # o_e = movie_embeds(torch.tensor(id_embeddings[id]))
        # rec_loss = torch.sum((t_e - o_e)**2)

        # neg_targets = set(filtered) - recs
        # for genre in movie.genre_ids:
        #     neg_targets -= genre_movies[genre]
        # neg_targets = random.choices(list(neg_targets), k=4)
        # nt_es = [movie_embeds(torch.tensor(id_embeddings[i])) for i in neg_targets]
        # neg_loss = torch.sum(torch.tensor([torch.sum((i - o_e)**2) for i in nt_es]))/4

        # movie_genres = movie.genre_ids
        # if movie_genres:
        #     g = random.choice(movie_genres)
        #     g_e = genre_embeds(torch.tensor(genre_id_embeddings[g]))
        #     genre_loss = torch.sum((g_e - o_e)**2)
        # else:
        #     genre_loss = 0

        # loss = nn_loss + rec_loss - 0.5 * neg_loss + 0.5 * genre_loss

    #     loss.backward()
    #     if i % 2 == 0:
    #         for o in optimizers:
    #             o.step()
    #             o.zero_grad()
    #     loss_counter.append(loss)
    # # score = knn_score()
    # avg_loss = sum(loss_counter) / len(loss_counter)
    # loss_counter = []
    # print(f"Epoch {j},\t{avg_loss=:.4f}")
    # # print(f"Epoch {j},\t{score=:.4f},\t{avg_loss=:.4f}")
    # if j % 100 == 0:
    #     ...
    #     pickle.dump(nn, open(f"nn_{dt.datetime.now().strftime('%y%m%d_%H:%M')}_ep{j}_{avg_loss.item():.4f}.pickle", "wb"))
        # pickle.dump((genre_embeds, genre_id_embeddings), open(f"genre_embeds_{dt.datetime.now().strftime('%y%m%d_%H:%M')}_ep{j}_{avg_loss.item():.4f}.pickle", "wb"))
        # pickle.dump((movie_embeds, id_embeddings), open(f"movie_embeds_{dt.datetime.now().strftime('%y%m%d_%H:%M')}_ep{j}_{avg_loss.item():.4f}.pickle", "wb"))
        # print("Saved")

# pickle.dump((genre_embeds, genre_id_embeddings), open("genre_embeds.pickle", "wb"))
# pickle.dump((movie_embeds, id_embeddings), open("movie_embeds.pickle", "wb"))
# print("Saved")
...