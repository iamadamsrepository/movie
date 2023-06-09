from typing import List
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EMBEDDING_SIZE = 512

EPOCHS = 150
BATCH_SIZE = 64
DEVICE = "cpu"
DROPOUT = 0.5


class Layer(nn.Module):
    def __init__(self, in_features, out_features, activation="relu") -> None:
        super().__init__()
        assert activation in {"relu", "norm"}
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(in_features, out_features)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "norm":
            self.activation = self.normalise_embedding

    def normalise_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        emb_norm = (
            torch.norm(emb, p=2, dim=1).unsqueeze(dim=1).expand_as(emb)
        )
        return emb.div(emb_norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 1:
            normed = self.batch_norm(x)
        else:
            normed = x
        dropped = self.dropout(normed)
        linear_out = self.linear(dropped)
        out = self.activation(linear_out)
        return out


class SentenceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = SentenceTransformer('all-MiniLM-L6-v2')
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.linear_1 = Layer(384, 384)
        self.linear_2 = Layer(384, EMBEDDING_SIZE)
        self.linear_3 = Layer(EMBEDDING_SIZE, EMBEDDING_SIZE, activation="norm")

    def trainable_parameters(self) -> List[nn.Parameter]:
        return sum(map(list, [self.linear_1.parameters(), self.linear_2.parameters(), self.linear_3.parameters()]), [])

    def forward(self, sentences):
        out_1 = self.pretrained.encode(sentences, convert_to_tensor=True).to(DEVICE)
        out_2 = self.linear_1(out_1)
        out_3 = self.linear_2(out_2)
        out_4 = self.linear_3(out_3)
        return out_4
    
    def save(self):
        pickle.dump(self, open("sentence_embedder.pickle", "wb"))

   
class PhraseDataset(Dataset):
    def __init__(self):
        self.genre_series = pd.read_csv(open("genres.csv", "r"), header=None)[0]
        self.genre_embeddings = pickle.load(open("genre_embeddings.pickle", "rb"))
        self.keyword_series = pd.read_csv(open("keywords.csv", "r"), header=None)[0]
        self.keyword_embeddings = pickle.load(open("keyword_embeddings.pickle", "rb"))

        self.phrases = pd.concat([self.genre_series, self.keyword_series]).reset_index()[0]
        self.embeddings = torch.cat([self.genre_embeddings.weight.data, self.keyword_embeddings.weight.data])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, x):
        name = self.phrases[x]
        embedding: torch.Tensor = self.embeddings[x]
        return name, embedding.to(DEVICE)


model = SentenceEmbedder().to(DEVICE)
optimizer = torch.optim.Adam(model.trainable_parameters(), lr=0.001)

dataset = PhraseDataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")
    model.train()
    loss_counter = []
    for i, (genre_names, genre_embeddings) in enumerate(loader):
        optimizer.zero_grad()
        output = model(genre_names)
        output = F.dropout(output, p=DROPOUT)
        genre_targets = genre_embeddings * (output != 0)
        loss = torch.mean(1 - F.cosine_similarity(output, genre_embeddings))
        loss.backward()
        optimizer.step()
        loss_counter.append(loss.item())
    print(f"  - Avg Loss    {sum(loss_counter)/len(loss_counter):.3f}")

    if epoch % 10 == 0:
        model.eval()
        nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(dataset.embeddings)
        scores = []
        for genre_name, _ in test_loader:
            output = model(genre_name)
            _, indices = nbrs.kneighbors(output.detach().to("cpu"))
            knn_genres = set(dataset.phrases[indices[0]])
            if genre_name[0] in knn_genres:
                scores.append(1)
            else:
                scores.append(0)
        print(f"  - Test Score  {sum(scores)/len(scores):.3f}")

model.save()