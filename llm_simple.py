import json
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

df = pd.read_csv(open("movie_data.csv", "r"))

def make_sentence(row):
    return f"movie released in {row['year']}, with genres: {' ,'.join(json.loads(row['genres']))}"

df["sentence"] = df.apply(make_sentence, axis=1)

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = SentenceTransformer('all-mpnet-base-v2')

embeddings = embedder.encode(df["sentence"], convert_to_tensor=True)

def print_closest(sentence):
    embedding = embedder.encode(sentence, convert_to_tensor=True)
    cos_scores = util.cos_sim(embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    print("\n\n======================\n\n")
    print(sentence)
    for score, j in zip(top_results[0], top_results[1]):
        j = j.item()
        print(df.loc[j, "year"], df.loc[j, "genres"], df.loc[j, "primaryTitle"], "(Score: {:.4f})".format(score))   

...
# for i in range(0, 1001, 50):
#     embedding = embeddings[i]

#     cos_scores = util.cos_sim(embedding, embeddings)[0]
#     top_results = torch.topk(cos_scores, k=5)

#     print("\n\n======================\n\n")
#     print(df.loc[i, "primaryTitle"])
#     for score, j in zip(top_results[0], top_results[1]):
#         j = j.item()
#         print(df.loc[j, "year"], df.loc[j, "genres"], df.loc[j, "primaryTitle"], "(Score: {:.4f})".format(score))

