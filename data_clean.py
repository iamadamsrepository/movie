import pandas as pd

# https://datasets.imdbws.com/

MIN_VOTES_FILTER = 200000

title_df = pd.read_csv("title_basics.tsv", sep='\t')
title_df = title_df[title_df["titleType"] == "movie"][title_df["isAdult"] == False][title_df["startYear"] != "\\N"]
title_df["year"] = title_df["startYear"].astype(int)
title_df = title_df[title_df["year"] < 2023]
title_df.sort_values(by=['year'], inplace=True, ascending=False)
title_df = title_df.drop(columns=["startYear", "endYear", "isAdult", "originalTitle", "titleType"])

ratings_df = pd.read_csv("title_ratings.tsv", sep="\t")
ratings_df = ratings_df[ratings_df["numVotes"] > MIN_VOTES_FILTER]

df = pd.merge(title_df, ratings_df, on='tconst', how='inner')
df.to_csv(open("dataset_movie_data.csv", "w"), index=False)