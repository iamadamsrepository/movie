import requests
import json
import pandas as pd
import math

def get_imdb_data(tconst):
    url = f"https://www.imdb.com/title/{tconst}/"
    headers = {
        "Host": "www.imdb.com",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.imdb.com/",
    }

    # Cookie: uu=eyJpZCI6InV1YTkxZTdlYTY4ZWMwNDY5NjhhOGYiLCJwcmVmZXJlbmNlcyI6eyJmaW5kX2luY2x1ZGVfYWR1bHQiOmZhbHNlfX0=; session-id=137-2666146-3798704; session-id-time=2082787201l; csm-hit=adb:adblk_no&t:1685626930576&tb:QV4P0CXKF5S9R96XPSGG+s-QV4P0CXKF5S9R96XPSGG|1685626930576; ubid-main=131-6718730-2532010; session-token=g4B67XGZj2JKYxs0CLSaLDUnSAQDgxgk1Uru+Dr11tSU4BCa9OxsCkl9ORmEEaTC46f6alYHXA5AtIVBIgZIR6yWxFWEYqNA8M9EChLKqtAK4WmGq+lCRWaQnZ28VytSOrFONBTS4mEPELz++4P8eZ4v16n1ic/nGHyCBz2nryAYRfYirB4k5VHE0R8JOzG2Fs+zVWu6niiSQuHi3dqygA==
    # Upgrade-Insecure-Requests: 1
    # Sec-Fetch-Dest: document
    # Sec-Fetch-Mode: navigate
    # Sec-Fetch-Site: same-origin
    # Sec-Fetch-User: ?1
    # TE: trailers}
    resp = requests.get(url, headers=headers)
    try:
        data = json.loads(resp.text.split("<script id=\"__NEXT_DATA__\" type=\"application/json\">")[1].split("</script>")[0])["props"]["pageProps"]
    except:
        ...
    return data

df = pd.read_csv(open("scraped_movie_data.csv", "r"))

for i, row in df.iterrows():
    if isinstance(row["scraped_data"], str):
        continue
    imdb_data = get_imdb_data(row["tconst"])
    df.loc[i, "scraped_data"] = json.dumps(imdb_data)
    if i % 10 == 0:
        df.to_csv(open("scraped_movie_data.csv", "w"), index=False)
df.to_csv(open("scraped_movie_data.csv", "w"), index=False)

def get_content_rating(data):
    try:
        return data["aboveTheFoldData"]["certificate"]["rating"]
    except:
        return None
    
def get_genres(data):
    return json.dumps([g["text"] for g in data["aboveTheFoldData"]["genres"]["genres"]])

def get_keywords(data):
    return json.dumps([k["node"]["text"] for k in data["aboveTheFoldData"]["keywords"]["edges"]])

def get_metascore(data):
    try:
        return data["aboveTheFoldData"]["metacritic"]["metascore"]["score"]
    except:
        return None

def get_countries(data):
    return json.dumps([i["id"] for i in data["aboveTheFoldData"]["countriesOfOrigin"]["countries"]])

df["content_rating"] = df["scraped_data"].apply(
    lambda text: get_content_rating(json.loads(text))
)
df["genres"] = df["scraped_data"].apply(
    lambda text: get_genres(json.loads(text))
)
df["keywords"] = df["scraped_data"].apply(
    lambda text: get_keywords(json.loads(text))
)
df["metascore"] = df["scraped_data"].apply(
    lambda text: get_metascore(json.loads(text))
)
df["countries"] = df["scraped_data"].apply(
    lambda text: get_countries(json.loads(text))
)
df = df.drop(columns=["scraped_data"])
df.to_csv(open("movie_data.csv", "w"), index=False)

genres = pd.Series(sorted(list(set(sum(map(json.loads, list(df["genres"])), [])))))
genres.to_csv(open("genres.csv", "w"), index=False, header=None)
keywords = pd.Series(sorted(list(set(sum(map(json.loads, list(df["keywords"])), [])))))
keywords.to_csv(open("keywords.csv", "w"), index=False, header=None)
...