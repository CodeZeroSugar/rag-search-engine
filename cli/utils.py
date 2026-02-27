import json
import string
from pathlib import Path


def get_data_file(filename):
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data" / filename
    return data_path


def get_stopwords():
    with open(get_data_file("stopwords.txt"), "r") as f:
        data = f.read()
    lines = data.splitlines()
    return lines


def clean_input(input):
    lower = input.lower()
    trans_table = str.maketrans("", "", string.punctuation)
    return lower.translate(trans_table)


def tokenize(input):
    final_tokens = []
    tokens = input.split()
    stopwords = get_stopwords()
    for token in tokens:
        if token == "" or token in stopwords:
            continue
        final_tokens.append(token)
    return final_tokens


def search_movies(query):
    movie_results = []
    with open(get_data_file("movies.json"), "r") as f:
        data = json.load(f)
    query_tokens = tokenize(clean_input(query))
    for movie in data["movies"]:
        if len(movie_results) == 5:
            break
        found = False
        movie_tokens = tokenize(clean_input(movie["title"]))
        for query_token in query_tokens:
            for movie_token in movie_tokens:
                if query_token in movie_token:
                    movie_results.append(movie["title"])
                    found = True
                    break
            if found:
                break

    return movie_results
