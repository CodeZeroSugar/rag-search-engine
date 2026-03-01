import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_data_file(filename):
    data_path = PROJECT_ROOT / "data" / filename
    return data_path


def get_stopwords():
    with open(get_data_file("stopwords.txt"), "r") as f:
        data = f.read()
    lines = data.splitlines()
    return lines


def get_stem(input):
    stemmer = PorterStemmer()
    return stemmer.stem(input)


def clean_input(input):
    lower = input.lower()
    trans_table = str.maketrans("", "", string.punctuation)
    return lower.translate(trans_table)


def tokenize(input):
    final_tokens = []
    stopwords = get_stopwords()
    tokens = input.split()
    for token in tokens:
        t = clean_input(token)
        if t == "" or t in stopwords:
            continue
        final_tokens.append(get_stem(t))
    return final_tokens


def search_movies(query):
    movie_results = []
    with open(get_data_file("movies.json"), "r") as f:
        data = json.load(f)
    query_tokens = tokenize(query)
    for movie in data["movies"]:
        if len(movie_results) == 5:
            break
        found = False
        movie_tokens = tokenize(movie["title"])
        for query_token in query_tokens:
            for movie_token in movie_tokens:
                if query_token in movie_token:
                    movie_results.append(movie["title"])
                    found = True
                    break
            if found:
                break

    return movie_results
