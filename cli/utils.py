import json
from pathlib import Path


def get_movies_path():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data" / "movies.json"
    return data_path


def search_movies(query):
    movie_results = []
    with open(get_movies_path(), "r") as f:
        data = json.load(f)
    for movie in data["movies"]:
        if len(movie_results) == 5:
            break
        if query in movie["title"]:
            movie_results.append(movie["title"])
    return movie_results
