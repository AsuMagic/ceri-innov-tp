import pandas as pd

movie_ids = set()


def add_dataset(path):
    df = pd.read_pickle(path)
    for movie_id in df["movie"]:
        movie_ids.add(movie_id)


for path in "dataset/dev.bin.zst", "dataset/test.bin.zst", "dataset/train.bin.zst":
    add_dataset(path)

print("\n".join(str(movie_id) for movie_id in movie_ids))
