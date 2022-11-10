import asyncio
import requests
import logging
import re
import json
import os
from pathlib import Path
from tqdm import tqdm

MAX_SUBSEQUENT_ERRORS = 5
MIN_INTERVAL_SECS = 1

re_flags = re.DOTALL | re.MULTILINE
match_desc_json = re.compile(r'<script type="application\/ld\+json">\s*({.*?})\s*<\/script>', re_flags)

html_root = Path("dataset/scrap/raw/")
json_root = Path("dataset/scrap")

os.makedirs(html_root, exist_ok=True)

with open("dataset/scrap/movieids.txt", "r+") as f:
    movie_ids = [int(movie_id) for movie_id in f.read().split("\n") if len(movie_id) != 0]

async def main(movie_ids):
    subsequent_errors = 0
    loop = asyncio.get_event_loop()

    for movie_id in (pbar := tqdm(movie_ids)):
        if subsequent_errors > MAX_SUBSEQUENT_ERRORS:
            logging.error(f"Failed for {subsequent_errors} requests in a row. Aborting.")

        pbar.set_description(f"{movie_id}")

        html_path = html_root / f"{movie_id}.html"
        json_path = json_root / f"{movie_id}.json"
        url = f"https://www.allocine.fr/film/fichefilm_gen_cfilm={movie_id}.html"

        if json_path.exists():
            subsequent_errors = 0
            continue

        r, _ = await asyncio.gather(
            loop.run_in_executor(None, requests.get, f"https://www.allocine.fr/film/fichefilm_gen_cfilm={movie_id}.html"),
            asyncio.sleep(MIN_INTERVAL_SECS)
        )

        r = requests.get(url)

        if r.status_code != 200:
            logging.warning(f"Fetch of {url} failed with status code {r.status_code}")
            subsequent_errors += 1
            continue

        assert r.status_code == 200

        matches = match_desc_json.findall(r.text)
        assert len(matches) == 1

        with open(html_path, "w+") as f:
            f.write(r.text)

        try:
            match = matches[0]
            _match_json = json.loads(match, strict=False) # just for validation in case something went wrong
        except Exception as e:
            logging.warning(f"Parsing {html_path} failed with {e}!")
            subsequent_errors += 1
            continue

        with open(json_path, "w+") as f:
            f.write(matches[0])

        subsequent_errors = 0

asyncio.run(main(movie_ids))