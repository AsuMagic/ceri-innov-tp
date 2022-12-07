from transformers import CamembertTokenizerFast
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("camembert_model", type=str)
parser.add_argument("in_model", type=str)
parser.add_argument("out_model", type=str)
args = parser.parse_args()

df = pd.read_pickle(args.in_model)

tokenizer = CamembertTokenizerFast.from_pretrained(args.camembert_model)

df["tokens"] = df["commentaire"].progress_apply(lambda text:
    tokenizer.encode(text, add_special_tokens=True))

df.to_pickle(args.out_model)
