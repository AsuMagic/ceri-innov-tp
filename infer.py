from argparse import ArgumentParser
from model import AllocinePredictor
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--set", type=str, required=True)

parser.add_argument("--output-file", type=str, required=True)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch-size", type=int, default=32)

args = parser.parse_args()

predictor = AllocinePredictor(args.model, args.device)
predictor.model.to(args.device)

df = pd.read_pickle(args.set)

df["pred"] = [
    pred.rating_class
    for pred in predictor(df["commentaire"], batch_size=args.batch_size, use_tqdm=True)
]

with open(args.output_file, "w") as f:
    for i, row in df.iterrows():
        f.write(f"review_{i} {str(row['pred']).replace('.', ',')}\n")
