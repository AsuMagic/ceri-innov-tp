from argparse import ArgumentParser
import re
import html
import pandas as pd
import logging

review_parser = re.compile(r"<movie>(.*?)<\/movie>\s*<review_id>(.*?)<\/review_id>\s*<name>(.*?)<\/name>\s*<user_id>(.*?)<\/user_id>\s*<note>(.*?)<\/note>\s*<commentaire>(.*?)<\/commentaire>", re.DOTALL)
test_review_parser = re.compile(r"<movie>(.*?)<\/movie>\s*<review_id>(.*?)<\/review_id>\s*<name>(.*?)<\/name>\s*<user_id>(.*?)<\/user_id>\s*<commentaire>(.*?)<\/commentaire>", re.DOTALL)

def load_dataset(xml_path, is_test: bool):
    logging.info(f"parsing XML from {xml_path}")

    parser = test_review_parser if is_test else review_parser

    columns = ["movie", "review_id", "name", "user_id", "note", "commentaire"]
    if is_test:
        columns.remove("note")

    with open(xml_path, "r+") as f:
        df = pd.DataFrame.from_records(
            (match.groups() for match in parser.finditer(f.read())),
            columns=columns
        )

    logging.info(f"parsing columns")
    df["name"]        = df["name"].apply(html.unescape)
    df["movie"]       = df["movie"].astype(int)
    df["review_id"]   = df["review_id"].str[7:].astype(int)
    if not is_test:
        df["note"] = df["note"].apply(lambda note: float(note.replace(",", ".")))
    df["commentaire"] = df["commentaire"].apply(html.unescape)
    df.set_index("review_id", inplace=True)

    logging.debug(f"summary of extracted table:\n{df}")

    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("in_xml", type=str)
    parser.add_argument("--pickle-to", type=str)
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    df = load_dataset(args.in_xml, is_test=args.test)

    if args.pickle_to is not None:
        logging.info(f"pickling to {args.pickle_to}")
        df.to_pickle(args.pickle_to)