import pandas as pd

def get_rating_class(rating: float):
    # TODO: à vérifier
    if rating <= 1.5: return 0
    if rating <= 3: return 1
    return 2

def attach_rating_class(df: pd.DataFrame):
    df["cls_note"] = df["note"].apply(get_rating_class)

def attach_train_metadata(df: pd.DataFrame):
    attach_rating_class(df)

def attach_test_metadata(df: pd.DataFrame):
    pass