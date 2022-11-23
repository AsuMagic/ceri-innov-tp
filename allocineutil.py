import random

import pandas as pd
import torch as torch


def truncate_if_needed(tokens, max_len):
    if len(tokens) > max_len:
        return torch.cat((tokens[:256], tokens[-256:]))

    return tokens


# suboptimal but easier to manipulate padding
def pad(tensors, pad_symbol: int):
    max_len = max(len(t) for t in tensors)

    for i in range(len(tensors)):
        padding_len = max_len - len(tensors[i])
        left_padding_len = random.randint(0, padding_len)
        right_padding_len = padding_len - left_padding_len

        left_padding = torch.full((left_padding_len,), pad_symbol)
        right_padding = torch.full((right_padding_len,), pad_symbol)

        tensors[i] = torch.cat((left_padding, tensors[i], right_padding), dim=-1)
    
    tensors = torch.stack(tensors)
    masks = tensors.ne(pad_symbol).float()
    
    return tensors, masks


def get_rating_class(rating: float):
    if rating == 0.5: return 0
    if rating == 1.0: return 1
    if rating == 1.5: return 2
    if rating == 2.0: return 3
    if rating == 2.5: return 4
    if rating == 3.0: return 5
    if rating == 3.5: return 6
    if rating == 4.0: return 7
    if rating == 4.5: return 8
    if rating == 5.0: return 9
    raise ValueError(f"invalid rating {rating}, expected between 0 and 5")


def get_3class_rating(rating: float):
    if rating <= 1.5: return 0
    if rating <= 2.0: return 1
    if rating <= 5.0: return 2
    raise ValueError(f"invalid rating {rating}, expected between 0 and 5")


def get_rating_value(value: int):
    if value == 0: return 0.5
    if value == 1: return 1.0
    if value == 2: return 1.5
    if value == 3: return 2.0
    if value == 4: return 2.5
    if value == 5: return 3.0
    if value == 6: return 3.5
    if value == 7: return 4.0
    if value == 8: return 4.5
    if value == 9: return 5.0
    raise ValueError(f"invalid rating {value}, expected integer between 0 and 9 to map to a rating between 0.5 and 5")


def attach_rating_class(df: pd.DataFrame):
    df["cls_note"] = df["note"].apply(get_rating_class)


def attach_train_metadata(df: pd.DataFrame):
    attach_rating_class(df)


def attach_test_metadata(df: pd.DataFrame):
    pass
