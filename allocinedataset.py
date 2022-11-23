import pandas as pd
import torch

from allocineutil import truncate_if_needed, pad


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = torch.tensor(row["tokens"])

        # FIXME: 512 is the max length of the model; add a parameter to the class
        tokens = truncate_if_needed(tokens, 512)

        return {
            "id": idx,
            "tokens": tokens,
            "text": row["commentaire"],
            "cls_note": torch.tensor(row["cls_note"]),
            "note": torch.tensor(row["note"], dtype=torch.float32)
        }


def collate(data, augment=False):
    tokens, masks = pad([eg["tokens"] for eg in data], 1)
    return {
        "id": [eg["id"] for eg in data],
        "tokens": tokens,
        "masks": masks,
        "cls_note": torch.stack([eg["cls_note"] for eg in data]),
        "note": torch.stack([eg["note"] for eg in data])
    }


def make_loader(set, batch_size, train=False):
    return torch.utils.data.DataLoader(
        set,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=lambda x: collate(x, train),
        pin_memory=True,
        num_workers=4
    )