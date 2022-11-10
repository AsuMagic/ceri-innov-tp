import random
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
import logging

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = torch.tensor(row["tokens"])

        if len(tokens) > 512:
            #logging.warning(f"ugly hack: truncating row {idx} tokens from {len(tokens)} to 512")
            # 0..255 is 256 items, -256..-1 is 256 items
            tokens = torch.cat((tokens[:256], tokens[-256:]))

        return {
            "id": idx,
            "tokens": tokens,
            "text": row["commentaire"],
            "cls_note": torch.tensor(row["cls_note"]),
            "note": torch.tensor(row["note"], dtype=torch.float32)
        }


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