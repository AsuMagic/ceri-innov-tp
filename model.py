import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentPredictor(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()

        self.bert = bert

        bert_emb_size = 256

        emb_size = bert_emb_size
        
        self.preprocess_tokens = nn.Sequential(
            nn.Linear(768, bert_emb_size),
            nn.Hardswish(),
            nn.Linear(bert_emb_size, bert_emb_size),
            nn.Hardswish(),
        )
        self.bert_gru = nn.GRU(
            bert_emb_size,
            bert_emb_size//2,
            2,
            bidirectional=True,
            batch_first=True)

        self.final = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Hardswish(),
            nn.Linear(emb_size, emb_size),
            nn.Hardswish(),
            nn.Linear(emb_size, 4) # 3 classes, 1 numeric for note
        )

    def forward(self, toks, mask=None):
        x = self.bert(toks, attention_mask=mask)[0]

        x = self.preprocess_tokens(x)
        x, _h = self.bert_gru(x)
        x = x[:,-1,:] # only care about the last output

        x = self.final(x)

        class_pred = x[..., :-1]
        note_pred = x[..., -1]

        return class_pred, note_pred
