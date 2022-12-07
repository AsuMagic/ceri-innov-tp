from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertTokenizerFast
from tqdm import tqdm

from allocineutil import get_rating_value, pad, truncate_if_needed


class SentimentPredictor(nn.Module):
    def __init__(
        self,
        bert: CamembertModel,
        bert_ablated_layers: int,
        gru_emb_size: int = 256,
        final_emb_size: int = 256,
    ):
        super().__init__()

        self.bert = bert
        self.bert_ablated_layers = bert_ablated_layers

        camembert_size = bert.config.hidden_size

        self.preprocess_tokens = nn.Sequential(
            nn.Linear(camembert_size, gru_emb_size),
            nn.Hardswish(),
            nn.Linear(gru_emb_size, gru_emb_size),
            nn.Hardswish(),
        )

        self.bert_gru = nn.GRU(
            gru_emb_size,
            gru_emb_size//2,
            2,
            bidirectional=True,
            batch_first=True
        )

        self.final = nn.Sequential(
            nn.Linear(final_emb_size, final_emb_size),
            nn.Hardswish(),
            nn.Linear(final_emb_size, final_emb_size),
            nn.Hardswish(),
            nn.Linear(final_emb_size, 11)  # 10 classes, 1 numeric for rating
        )

    def init_weights(self):
        # initialize the token preprocessor, the GRU and the final layer
        for module in [self.preprocess_tokens, self.bert_gru, self.final]:
            for name, param in module.named_parameters():
                # kaiming init
                if "weight" in name:
                    torch.nn.init.kaiming_normal_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def process_bert_output(self, x):
        x = self.preprocess_tokens(x)
        x, _h = self.bert_gru(x)
        x = x[:, -1, :]  # only care about the last output

        x = self.final(x)

        return x

    def forward(self, toks, mask=None):
        x = self.bert(toks, attention_mask=mask)[0]

        x = self.process_bert_output(x)

        class_pred = x[..., :-1]
        note_pred = x[..., -1]

        return class_pred, note_pred


@dataclass
class AllocinePrediction:
    rating_regression: float

    rating_class_pred: torch.Tensor
    rating_class: int


def behead_camembert(model, to_remove):
    model.encoder.layer = model.encoder.layer[:-to_remove]


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class AllocinePredictor:
    def __init__(self, path, device):
        bert_name = "camembert-base"

        self.tokenizer = CamembertTokenizerFast.from_pretrained(bert_name)
        camembert = CamembertModel.from_pretrained(bert_name)
        camembert.eval()

        state = torch.load(path, map_location=torch.device(device))

        self.model = SentimentPredictor(camembert)
        self.model.load_state_dict(state["model_state_dict"], strict=False)

        # Make sure we're removing as many layers as the model was trained with
        # bert_ablated_layers was loaded from the state dict above, so we can
        # only do it here
        # TODO: is there a better way to achieve this?
        behead_camembert(self.model.bert, self.model.bert_ablated_layers)

        self.model.bert.load_state_dict(state["model_state_dict"], strict=False)

        self.device = device
        self.model.bert.to(device)
        self.model.to(device)

    def __call__(self, inputs: List[str], batch_size: int = 32, use_tqdm: bool = False):
        preds = []

        with torch.no_grad():
            iter_wrapper = (lambda x: tqdm(list(x))) if use_tqdm else (lambda x: x)

            def encode_sentence(text):
                tokens = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True))

                max_bert_tokens = self.model.bert.config.max_position_embeddings

                tokens = truncate_if_needed(tokens, max_bert_tokens)
                return tokens

            for chunk in iter_wrapper(chunker(inputs, batch_size)):
                tokens = [encode_sentence(sentence) for sentence in chunk]
                tokens, mask = pad(tokens, 1)
                tokens = tokens.to(self.device)
                mask = mask.to(self.device)

                class_preds, note_preds = self.model(tokens, mask)

                for i in range(len(note_preds)):
                    preds.append(AllocinePrediction(
                        rating_regression=note_preds[i].item(),
                        rating_class_pred=class_preds[i].detach().cpu(),
                        rating_class=get_rating_value(class_preds.argmax(dim=-1)[i].item())
                    ))

        return preds