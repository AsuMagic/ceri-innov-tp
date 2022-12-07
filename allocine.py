import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CamembertModel, CamembertTokenizerFast
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from earlystop import EarlyStopping
import pandas as pd
import logging

from allocineutil import (
    attach_train_metadata,
    # attach_test_metadata
)
from allocinedataset import Dataset, make_loader
from model import SentimentPredictor


def behead_camembert(model, to_remove):
    model.encoder.layer = model.encoder.layer[:-to_remove]


def load_camembert(model_name, frozen):
    """Load a Camembert model and freeze its parameters if desired.
    When frozen, none of the CamemBERT weights are not adjusted during training."""

    logging.info("Loading CamemBERT tokenizer")
    tokenizer = CamembertTokenizerFast.from_pretrained(model_name)
    logging.info(f"Loading CamemBERT model (freeze={frozen})")
    camembert = CamembertModel.from_pretrained(model_name)

    if frozen:
        camembert.eval()
    else:
        camembert.train()
        # only fine tune some layers, otherwise the VRAM usage goes stonks
        for param in camembert.base_model.parameters():
            param.requires_grad = False

    return tokenizer, camembert


def eval(model, loader, ce_10_criterion, device):
    model.eval()

    running_loss = 0.0

    matches = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device):
                class_pred_logits, note_pred = model(batch["tokens"].to(device), batch["masks"].to(device))
                pred = torch.argmax(class_pred_logits, dim=1).detach().cpu()
                loss = ce_10_criterion(class_pred_logits, batch["cls_note"].to(device))

            matches += (pred == batch["cls_note"]).sum().item()

            total += len(pred)
            running_loss += loss.item() * batch["tokens"].shape[0]

    return matches / total, running_loss


def train(model, loader, ce_10_criterion, regression_criterion, optimizer, scaler, device):
    running_loss = 0.0

    matches = 0
    total = 0

    for batch in (pbar := tqdm(loader)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device):
            class_pred_logits, note_pred = model(batch["tokens"].to(device), batch["masks"].to(device))
            pred = torch.argmax(class_pred_logits, dim=1).detach().cpu()

            ce_10_loss = ce_10_criterion(class_pred_logits, batch["cls_note"].to(device))
            regression_loss = regression_criterion(note_pred, batch["note"].to(device))

            loss = ce_10_loss + regression_loss

        matches += (pred == batch["cls_note"]).sum().item()
        total += len(pred)
        
        running_loss += loss.item() * batch["tokens"].shape[0]

        pbar.set_description(f"train_acc={100*matches/total:5.3f}%")

        scaler.scale(loss).backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    return running_loss


def main():
    logging.basicConfig(level=logging.INFO)

    tqdm.pandas()

    parser = ArgumentParser()
    parser.add_argument("ckpt", type=str)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--freeze-camembert", default=False, action="store_true")
    parser.add_argument("--camembert-model", type=str, default="camembert-base")
    parser.add_argument("--camembert-removed-layers", type=int, default=3)

    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--dev-path", type=str, required=True)

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--batch-size", type=int, default=96)

    parser.add_argument("--comment", type=str, default="")

    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    writer = SummaryWriter(comment=args.comment)

    tokenizer, camembert = load_camembert(args.camembert_model, args.freeze_camembert)
    camembert.to(args.device)
    behead_camembert(camembert, args.camembert_removed_layers)

    def load_set(path):
        logging.info(f"Loading {path}")
        x = pd.read_pickle(path)
        x = Dataset(x)

        return x

    train_set = load_set(args.train_path)
    dev_set = load_set(args.dev_path)
    # test_set = load_set("dataset/test-metadata.bin.zst")

    attach_train_metadata(train_set.df)
    attach_train_metadata(dev_set.df)
    # attach_test_metadata(test_set.df)

    train_loader = make_loader(train_set, args.batch_size, train=True)
    dev_loader = make_loader(dev_set, args.batch_size)
    # test_loader = make_loader(test_set, args.batch_size)

    # Juicy model stuff
    logging.info("Setting up model")

    model = SentimentPredictor(
        camembert,
        args.camembert_removed_layers
    )

    #model = torch.jit.script(model)
    model.to(args.device)

    # example_forward_input = train_set[0]["tokens"].unsqueeze(0).to(args.device)
    # writer.add_graph(model, example_forward_input)
    # del example_forward_input

    # We use a custom loss function to handle the fact that the dataset is imbalanced
    class_freqs = train_set.df["cls_note"].value_counts().sort_index()
    class_weights = len(train_set) / class_freqs
    logging.info(f"Observed frequencies: {class_freqs} => weights {class_weights}")

    # We use the regression loss to make the model learn to predict an exact rating
    # TODO: need to test if this makes a difference; make it an option
    ce_10_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights.to_list(), device=args.device))
    regression_loss = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    early_stop = EarlyStopping(5, 0.01)

    # Doesn't seem to give any improvement
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    best_dev_acc = 0

    for epoch in range(args.epochs):
        loss = train(model, train_loader, ce_10_loss, regression_loss, optimizer, scaler, args.device)
        # scheduler.step()
        # scheduler.step(loss)
        
        writer.add_scalar("Loss/train", loss, epoch)

        current_dev_acc, dev_loss = eval(model, dev_loader, ce_10_loss, args.device)
        writer.add_scalar("Accuracy/dev", current_dev_acc, epoch)
        writer.add_scalar("Loss/dev", dev_loss, epoch)

        early_stop.update(current_dev_acc)
        if early_stop.should_stop():
            print("Early stopping patience exhausted; decided we should stop.")
            break

        if current_dev_acc > best_dev_acc:
            best_dev_acc = current_dev_acc
            # best_test_acc, test_loss = eval(model, test_loader, criterion, args.device)
            # writer.add_scalar("Accuracy/test", best_test_acc, epoch)
            # writer.add_scalar("Loss/test", test_loss, epoch)

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, args.ckpt)


if __name__ == "__main__":
    main()
