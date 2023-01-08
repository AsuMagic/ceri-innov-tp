import logging
import math
from argparse import ArgumentParser
from functools import partialmethod
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizerFast

# attach_test_metadata
from allocinedataset import Dataset, make_loader
from allocineutil import (
    attach_train_metadata,
)
from earlystop import EarlyStopping
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


def train(model, loader, ce_10_criterion, regression_criterion, optimizer, scaler, device, use_tqdm=True):
    running_loss = 0.0

    matches = 0
    total = 0

    for batch in (pbar := tqdm(loader, disable=not use_tqdm)):
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


def load_dataset(path, sample_frac=1.0):
    logging.info(f"Loading {path}, sampling {sample_frac:.2%}")
    x = pd.read_pickle(path)
    x = Dataset(x)
    attach_train_metadata(x.df)
    if sample_frac < 1.0:
        x.df = x.df.sample(frac=sample_frac)

    return x

def train_model(config):
    # writer = SummaryWriter(comment=args.comment)

    # which dataset to load, depending on the used camembert model?
    if "large" in config["camembert_model"]:
        sets = config["sets"]["large"]
    else:
        sets = config["sets"]["base"]

    # import the dataset with the proper % of samples
    # in hyperparameter search, we use a fraction of the dataset
    train_set = load_dataset(sets["train"], config["train_frac"])
    dev_set = load_dataset(sets["dev"], config["dev_frac"])

    train_loader = make_loader(train_set, config["batch_size"], train=True)
    dev_loader = make_loader(dev_set, config["batch_size"])

    # load camembert, apply the number of layers to remove
    tokenizer, camembert = load_camembert(config["camembert_model"], False)
    camembert.to(config["device"])
    behead_camembert(camembert, config["removed_layers"])

    # create the model
    model = SentimentPredictor(
        camembert,
        config["removed_layers"],
        config["pre_emb_size"],
        config["final_emb_size"],
        config["pre_layers"],
        config["final_layers"],
        config["dropout"],
        config["sequence_processor"]
    ).to(config["device"])

    # example_forward_input = train_set[0]["tokens"].unsqueeze(0).to(args.device)
    # writer.add_graph(model, example_forward_input)
    # del example_forward_input

    # we use a custom loss function to handle the fact that the dataset is imbalanced
    class_freqs = train_set.df["cls_note"].value_counts().sort_index()
    class_weights = len(train_set) / class_freqs
    logging.info(f"Observed frequencies: {class_freqs} => weights {class_weights}")

    # we use the regression loss to make the model learn to predict an exact rating
    # TODO: need to test if this makes a difference; make it an option
    ce_10_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights.to_list(), device=config["device"]))
    regression_loss = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()
    early_stop = EarlyStopping(5, 0.01)

    # Doesn't seem to give any improvement
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    best_dev_acc = 0

    for epoch in range(config["max_epochs"]):
        loss = train(
            model,
            train_loader,
            ce_10_loss,
            regression_loss,
            optimizer,
            scaler,
            config["device"],
            use_tqdm=not config["tuning"]
        )

        if math.isnan(loss):
            raise ValueError("Loss is NaN")

        # scheduler.step()
        # scheduler.step(loss)
        
        # writer.add_scalar("Loss/train", loss, epoch)

        current_dev_acc, dev_loss = eval(model, dev_loader, ce_10_loss, config["device"])
        # writer.add_scalar("Accuracy/dev", current_dev_acc, epoch)
        # writer.add_scalar("Loss/dev", dev_loss, epoch)

        # early_stop.update(current_dev_acc)
        # if early_stop.should_stop():
        #     print("Early stopping patience exhausted; decided we should stop.")
        #     break

        if current_dev_acc > best_dev_acc:
            best_dev_acc = current_dev_acc
            # best_test_acc, test_loss = eval(model, test_loader, criterion, config["device"])
            # writer.add_scalar("Accuracy/test", best_test_acc, epoch)
            # writer.add_scalar("Loss/test", test_loss, epoch)

        if config["tuning"]:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = Path(checkpoint_dir) / "checkpoint"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, path)

            tune.report(loss=dev_loss, accuracy=current_dev_acc)
        else:
            path = Path(config["save_dir"]) / f"checkpoint-{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tqdm.pandas()

    parser = ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")

    # use subparsers to decide for hyperparameter search or training
    subparsers = parser.add_subparsers(dest="mode")

    # hyperparameter search
    hp_parser = subparsers.add_parser("hyperparam")
    
    # training
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--hparams", type=str, default="hparams-stock.json")
    train_parser.add_argument("--max-epochs", type=int, default=10)
    train_parser.add_argument("--save-dir", type=str, default="checkpoints")

    args = parser.parse_args()

    if args.mode == "hyperparam":
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        config = {
            "camembert_model": tune.choice([
                "camembert-base",
                # no tokenizers
                # "camembert/camembert-base-ccnet",
                # "camembert/camembert-base-oscar-4gb",
                "camembert/camembert-large"
            ]),
            "pre_emb_size": tune.choice([32, 64, 128, 256]),
            "final_emb_size": tune.choice([32, 64, 128, 256]),
            "pre_layers": tune.choice([1, 2, 3]),
            "final_layers": tune.choice([2, 3, 4]),
            "dropout": tune.uniform(0.0, 0.5),
            "batch_size": tune.choice([8, 16, 32, 64, 96]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "removed_layers": tune.choice([0, 1, 2, 3, 4, 5, 6]),
            "sequence_processor": tune.choice(["gru", "pool"]),
            "max_epochs": 100,
            "tuning": True,
            "train_frac": 0.3,
            "dev_frac": 0.5
        }
    elif args.mode == "train":
        import json
        with open(args.hparams, "r") as f:
            config = json.load(f)
        config.update({
            "max_epochs": args.max_epochs,
            "save_dir": args.save_dir,
            "tuning": False,
            "train_frac": 1.0,
            "dev_frac": 1.0,
        })
    
    config.update({
        "device": args.device,
        "sets": {
            "base": {
                "train": Path("dataset/camembert-base/train-metadata.bin.zst").resolve(),
                "dev": Path("dataset/camembert-base/dev-metadata.bin.zst").resolve(),
            },
            "large": {
                "train": Path("dataset/camembert-large/train-metadata.bin.zst").resolve(),
                "dev": Path("dataset/camembert-large/dev-metadata.bin.zst").resolve(),
            }
        }
    })

    if args.mode == "hyperparam":
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=15,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])

        result = tune.run(
            train_model,
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=config,
            scheduler=scheduler,
            num_samples=30,
            progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
    elif args.mode == "train":
        train_model(config)