import os, numpy as np, pandas as pd, torch, torch.nn as nn
from datasets import Dataset
from config import MODELS_DIR, SLANTS_TRAIN
from models import Video

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)

MODEL_NAME   = "roberta-base"   # or "roberta-base"
MAX_LENGTH   = 256
VAL_FRACTION = 0.10
BATCH_TRAIN  = 16
BATCH_EVAL   = 32
EPOCHS       = 3
LR           = 2e-5
WEIGHT_GAMMA = 1
OUTDIR       = MODELS_DIR / "roberta_linear"

# M2 Max / Apple Silicon: prefer MPS, keep fp32
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tok)
print("Running on:", DEVICE)


def load():
    df = pd.read_pickle(SLANTS_TRAIN)
    df["n"] = (df["R"] + df["L"]).astype(float)
    df = df.loc[df["n"] > 12].reset_index(drop=True)
    df["text"] = df.apply(join_row, axis=1)

    ds = Dataset.from_pandas(df[["text","slant"]])
    ds = ds.map(tokenize, batched=True, remove_columns=["text","slant"])
    splits = ds.train_test_split(test_size=VAL_FRACTION, seed=42)
    train_ds, eval_ds = splits["train"], splits["test"]
    return train_ds, eval_ds


def join_row(row):
    title = str(row.get("title", "")).strip()
    desc  = str(row.get("description", "")).strip()
    tags  = row.get("tags", "")
    comments = row.get("comments", "")
    if isinstance(tags, (list, tuple)): tags = " ".join(map(str, tags))
    if isinstance(comments, (list, tuple)): comments = " ".join(map(str, comments))
    tags = str(tags).strip()
    comments = str(comments).strip()
    return " [SEP] ".join([s for s in (title, desc, tags, comments) if s])


def join_video(vid : Video):
    title = vid.title.strip()
    desc = vid.description.strip()
    tags = " ".join(vid.tags) if vid.tags else ""
    return " [SEP] ".join([s for s in (title, desc, tags) if s])


def tokenize(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = batch["slant"]
    return enc


def compute_metrics(pred):
    yhat, y = pred
    yhat = np.clip(yhat.reshape(-1), -1, 1)
    y    = y.reshape(-1)
    mae  = float(np.mean(np.abs(yhat - y)))
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    ybar = float(y.mean()); sst = float(((y - ybar)**2).sum() or 1.0)
    r2   = 1.0 - float(((y - yhat)**2).sum()) / sst
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train(train_ds, eval_ds):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    args = TrainingArguments(
        output_dir=OUTDIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy='epoch',
        load_best_model_at_end=True,
        fp16=False, bf16=False,     # important for MPS
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(OUTDIR)
    tok.save_pretrained(OUTDIR)


def predict_videos(model_path, vids : list[Video]):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    enc = tok([join_video(vid) for vid in vids], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1).cpu().numpy()
    return np.clip(logits, -1.0, 1.0)


if __name__ == "__main__":
    train_ds, eval_ds = load()
    train(train_ds, eval_ds)
