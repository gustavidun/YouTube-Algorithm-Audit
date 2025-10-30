import os, numpy as np, pandas as pd, torch, torch.nn as nn
from datasets import Dataset
from config import MODELS_DIR, SLANTS_TRAIN
from models import Video

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME   = "roberta-base"   # or "roberta-base"
MAX_LENGTH   = 256
VAL_FRACTION = 0.10
BATCH_TRAIN  = 16
BATCH_EVAL   = 32
EPOCHS       = 6
LR           = 2e-5
WEIGHT_GAMMA = 1
OUTDIR       = MODELS_DIR / "roberta1"

# M2 Max / Apple Silicon: prefer MPS, keep fp32
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
print(f"Running on: {DEVICE} with {torch.get_num_threads()} threads")

def load():
    df = pd.read_pickle(SLANTS_TRAIN).copy()
    df["n"] = (df["R"] + df["L"]).astype(float)
    df = df.loc[df["n"] > 12].reset_index(drop=True)
    df["text"] = df.apply(join_row, axis=1)
    df["y"]    = (df["R"] / df["n"]).astype("float32")        # soft label in [0,1]
    df["w"]    = np.power(df["n"].values, WEIGHT_GAMMA)       # sample weight (tempered)

    ds = Dataset.from_pandas(df[["text","y","w"]])
    ds = ds.map(tokenize, batched=True, remove_columns=["text","y","w"])
    splits = ds.train_test_split(test_size=VAL_FRACTION, seed=42)
    train_ds, eval_ds = splits["train"], splits["test"]
    train_ds = train_ds.with_format("torch", columns=["input_ids","attention_mask","labels","w"])
    eval_ds  = eval_ds .with_format("torch", columns=["input_ids","attention_mask","labels","w"])
    return train_ds, eval_ds


def join_row(row):
    title = str(row.get("title", "")).strip()
    desc  = str(row.get("description", "")).strip()
    tags  = row.get("tags", "")
    if isinstance(tags, (list, tuple)): tags = " ".join(map(str, tags))
    tags = str(tags).strip()
    return " [SEP] ".join([s for s in (title, desc, tags) if s])


def join_video(vid : Video):
    title = vid.title.strip()
    desc = vid.description.strip()
    tags = " ".join(vid.tags) if vid.tags else ""
    return " [SEP] ".join([s for s in (title, desc, tags) if s])


def tokenize(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = batch["y"]
    enc["w"] = batch["w"]
    return enc


class BinomialTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("w")            # [B]
        labels  = inputs.pop("labels").float()  # y in [0,1]
        outputs = model(**inputs)
        logits  = outputs.logits.squeeze(-1)    # [B]
        # BCE with logits, per-sample
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )                                       # [B]
        loss = (bce * weights).mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    p = 1 / (1 + np.exp(-logits.reshape(-1)))
    sl_pred = 2*p - 1
    sl_true = 2*labels.reshape(-1) - 1
    mae  = float(np.mean(np.abs(sl_pred - sl_true)))
    rmse = float(np.sqrt(np.mean((sl_pred - sl_true)**2)))
    return {"slant_mae": mae, "slant_rmse": rmse}


def train(train_ds, eval_ds):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    args = TrainingArguments(
        output_dir=OUTDIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="slant_mae",
        greater_is_better=False,
        report_to="none",
        fp16=False, bf16=False,     # important for MPS
        remove_unused_columns=False,
    )

    trainer = BinomialTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    last_ckpt = get_last_checkpoint(str(OUTDIR))
    trainer.train(resume_from_checkpoint=last_ckpt)
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
    p = 1 / (1 + np.exp(-logits))
    for i, vid in enumerate(vids):
        vid.slant = float(2*p[i] - 1)
    return vids

if __name__ == "__main__":
    train_ds, eval_ds = load()
    train(train_ds, eval_ds)
