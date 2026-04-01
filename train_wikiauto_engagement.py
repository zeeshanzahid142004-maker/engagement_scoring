import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score


# -------------------------------
# CONFIG
# -------------------------------
MODEL_ID = "distilbert-base-uncased"
OUTPUT_DIR = "./wikiauto_engagement_model"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")


# -------------------------------
# LOAD DATASET
# -------------------------------
print("📦 Loading WikiAuto...")
dataset = load_dataset("wikilarge")


# -------------------------------
# BUILD LABELED DATA
# -------------------------------
def build_pairs(example):
    data = []
    normal = example["normal"]
    simple = example["simple"]

    if normal and simple:
        data.append({"text": normal, "label": 0})
        data.append({"text": simple, "label": 1})

    return data



print("🧱 Building labeled dataset...")

train_rows = []
for ex in dataset["train"]:
    train_rows.extend(build_pairs(ex))

test_rows = []
for ex in dataset["validation"]:
    test_rows.extend(build_pairs(ex))


from datasets import Dataset

train_ds = Dataset.from_list(train_rows)
test_ds = Dataset.from_list(test_rows)


# -------------------------------
# TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )


train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# -------------------------------
# MODEL
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)


# -------------------------------
# METRICS
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# -------------------------------
# TRAINING ARGS
# -------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=torch.cuda.is_available()
)


# -------------------------------
# TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)


# -------------------------------
# TRAIN
# -------------------------------
print("🚀 Starting WikiAuto Engagement Training...")
trainer.train()


# -------------------------------
# SAVE
# -------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to {OUTPUT_DIR}")
