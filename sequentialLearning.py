import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score

# =========================
# 1️⃣ SETUP
# =========================
model_id = "distilbert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Training Engagement Model on: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# =========================
# 2️⃣ DATA: IMDb
# =========================
imdb_dataset = load_dataset("imdb")

# Optional: Reduce size for testing speed
# imdb_dataset["train"] = imdb_dataset["train"].shuffle(seed=42).select(range(2000))
# imdb_dataset["test"] = imdb_dataset["test"].shuffle(seed=42).select(range(500))

# =========================
# 3️⃣ TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_imdb = imdb_dataset.map(tokenize_function, batched=True)

# =========================
# 4️⃣ MODEL: Initialize
# =========================
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# =========================
# 5️⃣ METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# =========================
# 6️⃣ TRAINING ARGUMENTS
# =========================
training_args_imdb = TrainingArguments(
    output_dir="./imdb_engagement_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    bf16=True
)

# =========================
# 7️⃣ TRAINER: IMDb
# =========================
trainer = Trainer(
    model=model,
    args=training_args_imdb,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    compute_metrics=compute_metrics,
)

print("🚀 Starting Training on IMDb...")
trainer.train()

# Save IMDb-trained model
model.save_pretrained("./imdb_engagement_model")
tokenizer.save_pretrained("./imdb_engagement_model")
print("✅ IMDb-trained model saved.")

# =========================
# 8️⃣ DATA: Yelp Polarity
# =========================
yelp_dataset = load_dataset("yelp_polarity")

# Optional: Reduce size for speed
# yelp_dataset["train"] = yelp_dataset["train"].shuffle(seed=42).select(range(2000))
# yelp_dataset["test"] = yelp_dataset["test"].shuffle(seed=42).select(range(500))

tokenized_yelp = yelp_dataset.map(tokenize_function, batched=True)

# =========================
# 9️⃣ LOAD IMDb-TRAINED MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained("./imdb_engagement_model", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("./imdb_engagement_model")

# Optional: smaller LR for fine-tuning
training_args_yelp = TrainingArguments(
    output_dir="./final_engagement_model2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,  # smaller for fine-tuning
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args_yelp,
    train_dataset=tokenized_yelp["train"],
    eval_dataset=tokenized_yelp["test"],
    compute_metrics=compute_metrics,
)

print("🚀 Fine-tuning on Yelp Polarity...")
trainer.train()

# Save final model
model.save_pretrained("./final_engagement_model2")
tokenizer.save_pretrained("./final_engagement_model2")
print("✅ Final model saved to ./final_engagement_model2")
