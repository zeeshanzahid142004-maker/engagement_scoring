import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score

# 1. SETUP: Model ID and Hardware
# OLD:
# model_id = "microsoft/deberta-v3-xsmall"

# NEW (The Reliable One):
model_id = "distilbert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Training Engagement Model on: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# 2. DATA: Load IMDb (Engagement/Sentiment)
# This dataset is huge (25k rows), so it learns VERY well.
dataset = load_dataset("imdb")

# Reduce size for speed if needed (Optional: remove these lines to train on full data)
# dataset["train"] = dataset["train"].shuffle(seed=42).select(range(2000)) 
# dataset["test"] = dataset["test"].shuffle(seed=42).select(range(500))

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. PREPROCESSING: Tokenize the data
def tokenize_function(examples):
    # IMDb uses the column name "text", not "sentence"
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. MODEL: Load DeBERTa
# num_labels=2 (0 = Negative/Boring, 1 = Positive/Engaging)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# 5. METRICS: Use Accuracy for Engagement
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 6. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./my_engagement_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,             # Standard reliable speed
    per_device_train_batch_size=16, # DistilBERT is light, 16 fits easily
    num_train_epochs=2,             # It learns FAST. 2 is plenty.
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    bf16=True                       # Keep this for your RTX 5060 Ti speed
)

# 7. TRAINER: Initialize and Run
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("🚀 Starting Training on RTX 5060 Ti...")
trainer.train()

# 8. SAVE
model.save_pretrained("./final_engagement_model")
tokenizer.save_pretrained("./final_engagement_model")
print("✅ Model saved to ./final_engagement_model")