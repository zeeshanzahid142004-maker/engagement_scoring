import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score

# 1. SETUP: Force a new random seed to fix the "Bad Initialization"
torch.manual_seed(42)  
model_id = "microsoft/deberta-v3-xsmall"

# 2. DATA
print("⏳ Loading Dataset...")
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512) # DeBERTa likes full context

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Dynamic padding saves VRAM

# 3. MODEL
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# 4. METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 5. THE "GOLDILOCKS" ARGUMENTS
training_args = TrainingArguments(
    output_dir="./deberta_final_attempt",
    
    # --- THE FIXES ---
    learning_rate=2.5e-5,          # Strong enough to fix the bad start
    per_device_train_batch_size=8, # Fits in VRAM
    gradient_accumulation_steps=4, # Effective Batch = 32
    weight_decay=0.01,
    warmup_ratio=0.1,              # Warmup is CRITICAL for DeBERTa
    lr_scheduler_type="cosine",    # Smooth landing (Standard for DeBERTa)
    bf16=True,                     # RTX 5060 Ti Speed
    max_grad_norm=1.0,             # Loosened the brake slightly
    # -----------------
    
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# 6. TRAIN
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
   
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("🚀 Starting DeBERTa Training (The Real Attempt)...")
trainer.train()

# 7. SAVE
model.save_pretrained("./final_deberta_model")
tokenizer.save_pretrained("./final_deberta_model")
print("✅ DONE. Saved to ./final_deberta_model")