from datasets import load_dataset

print("📦 Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Export to CSV for your group mate
dataset["train"].to_csv("deberta_engagement_dataset.csv")
print("✅ Done! CSV created.")