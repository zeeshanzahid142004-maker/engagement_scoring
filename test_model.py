from transformers import pipeline

# Load YOUR saved model
model_path = "./final_engagement_model"
print(f"Loading model from {model_path}...")

# Create the pipeline
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0)

# Test sentences
tests = [
    "This movie was absolutely incredible, I couldn't look away!",  # Should be Label 1 (Positive)
    "The plot was boring and the acting was terrible.",             # Should be Label 0 (Negative)
    "I fell asleep halfway through.",                               # Should be Label 0 (Negative)
    "A masterpiece of visual storytelling.",                         # Should be Label 1 (Positive)
     "it was ok.",
     "I fell asleep.",
     "it was fine" ,
     "didnt think much abouyt it",
     "couldnt remember the characters name or story line after leaving the cinema",
     "i sleep everyday",
     "The data shows a significant increase in revenue.",  # Formal/Boring (Should be Negative/Low Score)
    "This strategy is an absolute game-changer for the industry!", # Exciting (Should be Positive)
    "the plot was moving",
    "the plot was barely moving",
    "the house was very big",
    "the house was gigantic"
    
]

print("\n--- RESULTS ---")
for text in tests:
    result = classifier(text)[0]
    label = "ENGAGING (Positive)" if result['label'] == "LABEL_1" else "BORING (Negative)"
    score = result['score']
    print(f"Text: '{text}'")
    print(f"Prediction: {label} ({score:.2%})\n")