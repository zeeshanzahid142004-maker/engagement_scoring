import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

# -------------------------------
# DEVICE
# -------------------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using device:", DEVICE)

# -------------------------------
# LOAD ENGAGEMENT SCORER
# -------------------------------
ENGAGEMENT_PATH = "./final_engagement_model2"

engagement_tokenizer = AutoTokenizer.from_pretrained(ENGAGEMENT_PATH)
engagement_model = AutoModelForSequenceClassification.from_pretrained(
    ENGAGEMENT_PATH
).to(DEVICE)
engagement_model.eval()


def score_engagement(text: str) -> float:
    inputs = engagement_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = engagement_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    return probs[0][1].item()


# -------------------------------
# LOAD FLAN-T5 REWRITER (PROPER WAY)
# -------------------------------
rewriter_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
rewriter_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
).to(DEVICE)
rewriter_model.eval()


def generate_rewrites(text, n=5):
    prompt = (
        "Rewrite the following sentence to be exciting and engaging "
        "while keeping the same meaning:\n"
        f"{text}"
    )

    inputs = rewriter_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    outputs = rewriter_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=n
    )

    return [
        rewriter_tokenizer.decode(o, skip_special_tokens=True).strip()
        for o in outputs
    ]


def improve_engagement(text):
    candidates = generate_rewrites(text, n=5)

    scored = [(c, score_engagement(c)) for c in candidates]
    best = max(scored, key=lambda x: x[1])

    print("\n---------------- ORIGINAL ----------------")
    print(text, " | score:", round(score_engagement(text), 4))

    print("\n---------------- CANDIDATES ----------------")
    for t, s in scored:
        print(round(s, 4), "->", t)

    print("\n---------------- BEST ----------------")
    print(best[0], "| score:", round(best[1], 4))

    return best[0]


# -------------------------------
# RUN TEST
# -------------------------------
if __name__ == "__main__":
    improve_engagement("The movie was slow and boring.")
    improve_engagement("It was very cold outside.")
