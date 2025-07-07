import pandas as pd
from difflib import SequenceMatcher

# === CONFIG ===
files = [
    "evaluation_groq70b_bleu_bert_human.csv",
    "evaluation_mistral7b_bleu_bert_human.csv"
]
similarity_threshold = 0.94  # adjust based on your tolerance


def compute_similarity(a, b):
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def update_human_column(path):
    df = pd.read_csv(path)

    new_human = []
    for i, row in df.iterrows():
        sim = compute_similarity(row["Prompt"], row["Matched_GT"])
        score = 1 if sim >= similarity_threshold else 0
        new_human.append(score)

    df["Human"] = new_human
    df.to_csv(path, index=False)
    print(f"✅ Recomputed Human scores for: {path}")


# === RUN ===
for f in files:
    update_human_column(f)

print("🎉 All Human scores updated!")
