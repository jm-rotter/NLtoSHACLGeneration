import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
csv_paths = [
    "evaluation_groq70b_bleu_bert_human.csv",
    "evaluation_mistral7b_bleu_bert_human.csv"
]

model_names = ["GROQ70B", "MISTRAL7B"]

# === Read CSVs ===
dataframes = [pd.read_csv(path) for path in csv_paths]

# === Compute metrics ===
accuracy = [(df["Human"] == 1).mean() for df in dataframes]
bleu = [df["BLEU"].mean() for df in dataframes]
bert = [df["BERT"].mean() for df in dataframes]

# === Accuracy Plot ===
plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracy, color='skyblue')
plt.title("Model Exact Match Accuracy")
plt.ylabel("Accuracy (0–1)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("evaluation/model_accuracy.png")

# === BLEU + BERT Scores Plot ===
bar_width = 0.35
x = range(len(model_names))

plt.figure(figsize=(8, 5))
plt.bar([i - bar_width/2 for i in x], bleu, width=bar_width, label="BLEU")
plt.bar([i + bar_width/2 for i in x], bert, width=bar_width, label="BERT", alpha=0.7)
plt.xticks(x, model_names)
plt.title("Model BLEU vs BERT Scores")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("evaluation/bleu_bert_scores.png")

print("✅ All plots saved to evaluation/ folder using CSV data.")
