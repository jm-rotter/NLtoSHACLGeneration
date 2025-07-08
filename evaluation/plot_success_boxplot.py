import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load CSVs ===
df1 = pd.read_csv("evaluation/groq70b_bleu_bert_human.csv")
df2 = pd.read_csv("evaluation/mistral7b_bleu_bert_human.csv")
df3 = pd.read_csv("evaluation/qwen7b_bleu_bert_human.csv")


# === Add model names ===
df1["Model"] = "groq70b"
df2["Model"] = "mistral7b"
df3["Model"] = "qwen7b"

# === Combine all DataFrames ===
df = pd.concat([df1, df2, df3])

# === Create Success/Unsuccessful Grouping ===
df["Success"] = df["Human"].apply(lambda x: "Success" if x == 1 else "Unsuccessful")
df["Group"] = df["Model"] + " - " + df["Success"]

# === Define Final 6-Color Palette ===
palette = {
    "groq70b - Success": "#2b922b",
    "groq70b - Unsuccessful": "#CB3333",
    "mistral7b - Success": "#2885c8",
    "mistral7b - Unsuccessful": "#ca731c",
    "qwen7b - Success": "#8a2be2",
    "qwen7b - Unsuccessful": "#888888"
}

# === BERT Score Boxplot + Overlay ===
plt.figure(figsize=(12, 6))
sns.boxplot(x="Success", y="BERT", hue="Group", data=df, palette=palette, showfliers=False)
sns.stripplot(x="Success", y="BERT", hue="Group", data=df, palette=palette,
              dodge=True, alpha=0.5, linewidth=1, edgecolor='gray')

plt.title("BERT Score Distribution: Successful vs Unsuccessful (3 Models)")
plt.ylabel("BERT Score")
plt.xlabel("")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("evaluation/bert_score_boxplot.png")
plt.show()
