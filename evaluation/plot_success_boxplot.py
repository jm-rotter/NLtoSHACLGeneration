import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv("evaluation_groq70b_bleu_bert_human.csv")
df2 = pd.read_csv("evaluation_mistral7b_bleu_bert_human.csv")

df1["Model"] = "groq70b"
df2["Model"] = "mistral7b"
df = pd.concat([df1, df2])

df["Success"] = df["Human"].apply(lambda x: "Success" if x == 1 else "Unsuccessful")
df["Group"] = df["Model"] + " - " + df["Success"]

# Final color palette (4)
palette = {
    "groq70b - Success": "#2b922b",        # green
    "groq70b - Unsuccessful": "#CB3333",   # red
    "mistral7b - Success": "#2885c8",      # blue
    "mistral7b - Unsuccessful": "#ca731c"  # orange
}

# === BERT Score Boxplot + Overlay Dots
plt.figure(figsize=(10, 6))
sns.boxplot(x="Success", y="BERT", hue="Group", data=df, palette=palette, showfliers=False)
sns.stripplot(x="Success", y="BERT", hue="Group", data=df, palette=palette,
              dodge=True, alpha=0.5, linewidth=1, edgecolor='gray')

plt.title("BERT Score Distribution: Successful vs Unsuccessful (4-color)")
plt.ylabel("BERT Score")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("evaluation/bert_score_boxplot_colored.png")
plt.show()
