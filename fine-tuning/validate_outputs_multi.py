
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import get_close_matches
from bert_score import score as bert_score  # BERT score enabled

# === SETTINGS ===
MODELS = [
    {"name": "mistral7b", "file": "./fine-tuning/mistral7b.jsonl"},
    {"name": "groq70b", "file": "./fine-tuning/groq70b.jsonl"},
    # {"name": "llama7b", "file": "./fine-tuning/llama7b.jsonl"},
    # {"name": "qwen7b", "file": "./fine-tuning/qwen7b.jsonl"},
]

ground_truth_file = "./groundTruth/ground_truth.jsonl"
threshold = 0.80  # fuzzy match threshold

# === LOAD GROUND TRUTH ===
with open(ground_truth_file, "r", encoding="utf-8") as f:
    gt_data = [json.loads(line) for line in f]

gt_map = {entry["output"].strip().lower(): entry["input"].strip() for entry in gt_data}
gt_keys = list(gt_map.keys())

# === EVALUATE EACH MODEL ===
for model in MODELS:
    model_name = model["name"]
    model_file = model["file"]
    output_file = f"evaluation_{model_name}_bleu_bert_human.csv"

    try:
        with open(model_file, "r", encoding="utf-8") as f:
            pred_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f" File not found: {model_file}")
        continue

    results = []
    smooth = SmoothingFunction().method4

    for item in pred_data:
        nl = item.get("NL", item.get("prompt", "")).strip().lower()
        pred = item.get("Generated", item.get("response", "")).strip()

        match = get_close_matches(nl, gt_keys, n=1, cutoff=threshold)
        if not match:
            continue
        matched_nl = match[0]
        gt = gt_map[matched_nl]

        bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth)
        human = 1 if pred == gt else 0

        #  BERT Score
        P, R, F1 = bert_score([pred], [gt], lang="en", verbose=False)
        bert = F1[0].item()

        results.append({
            "Prompt": nl,
            "Matched_GT": matched_nl,
            "BLEU": bleu,
            "BERT": bert,
            "Human": human,
            "Model": model_name
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f" Done! Scores saved to: {output_file}")
