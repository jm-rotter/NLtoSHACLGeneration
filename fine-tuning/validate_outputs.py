import sys
import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from syntheticDataSet.utils import load_translations_from_json

# Load real data from JSONL
generated_outputs, ground_truths = load_translations_from_json("syntheticDataSet/shacltranslations.jsonl")

# Optional: disable hardcoded test data
# generated_outputs = [
#     "The edifact-o:hasDocumentNumber must be a string with max 12 characters."
# ]
# ground_truths = [
#     "The data element 1004 in the BGM segment is too long if the document number exceeds 12 characters."
# ]

def compute_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

def compute_bert(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return F1[0].item()

def evaluate_all(generated, references):
    total_bleu = 0
    total_bert = 0
    count = len(generated)

    for i, (gen, ref) in enumerate(zip(generated, references)):
        bleu = compute_bleu(ref, gen)
        bert = compute_bert(ref, gen)
        total_bleu += bleu
        total_bert += bert
        print(f"Sample {i+1}:")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU Score: {bleu:.4f}")
        print(f"BERT Score: {bert:.4f}\n")

    print(f"\n✅ Evaluated {count} examples")
    print(f"🔹 Average BLEU Score: {total_bleu / count:.4f}")
    print(f"🔹 Average BERT Score: {total_bert / count:.4f}")

if __name__ == "__main__":
    evaluate_all(generated_outputs, ground_truths)
