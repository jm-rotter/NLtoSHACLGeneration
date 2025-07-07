import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_fn

# Load Groq inference file
inference_file = "groq_inferenced.jsonl"
ground_truth_file = "groundTruth/ground_truth.jsonl"

# Load ground truth map
ground_truth_map = {}
with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
    for line in gt_file:
        item = json.loads(line)
        ground_truth_map[item["input"].strip()] = item["output"].strip()

def get_ground_truth(shacl_shape):
    return ground_truth_map.get(shacl_shape.strip(), "")

def compute_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

def compute_bert_score(predictions, references):
    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return F1

def evaluate_all():
    references = []
    predictions = []
    inputs = []

    with open(inference_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            shacl_input = item["input"].strip()
            prediction = item["output"].strip()
            reference = get_ground_truth(shacl_input)

            predictions.append(prediction)
            references.append(reference)
            inputs.append(shacl_input)

    total_bleu = 0
    total_bert = 0
    exact_matches = 0
    mismatches = []
    count = len(predictions)

    F1_scores = compute_bert_score(predictions, references)

    for i in range(count):
        gen = predictions[i]
        ref = references[i]
        bleu = compute_bleu(ref, gen)
        bert = F1_scores[i].item()
        is_exact = gen.strip() == ref.strip()

        if is_exact:
            exact_matches += 1
        else:
            mismatches.append((i + 1, gen, ref))

        total_bleu += bleu
        total_bert += bert

    print("\n===== SUMMARY =====")
    print(f" Total Samples          : {count}")
    print(f" Exact Matches          : {exact_matches}")
    print(f" Mismatches             : {len(mismatches)}")
    print(f" Match Accuracy         : {(exact_matches / count) * 100:.2f}%")
    print(f" Average BLEU Score     : {total_bleu / count:.4f}")
    print(f" Average BERT Score     : {total_bert / count:.4f}")

    if mismatches:
        print("\n❗ Mismatched Predictions:")
        for idx, pred, ref in mismatches:
            print(f"\n Sample #{idx}")
            print(f" Predicted: {pred}")
            print(f" Expected : {ref}")

if __name__ == "__main__":
    evaluate_all()
