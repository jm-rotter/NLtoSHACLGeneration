import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

# Sample data: replace these with real examples later
generated_outputs = [
    "The edifact-o:hasDocumentNumber must be a string with max 12 characters."
]
ground_truths = [
    "The data element 1004 in the BGM segment is too long if the document number exceeds 12 characters."
]

def compute_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

def compute_bert(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return F1[0].item()

def evaluate_all(generated, references):
    for i, (gen, ref) in enumerate(zip(generated, references)):
        bleu = compute_bleu(ref, gen)
        bert = compute_bert(ref, gen)
        print(f"Sample {i+1}:")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU Score: {bleu:.4f}")
        print(f"BERT Score: {bert:.4f}\n")

if __name__ == "__main__":
    evaluate_all(generated_outputs, ground_truths)
