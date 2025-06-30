import json

source_file = "syntheticDataSet/training_translations.jsonl"
output_file = "groundTruth/ground_truth.jsonl"

with open(source_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        item = json.loads(line)
        new_item = {
            "input": item["response"].strip(),
            "output": item["prompt"].strip()
        }
        f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print("✅ ground_truth.jsonl created successfully!")
