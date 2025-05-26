#This file is seperate from the rest of the project structure


import json
import re

translations = []


with open("shacltranslations.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        prompt = entry["prompt"]
        response = entry["response"]

        # Extract the natural language sentence from the prompt
        match = re.search(r'Translate this natural language sentence to SHACL:\n"(.*?)"\n### Response:', prompt, re.DOTALL)
        if match:
            nl = match.group(1).strip()
            shacl = response.strip()
            translations.append((shacl, nl))

# Now you can access the (shacl, nl) pairs
for shacl, nl in translations:
    print("NL:", nl)
    print("SHACL:\n", shacl)
    print("=" * 40) 
    break



