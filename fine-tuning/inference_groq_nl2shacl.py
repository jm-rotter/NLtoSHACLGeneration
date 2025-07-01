import os
import json
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# File paths
INPUT_FILE = "syntheticDataset/shacltranslations.jsonl"
TXT_OUT = "fine-tuning/groq70b.txt"
JSONL_OUT = "fine-tuning/groq70b.jsonl"


# Load input NL prompts and GT SHACLs
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"✅ Loaded {len(data)} examples from {INPUT_FILE}")

# Check how many are already completed
already_done = 0
if os.path.exists(JSONL_OUT):
    with open(JSONL_OUT, "r", encoding="utf-8") as f:
        already_done = sum(1 for _ in f)

print(f"🔁 Resuming from example #{already_done + 1}")

# Prompt
def genprompt(to_translate):
    return """
    You are a SHACL expert. Convert the following natural language description into a syntactically valid and semantically correct SHACL shape using Turtle syntax.

    Requirements:

    Use the standard SHACL vocabulary: sh:NodeShape, sh:property, sh:path, sh:in, sh:datatype, sh:minCount, sh:maxCount, sh:message, etc.

    Include all the given options in sh:in if that case appears, there is no such thing as sh:List. 

    The property constraints must be defined inside a single blank node ([]) as the object of sh:property.

    Use RDF lists (i.e., ( ... )) for sh:in enumerations, containing string literals (e.g., "Commercial invoice").

    Use sh:path to point to the constrained property (e.g., sh:path edifact-o:hasDocumentType).

    Use the correct xsd:string directly for the datatype, e.g., sh:datatype xsd:string (without brackets).

    Use sh:minCount and sh:maxCount within the sh:property blank node (not at the top-level).

    Add an appropriate sh:message in the same blank node if a validation message is provided.

    Use sh:targetClass to define the class this shape targets.

    Output must be syntactically valid SHACL and complete Turtle, runnable in any SHACL engine.

    Always define SHACL NodeShapes using a valid IRI or blank node. Do NOT use square brackets in subject position (e.g., [:Something ...] is invalid). Use :Something instead.

    Include the following prefix declarations at the top: Use: """ + rag(to_translate) + """\n\n

    """ + to_translate
# Open output files
with open(TXT_OUT, "a", encoding="utf-8") as f_txt, open(JSONL_OUT, "a", encoding="utf-8") as f_jsonl:
    for i in tqdm(range(already_done, len(data))):
        nl = data[i]["prompt"]
        gt = data[i]["response"]

        messages = [
            {"role": "system", "content": "You are a SHACL expert."},
            {"role": "user", "content": build_prompt(nl)}
        ]

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )

        generated = response.choices[0].message.content.strip()

        # Save to .txt
        f_txt.write(f"\n=== Translation #{i+1} ===\n")
        f_txt.write(nl.strip() + "\n\n")
        f_txt.write(generated + "\n\nOriginal:\n")
        f_txt.write(gt.strip() + "\n\n\n")

        # Save to .jsonl
        json_line = json.dumps({"NL": nl, "GT": gt, "Generated": generated}, ensure_ascii=False)
        f_jsonl.write(json_line + "\n")


print("🎉 All inference completed and saved.")
