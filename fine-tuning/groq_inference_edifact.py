
import os
import sys
import json
from dotenv import load_dotenv
from groq import Groq

# Ensure project structure is included in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from syntheticDataSet.shaclParser import pullShapes
from syntheticDataSet.prompts import build_prompt, FEW_SHOT_EXAMPLES, INITIAL_PROMPT

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load shapes
shapes = pullShapes()
print(f"✅ Loaded {len(shapes)} shapes from dataset.")

# Main translation loop
for i, graph in enumerate(shapes):
    shape_text = graph.serialize(format='turtle')
    if isinstance(shape_text, bytes):
        shape_text = shape_text.decode("utf-8")

    prompt = build_prompt(shape_text, FEW_SHOT_EXAMPLES, INITIAL_PROMPT)
    print(f"[{i+1}] 📨 Sending prompt...")

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a SHACL expert."},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.choices[0].message.content.strip()
        print(f"[{i+1}] ✅ Response:\n{output[:150]}...")

        # 🔁 Save to JSONL
        with open("groq_inferenced.jsonl", "a", encoding="utf-8") as f_jsonl:
            json.dump({"input": shape_text, "output": output}, f_jsonl, ensure_ascii=False)
            f_jsonl.write("\n")

        # 🔁 Save to TXT
        with open("groq_inferenced.txt", "a", encoding="utf-8") as f_txt:
            f_txt.write(f"=== Translation #{i+1} ===\n")
            f_txt.write(shape_text.strip() + "\n\n")
            f_txt.write("↓\n\n")
            f_txt.write(output.strip() + "\n\n\n")

    except Exception as e:
        print(f"[{i+1}] ❌ Error: {e}")
        continue

print("✅ Inference completed and files are being updated continuously.")


# # Final save
# if not translations:
#     print("⚠️ No translations were added — check API, model, or input.")
# else:
#     with open("groq_inferenced.jsonl", "w", encoding="utf-8") as f_jsonl:
#         for t in translations:
#             f_jsonl.write(json.dumps(t, ensure_ascii=False) + "\n")

#     with open("groq_inferenced.txt", "w", encoding="utf-8") as f_txt:
#         for i, t in enumerate(translations):
#             f_txt.write(f"=== Translation #{i+1} ===\n")
#             f_txt.write(t["input"].strip() + "\n\n↓\n\n")
#             f_txt.write(t["output"].strip() + "\n\n\n")

