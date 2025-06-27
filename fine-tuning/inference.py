from unsloth import FastLanguageModel
import torch
from safetensors.torch import load_file
from rapidfuzz import fuzz

# Load the model
model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    offload_buffers=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load LoRA weights
lora_weights = load_file("lora_weights/adapter_model.safetensors", device="cpu")
missing, unexpected = model.load_state_dict(lora_weights, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Known entity → prefix mapping

ENTITY_PREFIXES = {
    "InvoiceDetails": "edifact-o:",
    "VATIdentifier": "p2p-o-org:",
    "Entity6": "foaf:",
    "Entity7": "foaf:",
    "Entity8": "foaf:",
    "Entity9": "foaf:",
    "Entity10": "foaf:",
    "Entity11": "schema:",
    "Entity12": "schema:",
    "Entity13": "schema:",
    "Entity14": "schema:",
    "Entity15": "schema:",
    "Entity16": "dcat:",
    "Entity17": "dcat:",
    "Entity18": "dcat:",
    "Entity19": "dcat:",
    "Entity20": "dcat:",
    "Entity21": "epo:",
    "Entity22": "epo:",
    "Entity23": "epo:",
    "Entity24": "epo:",
    "Entity25": "epo:",
    "Entity26": "epo:",
    "Entity27": "epo:",
    "Entity28": "epo:",
    "Entity29": "epo:",
    "Entity30": "epo:",
    "Entity31": "edifact-o:",
    "Entity32": "edifact-o:",
    "Entity33": "edifact-o:",
    "Entity34": "edifact-o:",
    "Entity35": "edifact-o:",
    "Entity36": "edifact-o:",
    "Entity37": "edifact-o:",
    "Entity38": "edifact-o:",
    "Entity39": "edifact-o:",
    "Entity40": "edifact-o:",
}

# Fuzzy match to extract relevant prefixes
def extract_used_prefixes(nl_text, threshold=80):
    found_prefixes = set()
    for entity, prefix in ENTITY_PREFIXES.items():
        score = fuzz.partial_ratio(entity.lower(), nl_text.lower())
        if score >= threshold:
            found_prefixes.add(prefix)
    found_prefixes.update({":", "sh:"})
    return sorted(found_prefixes)

# Generate SHACL prefix block
def build_prefix_block(prefixes):
    base_mapping = {
        ":": "<http://mapping.example.com/>",
        "sh:": "<http://www.w3.org/ns/shacl#>",
        "xsd:": "<http://www.w3.org/2001/XMLSchema#>",
        "edifact-o:": "<https://purl.org/edifact/ontology#>",
        "p2p-o-org:": "<http://example.com/p2p-org#>",
        "foaf:": "<http://xmlns.com/foaf/0.1/>",
        "schema:": "<http://schema.org/>",
        "dcat:": "<http://www.w3.org/ns/dcat#>",
        "epo:": "<http://publications.europa.eu/ontology/eprocurement#>",
        "vcard:": "<http://www.w3.org/2006/vcard/ns#>",
    }
    prefix_lines = [f"@prefix {p} {base_mapping[p]} ." for p in prefixes if p in base_mapping]
    return "\n".join(prefix_lines)

# Input NL sentence
nl_text = (
    "For instances of edifact-o:E-Invoice, the edifact-o:belongsToProcess property must "
    "have exactly one value, which must be \"ProcessExample\"."
)

# Prepare prompt
used_prefixes = extract_used_prefixes(nl_text)
prefix_block = build_prefix_block(used_prefixes)
instruction = "Translate the following sentence strictly into SHACL syntax. Only return the SHACL code block — do not explain or elaborate.\n"
prompt = f"{prefix_block}\n\n{instruction}\n{nl_text}"

# Tokenize and run inference
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)
print("\n")
