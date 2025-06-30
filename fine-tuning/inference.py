from rag import rag
from unsloth import FastLanguageModel
import torch
import json
from safetensors.torch import load_file

model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    offload_buffers=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

lora_weights = load_file("lora_weights/adapter_model.safetensors", device="cpu")
missing, unexpected = model.load_state_dict(lora_weights, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


to_translate = """The SHACL shape :Dokumentenname applies to all instances of the class edifact-o:InvoiceDetails, specifying that the property edifact-o:hasDocumentType must be a string with a value of either \"Commercial invoice\", \"Credit advice\", \"Value credit\", \"Value debit\", \"Handelsrechnung\", \"Gutschriftsanzeige\", \"Wertgutschrift\", or \"Wertbelastung\", and must occur exactly once, with validation failure resulting in the message \"Data element 1001 is missing in the BGM segment\" if this constraint is not met."""

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


def load_translations_from_json(filename):
    prompts = []
    responses = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            prompts.append(item["prompt"])
            responses.append(item["response"])
    return prompts, responses


def printTranslationsToFile(file, shaclTranslation, nlTranslation, generated, i):
    file.write(f"Translation #{i}\n")
    file.write(nlTranslation)
    file.write("\nGenerated:\n")
    file.write(generated)
    file.write("\nOriginal Ground Truth:\n")
    file.write(shaclTranslation)
    file.write("\n----\n\n")

def printTranslationsToJSONFile(file, shacl, nl, generated):
    json_line = json.dumps({"NL": nl, "GT": shacl, "Generated": generated})
    file.write(json_line + "\n")

prompts, responses = load_translations_from_json("shacltranslations.jsonl")

with open("mistral7b.txt", "w", encoding="utf-8") as txt_file, open("mistral7b.jsonl", "w", encoding="utf-8") as json_file:
    for i, (prompt, response) in enumerate(zip(prompts, responses)):

        inputs = tokenizer(genprompt(prompt), return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=4000,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        printTranslationsToFile(txt_file, response, prompt, generated_text, i)
        printTranslationsToJSONFile(json_file, response, prompt, generated_text)


