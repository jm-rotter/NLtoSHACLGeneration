from rdflib import Graph, Namespace, URIRef, Literal
from pathlib import Path
from shaclParser import pullShapes
from prompts import build_prompt, build_reflection_prompt, INITIAL_PROMPT, FEW_SHOT_EXAMPLES
from groq import Groq
from dotenv import load_dotenv
from utils import printTranslationsToFile, printTranslationsToJSONFile
from tqdm import tqdm
from rapidfuzz import fuzz
import os
import sys

ENTITY_PREFIXES = {
    "InvoiceDetails": "edifact-o:", "VATIdentifier": "p2p-o-org:",
    "Entity6": "foaf:", "Entity7": "foaf:", "Entity8": "foaf:",
    "Entity9": "foaf:", "Entity10": "foaf:", "Entity11": "schema:",
    "Entity12": "schema:", "Entity13": "schema:", "Entity14": "schema:",
    "Entity15": "schema:", "Entity16": "dcat:", "Entity17": "dcat:",
    "Entity18": "dcat:", "Entity19": "dcat:", "Entity20": "dcat:",
    "Entity21": "epo:", "Entity22": "epo:", "Entity23": "epo:",
    "Entity24": "epo:", "Entity25": "epo:", "Entity26": "epo:",
    "Entity27": "epo:", "Entity28": "epo:", "Entity29": "epo:",
    "Entity30": "epo:", "Entity31": "edifact-o:", "Entity32": "edifact-o:",
    "Entity33": "edifact-o:", "Entity34": "edifact-o:", "Entity35": "edifact-o:"
}

def extract_used_prefixes(nl_text, threshold=80):
    found_prefixes = set()
    for entity, prefix in ENTITY_PREFIXES.items():
        if fuzz.partial_ratio(entity.lower(), nl_text.lower()) >= threshold:
            found_prefixes.add(prefix)
    found_prefixes.update({":", "sh:"})
    return sorted(found_prefixes)

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
    return "\\n".join(f"@prefix {p} {base_mapping[p]} ." for p in prefixes if p in base_mapping)



def translateShape(shape, verbose):
    prompt = build_prompt(shape, FEW_SHOT_EXAMPLES, INITIAL_PROMPT)


    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip()

    reflection_prompt = build_reflection_prompt(shape, result)

    reflection_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": reflection_prompt}]
    )
    
    translation = reflection_response.choices[0].message.content.strip()

    if verbose:
        print("Initial Prompt:")
        print(prompt)
        print("\nLLM Response:")
        print(result)
        print("\nReflection Prompt:")
        print(reflection_prompt)
        print("\nLLM Response:")
        print(translation)

    return translation




dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)


VERBOSE_FLAG = False
API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"


client = Groq(api_key=GROQ_API_KEY)


shapes = pullShapes()

translations = [] 
for shape in tqdm(shapes, desc="Translating shapes"):
    serialized = shape.serialize(format='turtle')
    if isinstance(serialized,bytes):
        serialized = serialized.decode('utf-8')
    translations.append((serialized, translateShape(serialized, VERBOSE_FLAG)))

printTranslationsToFile("shacltranslations.txt", translations)
printTranslationsToJSONFile("shacltranslations.jsonl", translations)

