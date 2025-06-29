from rdflib import Graph, Namespace, URIRef, Literal
from pathlib import Path
from shaclParser import pullShapes
from prompts import build_prompt, build_reflection_prompt, INITIAL_PROMPT, FEW_SHOT_EXAMPLES
from groq import Groq
from dotenv import load_dotenv
from utils import printTranslationsToFile, printTranslationsToJSONFile
from tqdm import tqdm
import os
import sys


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
starting_idx = 192
curr_idx = 0
with open("training_translations.txt", "a") as txt_file, open("training_translations.jsonl", "a") as jsonl_file:

    for shape in tqdm(shapes, desc="Translating shapes"):
        if curr_idx < starting_idx: 
           curr_idx += 1
           continue
        serialized = shape.serialize(format='turtle')
        if isinstance(serialized,bytes):
            serialized = serialized.decode('utf-8')
        nl = translateShape(serialized, VERBOSE_FLAG)

        printTranslationsToFile(txt_file, serialized, nl, starting_idx)
        printTranslationsToJSONFile(jsonl_file, serialized, nl, starting_idx)
        starting_idx += 1


