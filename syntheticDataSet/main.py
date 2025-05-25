from rdflib import Graph, Namespace, URIRef, Literal
from shaclParser import pullShapes
from prompts import build_prompt, build_reflection_prompt, INITIAL_PROMPT, FEW_SHOT_EXAMPLES
from groq import Groq
from dotenv import load_dotenv
from utils import printTranslationsToFile
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



from pathlib import Path

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)


VERBOSE_FLAG = False
API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"


client = Groq(api_key=GROQ_API_KEY)


shapes = pullShapes()

translations = [] 
i = 0
for shape in shapes:
    print(f"Translated {i} shapes so far")
    i += 1
    serialized = shape.serialize(format='turtle')
    if isinstance(serialized,bytes):
        serialized = serialized.decode('utf-8')
    translations.append((serialized, translateShape(serialized, VERBOSE_FLAG)))

printTranslationsToFile("shacl_translations.txt", translations)


