from groq import Groq
from dotenv import load_dotenv
import os

# Load API key from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"

client = Groq(api_key=GROQ_API_KEY)

def build_prompt(shape: str, few_shot_examples: str, instruction: str) -> str:
    return f"{instruction}\n\n{few_shot_examples}\n\nTranslate the following SHACL shape into natural language:\n\n{shape}"

def build_reflection_prompt(shape: str, nl_translation: str) -> str:
    return f"""You just translated the following SHACL shape:
{shape}

into:

{nl_translation}

Is the meaning clear and concise? Are all important constraints (target class, property name, cardinality, message) mentioned? 
If you find any missing or unclear information, rewrite and improve your answer.
Only return the improved natural language sentence."""

def translate_shape_with_fewshot(shape: str, few_shot_examples: str, instruction: str) -> str:
    prompt = build_prompt(shape, few_shot_examples, instruction)

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

    return reflection_response.choices[0].message.content.strip()