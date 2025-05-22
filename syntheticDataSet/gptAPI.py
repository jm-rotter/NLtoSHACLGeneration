from openai import OpenAI
from rdflib import Graph, Namespace, URIRef, Literal
from shaclParser import pullShapes
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL="gpt-4o"

INITIAL_PROMPT = """
You are an AI assistent. Your task is to convert the following SHACL shape into a clear natural language documentation sentence for a human reader.
"""

FEW_SHOT_EXAMPLES = """
Few Shot Examples

SHACL
:LaengeBelegnummer 
    a sh:NodeShape;
    sh:targetClass edifact-o:InvoiceDetails;
    sh:property [
        sh:path edifact-o:hasDocumentNumber;
        sh:datatype xsd:string;
        sh:maxLength 12;
        sh:message "The data element 1004 in the BGM segment is too long";
    ]
.

Corresponding NL Translation

For the class InvoiceDetails, the property hasDocumentNumber (which represents the document number) must be a string with a maximum length of 12 characters. 
If the length exceeds 12 characters, the following message should be shown: "The data element 1004 in the BGM segment is too long".
"""


def buildPrompt(shape):
    return f"""
    Translate the following SHACL shape into natural language: formulate the response as documentation and respond only with the translation itself. 

    {shape}
    """

def buildRPrompt(shape, naturalLanguageTranslation):
    return f"""
    You just translated the following SHACL shape:
    {shape}

    into:

    {naturalLanguageTranslation}

    Is the meaning clear and concise? Are all important constraints (target class, property name, cardinality, message) mentioned? 
    If you find any missing or unclear information, rewrite and improve your answer.
    Only return the improved natural language sentence.
    """


client = OpenAI(api_key=API_KEY)


def translateSHAPE(shape):

    response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": INITIAL_PROMPT + FEW_SHOT_EXAMPLES + buildPrompt(shape)}]
            )
    naturalLanguageTranslation = response.choices[0].message.content
    print(naturalLanguageTranslation)

    reflectionResponse = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": buildRPrompt(shape, naturalLanguageTranslation)}]
            )
    print(reflectionResponse.choices[0].message.content)

    return reflectionResponse.choices[0].message.content 


shapes = pullShapes()

translations = {}

for shape in shapes:
    translateSHAPE(shape) 
    break





