
from rdflib import Graph, Namespace, URIRef, Literal
from shaclParser import pullShapes
from dotenv import load_dotenv
from groq import Groq
import os

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#MODEL="gpt-4o"
MODEL="llama3-70b-8192"

INITIAL_PROMPT = """
You are an AI assistent. Your task is to convert the following SHACL shape into a clear natural language documentation sentence for a human reader.
"""

FEW_SHOT_EXAMPLES = """
Few Shot Examples

SHACL(1)
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


SHACL(2)
:Dokumentfunktion 
a sh:NodeShape; sh:targetClass
edifact-o:InvoiceDetails; sh:property
    [ sh:path edifact-o:hasDocumentFunction;
         sh:datatype xsd:string;
        sh:minCount 1;
        sh:maxCount 1;
        sh:in ("Cancellation" "Replacement" "Duplicate" "Original" "Copy" "Additional transfer" "Stornierung" "Ersatz" "Duplikat" "Original" "Kopie" "Zusaetzliche Uebertragung");
        sh:message "Data element 1225 is missing in the BGM segment, i.e. the specification of the document function";
    ]       
.

Corresponding NL Translation
For each InvoiceDetails, the hasDocumentFunction property must be a string with exactly one value from a predefined list such as "Original", "Copy", "Cancellation", etc.
If missing or invalid, show the message: "Data element 1225 is missing in the BGM segment, i.e. the specification of the document function".

SHACL(3)
:UmsatzsteuernummerKaeufer 
    a sh:NodeShape;
    sh:targetClass org:FormalOrganization;
    sh:message "The segment RFF+VA is missing for NAD+BY";
    sh:or (
        [ sh:not [
            a sh:PropertyShape;
            sh:path rdf:type;
            sh:hasValue "http://example.com/BuyerRole";
        ] ]
        [
            sh:path p2p-o-org:VATIdentifier;
            sh:minCount 1;
            sh:maxCount 1;
            sh:maxLength 14;
        ]
    ) 
.

Corresponding NL Translation
For each FormalOrganization not identified as a Buyer, it must have a VATIdentifier property with exactly one value that is a string up to 14 characters long.
If neither applies, display: “The segment RFF+VA is missing for NAD+BY.”

SHACL(4)
:SumNetPrice a sh:NodeShape;
    sh:targetClass edifact-o:InvoiceDetails;
    sh:sparql [
        a sh:SPARQLConstraint;
        sh:message "hasTotalLineItemAmount must equal the sum of hasLineItemAmount";
        sh:select \"\"\"SELECT $this (edifact-o:hasTotalLineItemAmount AS ?path) (?totalAmount AS ?value)
         WHERE { 
          $this a edifact-o:InvoiceDetails ;
          edifact-o:hasTotalLineItemAmount ?totalAmount .
        {
        SELECT $this (SUM(?itemAmount) AS ?sum)
        WHERE {
            ?item edifact-o:isItemOf ?invoice ;
                  edifact-o:hasLineItemAmount ?itemAmount .
            ?invoice edifact-o:hasInvoiceDetails $this .
        } GROUP BY $this
    }
    FILTER (?sum != ?totalAmount)
}\"\"\";

.

Corresponding NL Translation
The hasTotalLineItemAmount of an InvoiceDetails must be equal to the sum of all hasLineItemAmount values across its items.
If this condition is not met, show: “hasTotalLineItemAmount must equal the sum of hasLineItemAmount.”

"""




def buildPrompt(shape):
    return f"""\nTranslate the following SHACL shape into natural language: formulate the response as documentation and respond only with the translation itself. 

{shape}"""

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


#client = OpenAI(api_key=API_KEY)
client = Groq(api_key=GROQ_API_KEY)

def translateSHAPE(shape):

    response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": INITIAL_PROMPT + FEW_SHOT_EXAMPLES + buildPrompt(shape)}]
            )
    naturalLanguageTranslation = response.choices[0].message.content
    
    #print("INPUT: \n" + INITIAL_PROMPT + FEW_SHOT_EXAMPLES + buildPrompt(shape))
    #print("\n\n")
    #print("OUTPUT \n" + naturalLanguageTranslation)
    #print("\n\n")

    reflectionResponse = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": buildRPrompt(shape, naturalLanguageTranslation)}]
            )

    #print("INPUT: \n" + buildRPrompt(shape, naturalLanguageTranslation))
    #print("\n\n")
    #print("OUTPUT \n" + reflectionResponse.choices[0].message.content)
    #print("\n\n")
    return reflectionResponse.choices[0].message.content 


shapes = pullShapes()

translations = [] 
i = 0
for shape in shapes:
    print(i)
    i += 1
    serialized = shape.serialize(format='turtle')
    if isinstance(serialized,bytes):
        serialized = serialized.decode('utf-8')
    translations.append((serialized, translateSHAPE(serialized)))


file = open("shacl_translations.txt", "w", encoding="utf-8")


for i, (shaclTranslation, nlTranslation) in enumerate(translations):
    file.write(f"Translation #{i}\n")
    file.write(shaclTranslation)
    file.write("\n")
    file.write(nlTranslation)
    file.write("\n----\n\n")
    




