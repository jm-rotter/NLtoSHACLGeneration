def build_prompt(shape: str, few_shot_examples: str, instruction: str) -> str:
    return f"{instruction}\n\n{few_shot_examples}\n\nTranslate the following SHACL shape into natural language: formulate the response as documentation and respond only with the translation itself.\n\n{shape}"

def build_reflection_prompt(shape: str, nl_translation: str) -> str:
    return f"You just translated the following SHACL shape: \n{shape}\ninto:\n{nl_translation}\nIs the meaning clear and concise? Are all important constraints (target class, property name, cardinality, message) mentioned? If you find any missing or unclear information, rewrite and improve your answer. Only return the improved natural language sentence. Do not under any condition write anything other than the translation!"


INITIAL_PROMPT = """You are an AI assistant. Your task is to convert the following SHACL shape into a clear natural language documentation sentence for a human reader.

When referring to SHACL classes and properties in your natural language output:
- Always include the full prefixed name (e.g., edifact-o:InvoiceDetails, p2p-o:VATIdentifier).
- Do NOT include the prefix for 'sh:' terms (e.g., sh:NodeShape should be just NodeShape).
- Write clear, formal documentation-style sentences that cover all key constraints."""



FEW_SHOT_EXAMPLES = """


SHACL(0)
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
The SHACL shape :LaengeBelegnummer targets all instances of the class edifact-o:InvoiceDetails. It enforces a constraint on the property edifact-o:hasDocumentNumber, requiring its value to be a string (xsd:string) with a maximum length of 12 characters. If this constraint is violated, the validation will produce the message:
"The data element 1004 in the BGM segment is too long."

SHACL(1)
:Dokumentfunktion 
    a sh:NodeShape; 
    sh:targetClass edifact-o:InvoiceDetails; 
    sh:property [
        sh:path edifact-o:hasDocumentFunction;
        sh:datatype xsd:string;
        sh:minCount 1;
        sh:maxCount 1;
        sh:in ("Cancellation" "Replacement" "Duplicate" "Original" "Copy" "Additional transfer" "Stornierung" "Ersatz" "Duplikat" "Original" "Kopie" "Zusaetzliche Uebertragung");
        sh:message "Data element 1225 is missing in the BGM segment, i.e. the specification of the document function";
    ]
.

Corresponding NL Translation
The SHACL shape :Dokumentfunktion applies to all instances of the class edifact-o:InvoiceDetails. It specifies a property constraint on edifact-o:hasDocumentFunction that requires exactly one value (minCount and maxCount set to 1). The value must be a string (xsd:string) and must be one of the predefined allowed values:
"Cancellation", "Replacement", "Duplicate", "Original", "Copy", "Additional transfer", "Stornierung", "Ersatz", "Duplikat", "Original", "Kopie", "Zusaetzliche Uebertragung."
If this property is missing or its value is outside this list, validation will fail with the message:
"Data element 1225 is missing in the BGM segment, i.e. the specification of the document function."

SHACL(2)
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

The SHACL shape :UmsatzsteuernummerKaeufer applies to all instances of the class org:FormalOrganization. It specifies a validation rule with an sh:or condition. Either the organization does not have the RDF type "http://example.com/BuyerRole" or, if it does have the "http://example.com/BuyerRole" type, then the property p2p-o-org:VATIdentifier must be present exactly once (minimum count 1 and maximum count 1), and its value must be a string with a maximum length of 14 characters.
If neither of these conditions is met (i.e., the organization is a buyer but the VAT identifier is missing or invalid), the validation will fail with the message: "The segment RFF+VA is missing for NAD+BY."


SHACL(3)

:edifact-oEntity50Shape a NodeShape;
    targetClass edifact-o:Entity50;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
The SHACL shape :edifact-oEntity50Shape applies to all instances of the class edifact-o:Entity50. It specifies a property constraint on edifact-o:invoiceDate that requires exactly one value (with minCount and maxCount set to 1). The value must be a date (xsd:date).
If this property is missing or appears more than once, validation will fail with the message:
"The value must appear exactly once."

"""



