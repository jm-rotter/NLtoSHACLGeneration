def build_prompt(shape: str, few_shot_examples: str, instruction: str) -> str:
    return f"{instruction}\n\n{few_shot_examples}\n\nTranslate the following SHACL shape into natural language: formulate the response as documentation and respond only with the translation itself.\n\n{shape}"

def build_reflection_prompt(shape: str, nl_translation: str) -> str:
    return f"You just translated the following SHACL shape: \n{shape}\ninto:\n{nl_translation}\nIs the meaning clear and concise? Are all important constraints (target class, property name, cardinality, message) mentioned? If you find any missing or unclear information, rewrite and improve your answer. Only return the improved natural language sentence."


INITIAL_PROMPT = """You are an AI assistant. Your task is to convert the following SHACL shape into a clear natural language documentation sentence for a human reader.

When referring to SHACL classes and properties in your natural language output:
- Always include the full prefixed name (e.g., edifact-o:InvoiceDetails, p2p-o:VATIdentifier).
- Do NOT include the prefix for 'sh:' terms (e.g., sh:NodeShape should be just NodeShape).
- Write clear, formal documentation-style sentences that cover all key constraints."""



FEW_SHOT_EXAMPLES = """


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
For instances of edifact-o:InvoiceDetails, the edifact-o:hasDocumentNumber property must be a string with a maximum length of 12 characters.
If it exceeds 12 characters, show: "The data element 1004 in the BGM segment is too long".

SHACL(2)
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
For edifact-o:InvoiceDetails, the edifact-o:hasDocumentFunction property must be a string with exactly one value from a fixed list (e.g., "Original", "Copy", "Cancellation").
If this property is missing or invalid, display: "Data element 1225 is missing in the BGM segment, i.e. the specification of the document function".

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
If an org:FormalOrganization is not of rdf:type "http://example.com/BuyerRole", it must have a p2p-o-org:VATIdentifier property (a string with max length 14).
Otherwise, show: "The segment RFF+VA is missing for NAD+BY".

SHACL(4)
:CountryCodeValidation a sh:NodeShape;
    sh:targetClass org:FormalOrganization;
    sh:sparql [
        a sh:SPARQLConstraint;
        sh:message "The country code must be a valid ISO 3166-1 alpha-3 code";
        sh:prefixes [
            sh:declare [
                sh:prefix "org";
                sh:namespace "http://www.w3.org/ns/org#"^^xsd:anyURI;
            ]
        ];
        sh:select \"\"\"
        SELECT ?this WHERE {
            ?this org:hasCountryCode ?code .
            FILTER(STRLEN(?code) != 3)
        }
        \"\"\";
    ] 
.

Corresponding NL Translation
Each org:FormalOrganization must have a org:hasCountryCode that is exactly 3 characters long (per ISO 3166-1 alpha-3 standard).
If not, display: "The country code must be a valid ISO 3166-1 alpha-3 code".


SHACL(5)
:Rechnungsdatum 
    a sh:NodeShape;
    sh:targetClass edifact-o:InvoiceDetails;
    sh:property [
        sh:path edifact-o:invoiceDate;
        sh:datatype xsd:date;
        sh:minCount 1;
        sh:maxCount 1;
        sh:message "The invoice date (DTM+137) is required and must appear exactly once.";
    ] .

Corresponding NL Translation
For each edifact-o:InvoiceDetails, the property edifact-o:invoiceDate must be a date value and must appear exactly once. If this property is missing or appears multiple times, display: "The invoice date (DTM+137) is required and must appear exactly once."


SHACL(6)
:Lieferadresse 
    a sh:NodeShape;
    sh:targetClass org:FormalOrganization;
    sh:property [
        sh:path vcard:hasStreetAddress;
        sh:datatype xsd:string;
        sh:minCount 1;
        sh:maxCount 1;
        sh:maxLength 35;
        sh:message "Street address (3042) in NAD segment is required and should not exceed 35 characters.";
    ] .

Corresponding NL Translation
For each org:FormalOrganization, the property vcard:hasStreetAddress must be a string with at most 35 characters and must appear exactly once. If not, show: "Street address (3042) in NAD segment is required and should not exceed 35 

SHACL(7)
:SteuerID 
    a sh:NodeShape;
    sh:targetClass org:FormalOrganization;
    sh:property [
        sh:path p2p-o-org:VATIdentifier;
        sh:datatype xsd:string;
        sh:minCount 1;
        sh:maxCount 1;
        sh:maxLength 14;
        sh:message "VAT Identifier (RFF+VA) is missing or too long.";
    ] .
Corresponding NL Translation
For each org:FormalOrganization, the property p2p-o-org:VATIdentifier must be a string of up to 14 characters and appear exactly once. If missing or too long, show the error message: "VAT Identifier (RFF+VA) is missing or too long."


"""



