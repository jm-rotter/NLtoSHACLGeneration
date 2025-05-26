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


SHACL(0')
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

SHACL(0)
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

SHACL(1)
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

""" 

"""

SHACL(2)
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

SHACL(3)
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




SHACL(4)
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

SHACL(5)
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



#Domain: People, names, social info
Prefixes: foaf:, schema:, xsd:

SHACL(6)
:foafEntity6Shape a NodeShape;
    targetClass foaf:Entity6;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        in ("A" "B" "C" "D");
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each foaf:Entity6, the property schema:email must be of type string and must appear at least once. It must be one of the following values: A, B, C, D. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(7)
:foafEntity7Shape a NodeShape;
    targetClass foaf:Entity7;
    property [
        path foaf:name;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        in ("A" "B" "C" "D");
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each foaf:Entity7, the property foaf:name must be of type string and must appear at least once. It must be one of the following values: A, B, C, D. If this condition is not met, display: "This property is required and must be valid."

SHACL(8)
:foafEntity8Shape a NodeShape;
    targetClass foaf:Entity8;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 0;
        maxCount 2;
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each foaf:Entity8, the property edifact-o:hasDocumentFunction must be of type string and must appear between 0 and 2 times. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(9)

:foafEntity9Shape a NodeShape;
    targetClass foaf:Entity9;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        in ("A" "B" "C" "D");
        message "The value must match the expected format.";
    ] .
NL Translation:
For each foaf:Entity9, the property edifact-o:hasDocumentFunction must be of type string and must appear at least once. It must be one of the following values: A, B, C, D. If this condition is not met, display: "The value must match the expected format."

SHACL(10)

:foafEntity10Shape a NodeShape;
    targetClass foaf:Entity10;
    property [
        path vcard:hasStreetAddress;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "The value is missing or incorrectly formatted.";
    ] .
NL Translation:
For each foaf:Entity10, the property vcard:hasStreetAddress must be of type string and must appear at most once. If this condition is not met, display: "The value is missing or incorrectly formatted."


#####
Domain: Generic web schemas (email, date, etc.)
Prefixes: schema:, xsd:

SHACL(11)
:schemaEntity11Shape a NodeShape;
    targetClass schema:Entity11;
    property [
        path schema:birthDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each schema:Entity11, the property schema:birthDate must be of type date and must appear exactly once. If this condition is not met, display: "The value must appear exactly once."

SHACL(12)

:schemaEntity12Shape a NodeShape;
    targetClass schema:Entity12;
    property [
        path schema:gender;
        datatype xsd:string;
        minCount 0;
        maxCount 5;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each schema:Entity12, the property schema:gender must be of type string and must appear between 0 and 5 times. If this condition is not met, display: "This property is required and must be valid."

SHACL(13)

:schemaEntity13Shape a NodeShape;
    targetClass schema:Entity13;
    property [
        path schema:identifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        in ("A" "B" "C");
        message "The value must match the expected format.";
    ] .
NL Translation:
For each schema:Entity13, the property schema:identifier must be of type string and must appear at least once. It must be one of the following values: A, B, C. If this condition is not met, display: "The value must match the expected format."

SHACL(14)

:schemaEntity14Shape a NodeShape;
    targetClass schema:Entity14;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each schema:Entity14, the property schema:email must be of type string and must appear at least once. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(15)

:schemaEntity15Shape a NodeShape;
    targetClass schema:Entity15;
    property [
        path schema:gender;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each schema:Entity15, the property schema:gender must be of type string and must appear at most once. If this condition is not met, display: "This property is required and must be valid."




### Public Data Catalogs
Domain: Open data, DCAT, catalog validation
Prefixes: dcat:, vcard:, xsd:

SHACL(16)

:dcatEntity16Shape a NodeShape;
    targetClass dcat:Entity16;
    property [
        path vcard:hasStreetAddress;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        message "The value is missing or incorrectly formatted.";
    ] .
NL Translation:
For each dcat:Entity16, the property vcard:hasStreetAddress must be of type string and must appear exactly once. If this condition is not met, display: "The value is missing or incorrectly formatted."

SHACL(17)

:dcatEntity17Shape a NodeShape;
    targetClass dcat:Entity17;
    property [
        path vcard:hasStreetAddress;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        in ("A" "B" "C" "D");
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each dcat:Entity17, the property vcard:hasStreetAddress must be of type string and must appear at most once. It must be one of the following values: A, B, C, D. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(18)

:dcatEntity18Shape a NodeShape;
    targetClass dcat:Entity18;
    property [
        path schema:identifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        message "The value must match the expected format.";
    ] .
NL Translation:
For each dcat:Entity18, the property schema:identifier must be of type string and must appear at least once. If this condition is not met, display: "The value must match the expected format."

SHACL(19)

:dcatEntity19Shape a NodeShape;
    targetClass dcat:Entity19;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 1;
        maxCount 5;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each dcat:Entity19, the property schema:email must be of type string and must appear between 1 and 5 times. If this condition is not met, display: "This property is required and must be valid."

SHACL(20)

:dcatEntity20Shape a NodeShape;
    targetClass dcat:Entity20;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each dcat:Entity20, the property schema:email must be of type string and must appear at most once. If this condition is not met, display: "The value must appear exactly once."


###
Domain: Procurement, contracts, law
Prefixes: epo:, gr:, cpv:, xsd:


SHACL(21)

:epoEntity21Shape a NodeShape;
    targetClass epo:Entity21;
    property [
        path schema:identifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        in ("A" "B" "C");
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each epo:Entity21, the property schema:identifier must be of type string and must appear at least once. It must be one of the following values: A, B, C. If this condition is not met, display: "This property is required and must be valid."

SHACL(22)

:epoEntity22Shape a NodeShape;
    targetClass epo:Entity22;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 0;
        maxCount 5;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each epo:Entity22, the property schema:email must be of type string and must appear between 0 and 5 times. If this condition is not met, display: "The value must appear exactly once."

SHACL(23)

:epoEntity23Shape a NodeShape;
    targetClass epo:Entity23;
    property [
        path schema:gender;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        in ("A" "B");
        message "The value is missing or incorrectly formatted.";
    ] .
NL Translation:
For each epo:Entity23, the property schema:gender must be of type string and must appear exactly once. It must be one of the following values: A, B. If this condition is not met, display: "The value is missing or incorrectly formatted."

 SHACL(24)

:epoEntity24Shape a NodeShape;
    targetClass epo:Entity24;
    property [
        path schema:birthDate;
        datatype xsd:date;
        minCount 0;
        maxCount 2;
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each epo:Entity24, the property schema:birthDate must be of type date and must appear between 0 and 2 times. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(25)

:epoEntity25Shape a NodeShape;
    targetClass epo:Entity25;
    property [
        path schema:identifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each epo:Entity25, the property schema:identifier must be of type string and must appear at least once. If this condition is not met, display: "This property is required and must be valid."

SHACL(26)

:epoEntity26Shape a NodeShape;
    targetClass epo:Entity26;
    property [
        path schema:identifier;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        in ("X", "Y", "Z");
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each epo:Entity26, the property schema:identifier must be of type string and must appear exactly once. It must be one of the following values: X, Y, Z. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(27)

:epoEntity27Shape a NodeShape;
    targetClass epo:Entity27;
    property [
        path schema:gender;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each epo:Entity27, the property schema:gender must be of type string and must appear at most once. If this condition is not met, display: "The value must appear exactly once."

SHACL(28)

:epoEntity28Shape a NodeShape;
    targetClass epo:Entity28;
    property [
        path schema:birthDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each epo:Entity28, the property schema:birthDate must be of type date and must appear exactly once. If this condition is not met, display: "This property is required and must be valid."

SHACL(29)

:epoEntity29Shape a NodeShape;
    targetClass epo:Entity29;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 0;
        maxCount 3;
        in ("A" "B");
        message "The value must match the expected format.";
    ] .
NL Translation:
For each epo:Entity29, the property schema:email must be of type string and must appear between 0 and 3 times. It must be one of the following values: A, B. If this condition is not met, display: "The value must match the expected format."

SHACL(30)

:epoEntity30Shape a NodeShape;
    targetClass epo:Entity30;
    property [
        path schema:email;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each epo:Entity30, the property schema:email must be of type string and must appear exactly once. If this condition is not met, display: "The value must appear exactly once."

###
Domain: edifact, p2p, business validation
Prefixes: edifact-o:, p2p-o:, xsd:

SHACL(31)

:edifact-oEntity31Shape a NodeShape;
    targetClass edifact-o:Entity31;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each edifact-o:Entity31, the property edifact-o:invoiceDate must be of type date and must appear exactly once. If this condition is not met, display: "This property is required and must be valid."

SHACL(32)

:edifact-oEntity32Shape a NodeShape;
    targetClass edifact-o:Entity32;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        in ("DE", "FR", "IT");
        message "The value must match the expected format.";
    ] .
NL Translation:
For each edifact-o:Entity32, the property p2p-o-org:VATIdentifier must be of type string and must appear at least once. It must be one of the following values: DE, FR, IT. If this condition is not met, display: "The value must match the expected format."

SHACL(33)

:edifact-oEntity33Shape a NodeShape;
    targetClass edifact-o:Entity33;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "The value is missing or incorrectly formatted.";
    ] .
NL Translation:
For each edifact-o:Entity33, the property p2p-o-org:VATIdentifier must be of type string and must appear at most once. If this condition is not met, display: "The value is missing or incorrectly formatted."

SHACL(34)

:edifact-oEntity34Shape a NodeShape;
    targetClass edifact-o:Entity34;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        in ("Original", "Copy", "Cancellation");
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each edifact-o:Entity34, the property edifact-o:hasDocumentFunction must be of type string and must appear exactly once. It must be one of the following values: Original, Copy, Cancellation. If this condition is not met, display: "This property is required and must be valid."

SHACL(35)

:edifact-oEntity35Shape a NodeShape;
    targetClass edifact-o:Entity35;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 1;
        maxCount 5;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each edifact-o:Entity35, the property p2p-o-org:VATIdentifier must be of type string and must appear between 1 and 5 times. If this condition is not met, display: "The value must appear exactly once."

SHACL(36)

:edifact-oEntity36Shape a NodeShape;
    targetClass edifact-o:Entity36;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 2;
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each edifact-o:Entity36, the property edifact-o:invoiceDate must be of type date and must appear at least once. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(37)

:edifact-oEntity37Shape a NodeShape;
    targetClass edifact-o:Entity37;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "The value must match the expected format.";
    ] .
NL Translation:
For each edifact-o:Entity37, the property edifact-o:hasDocumentFunction must be of type string and must appear at most once. If this condition is not met, display: "The value must match the expected format."

SHACL(38)

:edifact-oEntity38Shape a NodeShape;
    targetClass edifact-o:Entity38;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each edifact-o:Entity38, the property p2p-o-org:VATIdentifier must be of type string and must appear at most once. If this condition is not met, display: "The value must appear exactly once."

SHACL(39)

:edifact-oEntity39Shape a NodeShape;
    targetClass edifact-o:Entity39;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 1;
        maxCount 3;
        in ("A", "B", "C", "D");
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each edifact-o:Entity39, the property edifact-o:hasDocumentFunction must be of type string and must appear at least once. It must be one of the following values: A, B, C, D. If this condition is not met, display: "This property is required and must be valid."

SHACL(40)

:edifact-oEntity40Shape a NodeShape;
    targetClass edifact-o:Entity40;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each edifact-o:Entity40, the property edifact-o:invoiceDate must be of type date and must appear exactly once. If this condition is not met, display: "The value must appear exactly once."

###
Domain: edifact, p2p, business validation
Prefixes: edifact-o:, p2p-o:, xsd:

SHACL(41)

:edifact-oEntity41Shape a NodeShape;
    targetClass edifact-o:Entity41;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        in ("Original", "Cancellation", "Replacement");
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each edifact-o:Entity41, the property edifact-o:hasDocumentFunction must be of type string and must appear exactly once. It must be one of the following values: Original, Cancellation, Replacement. If this condition is not met, display: "This property is required and must be valid."

SHACL(42)

:edifact-oEntity42Shape a NodeShape;
    targetClass edifact-o:Entity42;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 0;
        maxCount 1;
        message "The value is missing or incorrectly formatted.";
    ] .
NL Translation:
For each edifact-o:Entity42, the property edifact-o:invoiceDate must be of type date and must appear at most once. If this condition is not met, display: "The value is missing or incorrectly formatted."

SHACL(43)

:edifact-oEntity43Shape a NodeShape;
    targetClass edifact-o:Entity43;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each edifact-o:Entity43, the property p2p-o-org:VATIdentifier must be of type string and must appear at least once. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(44)

:edifact-oEntity44Shape a NodeShape;
    targetClass edifact-o:Entity44;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 1;
        maxCount 1;
        message "The value must match the expected format.";
    ] .
NL Translation:
For each edifact-o:Entity44, the property p2p-o-org:VATIdentifier must be of type string and must appear exactly once. If this condition is not met, display: "The value must match the expected format."

SHACL(45)

:edifact-oEntity45Shape a NodeShape;
    targetClass edifact-o:Entity45;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        in ("DE", "FR", "IT");
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each edifact-o:Entity45, the property p2p-o-org:VATIdentifier must be of type string and must appear at most once. It must be one of the following values: DE, FR, IT. If this condition is not met, display: "This property is required and must be valid."

SHACL(46)

:edifact-oEntity46Shape a NodeShape;
    targetClass edifact-o:Entity46;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 3;
        message "The value must appear exactly once.";
    ] .
NL Translation:
For each edifact-o:Entity46, the property edifact-o:invoiceDate must be of type date and must appear at least once. If this condition is not met, display: "The value must appear exactly once."

SHACL(47)

:edifact-oEntity47Shape a NodeShape;
    targetClass edifact-o:Entity47;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        in ("2023-01-01", "2023-12-31");
        message "The value is missing or incorrectly formatted.";
    ] .
NL Translation:
For each edifact-o:Entity47, the property edifact-o:invoiceDate must be of type date and must appear exactly once. It must be one of the following values: 2023-01-01, 2023-12-31. If this condition is not met, display: "The value is missing or incorrectly formatted."

SHACL(48)

:edifact-oEntity48Shape a NodeShape;
    targetClass edifact-o:Entity48;
    property [
        path edifact-o:hasDocumentFunction;
        datatype xsd:string;
        minCount 0;
        maxCount 1;
        in ("X", "Y");
        message "Provide a value that meets the criteria.";
    ] .
NL Translation:
For each edifact-o:Entity48, the property edifact-o:hasDocumentFunction must be of type string and must appear at most once. It must be one of the following values: X, Y. If this condition is not met, display: "Provide a value that meets the criteria."

SHACL(49)

:edifact-oEntity49Shape a NodeShape;
    targetClass edifact-o:Entity49;
    property [
        path p2p-o-org:VATIdentifier;
        datatype xsd:string;
        minCount 1;
        maxCount 2;
        message "This property is required and must be valid.";
    ] .
NL Translation:
For each edifact-o:Entity49, the property p2p-o-org:VATIdentifier must be of type string and must appear at least once. If this condition is not met, display: "This property is required and must be valid."

SHACL(50)

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
For each edifact-o:Entity50, the property edifact-o:invoiceDate must be of type date and must appear exactly once. If this condition is not met, display: "The value must appear exactly once."

"""



