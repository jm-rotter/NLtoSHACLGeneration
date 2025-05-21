from rdflib import Graph, Namespace, URIRef, Literal

# Load the snippet (could be from file or string)
shacl_ttl = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <http://mapping.example.com/> .
@prefix cmns-id: <https://www.omg.org/spec/Commons/Identifiers/> .
@prefix d2rq: <http://www.wiwiss.fu-berlin.de/suhl/bizer/D2RQ/0.1#> .
@prefix dc: <http://purl.org/dc/elements/1.1#> .
@prefix edifact-o: <https://purl.org/edifact/ontology#> .
@prefix eli: <http://publications.europa.eu/resource/dataset/eli/> .
@prefix fnml: <http://semweb.mmlab.be/ns/fnml#> .
@prefix fno: <https://w3id.org/function/ontology#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix frapo: <http://purl.org/cerif/frapo/> .
@prefix org: <http://www.w3.org/ns/org#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix p2p-o: <https://purl.org/p2p-o#> .
@prefix p2p-o-doc: <https://purl.org/p2p-o/document#> .
@prefix p2p-o-doc-line: <https://purl.org/p2p-o/documentline#> .
@prefix p2p-o-inv: <https://purl.org/p2p-o/invoice#> .
@prefix p2p-o-item: <https://purl.org/p2p-o/item#> .
@prefix p2p-o-org: <https://purl.org/p2p-o/organization#> .
@prefix ql: <http://semweb.mmlab.be/ns/ql#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rml: <http://semweb.mmlab.be/ns/rml#> .
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix schema: <http://schema.org/> .
@prefix vcard: <http://www.w3.org/2006/vcard/ns#> .


:Dokumentenname 
    a sh:NodeShape;
    sh:targetClass edifact-o:InvoiceDetails;
    sh:property [
        sh:path edifact-o:hasDocumentType;
        sh:datatype xsd:string;
        sh:minCount 1;
        sh:maxCount 1;
        sh:in ("Commercial invoice" "Credit advice" "Value credit" "Value debit" "Handelsrechnung" "Gutschriftsanzeige" "Wertgutschrift" "Wertbelastung");
        sh:message "Data element 1001 is missing in the BGM segment";
    ]
.
"""

# Parse into RDF graph
g = Graph()
g.parse(data=shacl_ttl, format="turtle")

print("Namespaces in the graph:")
for prefix, ns in g.namespace_manager.namespaces():
    print(f"{prefix}: {ns}")

print("\nTriples with prefixes:")

for s, p, o in g:
    # Convert predicate to prefixed form
    try:
        prefix, namespace, local_name = g.namespace_manager.compute_qname(p)
        pred_prefixed = f"{prefix}:{local_name}"
    except Exception:
        pred_prefixed = str(p)

    # For objects that are URIs, get prefixed form if possible
    if isinstance(o, URIRef):
        try:
            prefix_o, ns_o, local_o = g.namespace_manager.compute_qname(o)
            obj_prefixed = f"{prefix_o}:{local_o}"
        except Exception:
            obj_prefixed = str(o)
    elif isinstance(o, Literal):
        obj_prefixed = f'"{o}"'
    else:
        obj_prefixed = str(o)

    print(f"Subject: {s}")
    print(f"Predicate: {pred_prefixed}")
    print(f"Object: {obj_prefixed}")
    print("---")

