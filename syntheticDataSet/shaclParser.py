from rdflib import Graph, Namespace, URIRef, Literal
import os

shaclPrefixes = """
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
"""

def pullShapes():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaclDataset.ttl')
    with open(file_path, 'r') as text:
        text_shapes = []
        curr = ""
        for line in text:
            if line.strip() == '':
                if curr.strip():
                    text_shapes.append(curr)
                curr = ""
            else:
                curr += line
        if curr.strip():
            text_shapes.append(curr)

    shapes = []
    for shape_text in text_shapes:
        g = Graph()
        g.parse(data=shaclPrefixes + shape_text, format='turtle')

        shapes.append(g)


    for i, g in enumerate(shapes):
        serialized = g.serialize(format='turtle')
        if isinstance(serialized,bytes):
            serialized = serialized.decode('utf-8')
    #    print(f"Shape {i} serialized:")
    #    print(serialized)
    #    print("------")

    return shapes

