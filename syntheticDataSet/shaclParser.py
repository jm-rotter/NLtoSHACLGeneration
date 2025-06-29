from rdflib import Graph, RDF, Namespace
from rdflib.term import BNode
import os

SH = Namespace("http://www.w3.org/ns/shacl#")

def pullShapes(ttl_filename: str = "shacldataset.ttl"):
    """
    Loads the full TTL into one graph, then for each subject S that has
    rdf:type sh:NodeShape, creates a subgraph containing:
      - all triples (S, p, o)
      - for any blank node o that is a property shape, all triples (o, p2, o2)
    Returns a list of rdflib.Graphs.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, ttl_filename)

    # 1) Load the entire dataset
    full = Graph()
    full.parse(path, format="turtle")

    shapes = []
    for shape in full.subjects(RDF.type, SH.NodeShape):
        g = Graph()
        # Re-bind all namespace prefixes
        for prefix, uri in full.namespaces():
            g.bind(prefix, uri)

        # Copy triples about the shape
        for p, o in full.predicate_objects(shape):
            g.add((shape, p, o))

            # If the object is a blank node, copy its own triples
            if isinstance(o, BNode):
                for p2, o2 in full.predicate_objects(o):
                    g.add((o, p2, o2))
                    # And handle one more level of nesting:
                    if isinstance(o2, BNode):
                        for p3, o3 in full.predicate_objects(o2):
                            g.add((o2, p3, o3))

        shapes.append(g)

    return shapes
