"""
kg_retriever.py — read-only queries against the knowledge graph built
by kg_builder.py. Kept separate from kg_builder.py so "writing" the
graph and "reading" from it are independent responsibilities.
"""

from src import kg_builder


def query_related(entity: str) -> list[tuple[str, str, str]]:
    """Returns (source, relation, target) triples connected to this entity."""
    graph = kg_builder.load_graph()
    if entity not in graph:
        return []
    results = []
    for _, target, data in graph.out_edges(entity, data=True):
        results.append((entity, data.get("relation", "related_to"), target))
    for source, _, data in graph.in_edges(entity, data=True):
        results.append((source, data.get("relation", "related_to"), entity))
    return results


def get_all_entities_by_type(entity_type: str) -> list[str]:
    """Returns all entities of a given type (e.g. 'machine', 'fault_code', 'document')."""
    graph = kg_builder.load_graph()
    return [n for n, data in graph.nodes(data=True) if data.get("type") == entity_type]


def graph_summary() -> dict:
    """Quick stats about the current graph — useful for debugging/demo."""
    graph = kg_builder.load_graph()
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "machines": get_all_entities_by_type("machine"),
        "fault_codes": get_all_entities_by_type("fault_code"),
    }


if __name__ == "__main__":
    print("Graph summary:", graph_summary())

    print("\nTest query: relationships for 'E205'")
    for triple in query_related("E205"):
        print(f"  {triple[0]} --{triple[1]}--> {triple[2]}")