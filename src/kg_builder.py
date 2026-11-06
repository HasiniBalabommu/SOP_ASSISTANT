"""
kg_builder.py — extracts industrial entities (machines, fault codes,
SOPs) and their relationships from chunked text, building an
incrementally-updatable knowledge graph using networkx.
"""

import json
import re
import pickle
from pathlib import Path

import networkx as nx

from src import config

KG_PATH = config.DATA_DIR / "kg" / "graph.pkl"

# --- Pattern-based entity extraction ---
FAULT_CODE_PATTERN = re.compile(r"\b[A-Z]\d{2,4}\b")            # e.g. E104
MACHINE_PATTERN = re.compile(r"\b(Compressor|Pump|Generator|Motor|Fan)\s+[A-Z0-9]+\b", re.IGNORECASE)


def load_graph() -> nx.DiGraph:
    KG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if KG_PATH.exists():
        with open(KG_PATH, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()


def save_graph(graph: nx.DiGraph) -> None:
    KG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KG_PATH, "wb") as f:
        pickle.dump(graph, f)


def extract_entities(text: str) -> dict:
    """Returns {'machines': [...], 'fault_codes': [...]} found in this chunk."""
    machines = list(set(MACHINE_PATTERN.findall(text)))
    fault_codes = list(set(FAULT_CODE_PATTERN.findall(text)))
    return {"machines": machines, "fault_codes": fault_codes}


def add_chunks_file(chunks_json_path: Path) -> int:
    """
    Reads a chunks JSON file, extracts entities from each chunk, and
    adds nodes/edges into the persistent graph. Adding the same
    entity/edge twice is safe — networkx just no-ops on duplicates.
    """
    with open(chunks_json_path, "r") as f:
        chunks = json.load(f)
    if not chunks:
        return 0

    graph = load_graph()
    edges_added = 0

    for c in chunks:
        entities = extract_entities(c["text"])
        source_doc = c["source"]

        # Document node
        graph.add_node(source_doc, type="document")

        for machine in entities["machines"]:
            graph.add_node(machine, type="machine")
            graph.add_edge(source_doc, machine, relation="mentions")

        for code in entities["fault_codes"]:
            graph.add_node(code, type="fault_code")
            graph.add_edge(source_doc, code, relation="mentions")
            edges_added += 1

            # Link machines mentioned in the SAME chunk to this fault code
            for machine in entities["machines"]:
                graph.add_edge(machine, code, relation="has_fault")
                edges_added += 1

    save_graph(graph)
    return edges_added


def query_related(entity: str) -> list[tuple[str, str, str]]:
    """Returns (source, relation, target) triples connected to this entity."""
    graph = load_graph()
    if entity not in graph:
        return []
    results = []
    for _, target, data in graph.out_edges(entity, data=True):
        results.append((entity, data.get("relation", "related_to"), target))
    for source, _, data in graph.in_edges(entity, data=True):
        results.append((source, data.get("relation", "related_to"), entity))
    return results


if __name__ == "__main__":
    chunk_files = list(config.CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        print(f"No chunk files in {config.CHUNKS_DIR} — run ingest.py first.")
    else:
        total = 0
        for cf in chunk_files:
            n = add_chunks_file(cf)
            print(f"Processed {cf.name}: {n} relationships added")
            total += n

        graph = load_graph()
        print(f"\nGraph now has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        print("\nTest query: relationships for 'E104'")
        for triple in query_related("E104"):
            print(f"  {triple[0]} --{triple[1]}--> {triple[2]}")