"""
Standalone community detection script.
Loads graph from checkpoint JSON (saved by kt_gen.py before community detection),
runs FastTreeComm, and saves the final ntsb_full_new.json.

Usage:
    OPENBLAS_NUM_THREADS=1 python run_community_detection.py
"""

import json
import os
import sys
import time

import networkx as nx

# Must be set before importing numpy/scipy/sentence-transformers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from config import get_config
from utils import tree_comm
from utils.logger import logger

DATASET_NAME = "ntsb_full"
CHECKPOINT_PATH = f"output/graphs/{DATASET_NAME}_checkpoint.json"
OUTPUT_PATH = f"output/graphs/{DATASET_NAME}_new.json"

LABEL_TO_LEVEL = {
    "attribute": 1,
    "entity": 2,
    "keyword": 3,
    "community": 4,
}


def load_graph_from_checkpoint(path: str) -> nx.MultiDiGraph:
    """Reconstruct a NetworkX MultiDiGraph from a checkpoint JSON."""
    logger.info(f"Loading checkpoint from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    graph = nx.MultiDiGraph()
    # Use (label, name) as a stable node key
    node_registry: dict[tuple, str] = {}
    node_counter = [0]

    def get_or_create_node(label: str, properties: dict) -> str:
        name = properties.get("name", "")
        key = (label, name)
        if key not in node_registry:
            node_id = f"{label}_{node_counter[0]}"
            node_counter[0] += 1
            level = LABEL_TO_LEVEL.get(label, 2)
            graph.add_node(node_id, label=label, properties=properties, level=level)
            node_registry[key] = node_id
        return node_registry[key]

    for edge in edges:
        u_label = edge["start_node"]["label"]
        u_props = edge["start_node"]["properties"]
        v_label = edge["end_node"]["label"]
        v_props = edge["end_node"]["properties"]
        relation = edge["relation"]

        u_id = get_or_create_node(u_label, u_props)
        v_id = get_or_create_node(v_label, v_props)
        graph.add_edge(u_id, v_id, relation=relation)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    logger.info(f"Graph loaded: {n_nodes:,} nodes, {n_edges:,} edges")
    return graph


def format_output(graph: nx.MultiDiGraph):
    output = []
    for u, v, data in graph.edges(data=True):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        output.append({
            "start_node": {
                "label": u_data["label"],
                "properties": u_data["properties"],
            },
            "relation": data["relation"],
            "end_node": {
                "label": v_data["label"],
                "properties": v_data["properties"],
            },
        })
    return output


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    config = get_config()

    graph = load_graph_from_checkpoint(CHECKPOINT_PATH)

    level2_nodes = [n for n, d in graph.nodes(data=True) if d.get("level") == 2]
    logger.info(f"Level-2 (entity) nodes: {len(level2_nodes):,}")

    logger.info("Initialising FastTreeComm ...")
    start = time.time()
    _tree_comm = tree_comm.FastTreeComm(
        graph,
        embedding_model=config.tree_comm.embedding_model,
        struct_weight=config.tree_comm.struct_weight,
    )

    logger.info("Running community detection ...")
    comm_to_nodes = _tree_comm.detect_communities(level2_nodes)
    logger.info(f"Communities detected: {len(comm_to_nodes)}")

    logger.info("Creating super nodes ...")
    _tree_comm.create_super_nodes_with_keywords(comm_to_nodes, level=4)

    elapsed = time.time() - start
    logger.info(f"Community detection + super nodes: {elapsed:.1f}s")

    logger.info(f"Saving final graph to {OUTPUT_PATH} ...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = format_output(graph)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    logger.info(f"Done. Final graph: {OUTPUT_PATH} ({size_mb:.0f}MB)")


if __name__ == "__main__":
    main()
