"""
fix_chunks.py — Reconstruct output/chunks/{dataset}.txt from the graph + corpus.

The chunk file was not generated for ntsb_full because the process crashed
before save_chunks_to_file() ran. This script reverse-engineers the mapping:
  chunk_id → corpus_document
by matching accident entity names in the graph to corpus document titles.

Usage:
    cd /root/youtu-graphrag
    python scripts/fix_chunks.py --dataset ntsb_sample
    python scripts/fix_chunks.py --dataset ntsb_full
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, '/root/youtu-graphrag')
from config import get_config

def build_chunkid_to_doc(graph_edges, corpus_docs):
    """
    Build chunk_id → corpus_doc mapping using accident entities as the bridge.
    Each entity node has a 'chunk id' property. Accident entities have 'name'
    matching the corpus document title (e.g. 'DFW08RA039' ↔ 'DFW08RA039.md').
    """
    # Build: title_base → corpus_doc (strip .md suffix for matching)
    title_to_doc = {}
    for doc in corpus_docs:
        title = doc.get('title', '')
        base = title.replace('.md', '').strip()
        title_to_doc[base] = doc

    # Build: chunk_id → title_base by scanning accident-schema entities
    chunk_to_title = {}
    chunk_to_any = {}   # fallback: any entity's chunk_id → name (for non-accident docs)
    for edge in graph_edges:
        for side in ('start_node', 'end_node'):
            node = edge[side]
            props = node.get('properties', {})
            chunk_id = props.get('chunk id')
            if not chunk_id:
                continue
            name = props.get('name', '')
            schema_type = props.get('schema_type', '')

            if schema_type == 'accident' and name in title_to_doc:
                chunk_to_title[chunk_id] = name

            # Fallback: store any name for this chunk_id
            if chunk_id not in chunk_to_any:
                chunk_to_any[chunk_id] = name

    # Build: chunk_id → doc
    chunk_to_doc = {}
    for chunk_id, title_base in chunk_to_title.items():
        if title_base in title_to_doc:
            chunk_to_doc[chunk_id] = title_to_doc[title_base]

    # For chunks we couldn't match via accident entities, try fuzzy matching
    unmatched = set(chunk_to_any.keys()) - set(chunk_to_doc.keys())
    if unmatched:
        print(f"  Trying fallback match for {len(unmatched)} unmatched chunk IDs...")
        for chunk_id in unmatched:
            name = chunk_to_any[chunk_id]
            # Try partial match
            for base, doc in title_to_doc.items():
                if name in base or base in name:
                    chunk_to_doc[chunk_id] = doc
                    break

    return chunk_to_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    config = get_config()
    ds_config = config.get_dataset_config(args.dataset)

    graph_path = ds_config.graph_output
    # The construction pipeline saves as _new.json; try that if base path not found
    if not os.path.exists(graph_path):
        graph_path_new = graph_path.replace('.json', '_new.json')
        if os.path.exists(graph_path_new):
            graph_path = graph_path_new
        else:
            # Also try dataset_name_new.json convention used by backend
            graph_path_alt = f"output/graphs/{args.dataset}_new.json"
            if os.path.exists(graph_path_alt):
                graph_path = graph_path_alt
            else:
                print(f"ERROR: Graph not found: {graph_path} (also tried {graph_path_new}, {graph_path_alt})")
                sys.exit(1)

    corpus_path = ds_config.corpus_path
    chunk_file = f"output/chunks/{args.dataset}.txt"
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found: {corpus_path}")
        sys.exit(1)

    print(f"Loading graph: {graph_path} ...")
    with open(graph_path) as f:
        edges = json.load(f)
    print(f"  {len(edges):,} edges loaded")

    print(f"Loading corpus: {corpus_path} ...")
    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"  {len(corpus):,} documents loaded")

    print("Reconstructing chunk_id → document mapping ...")
    chunk_to_doc = build_chunkid_to_doc(edges, corpus)
    print(f"  Matched {len(chunk_to_doc):,} / {len(corpus):,} documents")

    # If existing chunk file has entries not in graph (e.g. partial previous run), merge
    existing = {}
    if os.path.exists(chunk_file):
        print(f"Merging with existing chunk file: {chunk_file}")
        with open(chunk_file) as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2 and parts[0].startswith('id: '):
                        existing[parts[0][4:]] = parts[1][7:] if parts[1].startswith('Chunk: ') else parts[1]
        print(f"  {len(existing):,} existing entries")

    os.makedirs('output/chunks', exist_ok=True)
    written = 0
    with open(chunk_file, 'w', encoding='utf-8') as f:
        # Write matched chunks
        for chunk_id, doc in chunk_to_doc.items():
            f.write(f"id: {chunk_id}\tChunk: {doc}\n")
            written += 1
        # Write any existing entries not already covered
        for chunk_id, text in existing.items():
            if chunk_id not in chunk_to_doc:
                f.write(f"id: {chunk_id}\tChunk: {text}\n")
                written += 1

    size_mb = os.path.getsize(chunk_file) / 1e6
    print(f"✅ Written {written:,} chunks to {chunk_file} ({size_mb:.1f}MB)")
    print(f"   Coverage: {len(chunk_to_doc)/len(corpus)*100:.1f}% of corpus")


if __name__ == '__main__':
    main()
